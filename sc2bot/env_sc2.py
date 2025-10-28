from __future__ import annotations

import ast
import logging
import os
import queue
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer

from sc2bot.utils import paths
from sc2bot.utils.seeds import set_global_seed

__all__ = ["RunConfig", "StarCraftIIGymEnv"]

logger = logging.getLogger(__name__)

_STEP_QUEUE_SENTINEL = object()

# Mapping integers to python-sc2 difficulty enums for simple numeric configs.
_DIFFICULTY_BY_INT: Dict[int, Difficulty] = {
    0: Difficulty.VeryEasy,
    1: Difficulty.VeryEasy,
    2: Difficulty.Easy,
    3: Difficulty.Medium,
    4: Difficulty.MediumHard,
    5: Difficulty.Hard,
    6: Difficulty.Harder,
    7: Difficulty.VeryHard,
    8: Difficulty.CheatVision,
    9: Difficulty.CheatMoney,
    10: Difficulty.CheatInsane,
}

_RACE_LOOKUP: Dict[str, Race] = {
    "random": Race.Random,
    "terran": Race.Terran,
    "protoss": Race.Protoss,
    "zerg": Race.Zerg,
}


@dataclass(frozen=True)
class RunConfig:
    """Container holding runtime configuration for SC2 sessions."""

    race: Race
    map_name: str
    ai_difficulty: Difficulty
    seed: Optional[int] = None
    opponent_race: Race = Race.Protoss
    realtime: bool = False
    save_replay: bool = True
    replay_prefix: str = "episode"

    def with_overrides(self, **overrides: Any) -> "RunConfig":
        return replace(self, **overrides)


@dataclass
class _StepPayload:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StarCraftIIGymEnv(gym.Env[np.ndarray, int]):
    """
    A Gymnasium environment wrapper around python-sc2 games.

    The environment launches a StarCraft II session in a background thread and
    streams simple numerical observations to the agent, alongside a small action
    space intended for experimentation. Configuration is sourced from
    configs/run.yaml by default and can be overridden at runtime.
    """

    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._config_path = Path(config_path) if config_path else paths.RUN_CONFIG_PATH
        self._base_config = _load_run_config(self._config_path)
        self._render_mode = render_mode
        self._episode_active = False
        self._episode_index = 0
        self._last_seed: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_exception: Optional[BaseException] = None
        self._bot: Optional[_GymControllerBot] = None
        self._action_queue: queue.Queue[int | None] = queue.Queue()
        self._step_queue: queue.Queue[_StepPayload | object] = queue.Queue()
        self._last_iteration: Optional[int] = None

        self._runs_dir = paths.ensure_runs_dir()
        self._replays_dir = paths.ensure_replays_dir()

        # Resolve SC2 installation directory if possible so python-sc2 can find it.
        try:
            sc2_dir = paths.resolve_sc2_dir(ensure_exists=False)
            if sc2_dir:
                os.environ.setdefault("SC2PATH", str(sc2_dir))
        except FileNotFoundError:
            logger.warning(
                "StarCraft II directory could not be resolved; relying on python-sc2 defaults."
            )

        # Observation: [minerals, vespene, supply_used, supply_cap, workers, army, larva]
        obs_high = np.array([10000, 10000, 200, 200, 200, 400, 100], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros_like(obs_high),
            high=obs_high,
            dtype=np.float32,
        )

        # Actions: 0 noop, 1 build drone, 2 build overlord, 3 build spawning pool.
        self.action_space = spaces.Discrete(4)

    # --------------------------------------------------------------------- API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        options = options or {}
        self.close()

        config = self._prepare_run_config(options)
        self._episode_index += 1
        self._thread_exception = None
        self._action_queue = queue.Queue()
        self._step_queue = queue.Queue()

        resolved_seed = seed if seed is not None else config.seed
        self._last_seed = set_global_seed(resolved_seed)

        bot = _GymControllerBot(
            action_queue=self._action_queue,
            step_queue=self._step_queue,
        )

        self._bot = bot
        self._episode_active = True
        self._thread = threading.Thread(
            target=self._run_game_thread,
            name=f"SC2Env-{self._episode_index}",
            args=(config, bot),
            daemon=True,
        )
        self._thread.start()

        payload = self._wait_for_step_data()
        info = dict(payload.info)
        info.update(
            {
                "seed": self._last_seed,
                "episode": self._episode_index,
                "config_path": str(self._config_path),
            }
        )
        iteration = payload.info.get("iteration")
        if isinstance(iteration, int):
            self._last_iteration = iteration
        else:
            self._last_iteration = None
        return payload.observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self._episode_active:
            raise ResetNeeded("Environment reset required before stepping.")
        if not self._thread or not self._thread.is_alive():
            self._raise_thread_exception()
            raise RuntimeError("StarCraft II game thread is not running.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._action_queue.put(action)
        min_iteration = self._last_iteration + 1 if self._last_iteration is not None else None
        payload = self._wait_for_step_data(min_iteration=min_iteration)
        iteration = payload.info.get("iteration")
        if isinstance(iteration, int):
            self._last_iteration = iteration
        if payload.terminated or payload.truncated:
            self._episode_active = False
        return (
            payload.observation,
            payload.reward,
            payload.terminated,
            payload.truncated,
            payload.info,
        )

    def close(self) -> None:
        if self._bot is not None:
            self._bot.request_stop()
        if self._thread and self._thread.is_alive():
            try:
                self._action_queue.put_nowait(None)
            except queue.Full:
                self._action_queue.put(None)
            self._thread.join(timeout=15)
        self._thread = None
        self._bot = None
        self._episode_active = False
        self._last_iteration = None

    # ----------------------------------------------------------------- Helpers
    def _run_game_thread(self, config: RunConfig, bot: "_GymControllerBot") -> None:
        map_obj = maps.get(config.map_name)
        replay_path = None
        if config.save_replay:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{config.replay_prefix}_{self._episode_index}_{timestamp}.SC2Replay"
            replay_path = str(self._replays_dir / filename)

        try:
            with _suppress_non_main_thread_signals():
                run_game(
                    map_obj,
                    [
                        Bot(config.race, bot),
                        Computer(config.opponent_race, config.ai_difficulty),
                    ],
                    realtime=config.realtime,
                    save_replay_as=replay_path,
                )
        except Exception as exc:
            logger.exception("StarCraft II run failed: %s", exc)
            self._thread_exception = exc
        finally:
            bot.request_stop()
            self._step_queue.put(_STEP_QUEUE_SENTINEL)

    def _prepare_run_config(self, options: Dict[str, Any]) -> RunConfig:
        config_override_path = options.get("config_path")
        if config_override_path:
            self._config_path = Path(config_override_path)
            base_config = _load_run_config(self._config_path)
        else:
            base_config = self._base_config

        overrides = options.get("config_overrides")
        if overrides:
            normalized: Dict[str, Any] = {}
            for key, value in overrides.items():
                if key == "race":
                    normalized[key] = _parse_race(value)
                elif key in {"opponent_race", "enemy_race"}:
                    normalized["opponent_race"] = _parse_race(value)
                elif key in {"ai_difficulty", "difficulty"}:
                    normalized["ai_difficulty"] = _parse_difficulty(value)
                else:
                    normalized[key] = value
            config = base_config.with_overrides(**normalized)
        else:
            config = base_config
        return config

    def _wait_for_step_data(self, min_iteration: Optional[int] = None) -> _StepPayload:
        while True:
            self._raise_thread_exception()
            try:
                payload = self._step_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if payload is _STEP_QUEUE_SENTINEL:
                self._raise_thread_exception()
                raise RuntimeError("Game finished without emitting further step data.")
            if not isinstance(payload, _StepPayload):
                continue
            iteration = payload.info.get("iteration")
            if (
                min_iteration is not None
                and isinstance(iteration, int)
                and iteration < min_iteration
                and not (payload.terminated or payload.truncated)
            ):
                continue
            return payload

    def _raise_thread_exception(self) -> None:
        if self._thread_exception is not None:
            exc = self._thread_exception
            self._thread_exception = None
            raise RuntimeError("StarCraft II thread failed") from exc


class _GymControllerBot(BotAI):
    """
    Custom python-sc2 bot that communicates with the Gym environment via queues.

    The bot keeps a small set of macro-oriented actions and reports compact
    observations containing basic economic state. Rewards are simple dense
    deltas based on accumulated resources, unit counts, and structures.
    """

    def __init__(
        self,
        action_queue: "queue.Queue[int | None]",
        step_queue: "queue.Queue[_StepPayload | object]",
    ) -> None:
        super().__init__()
        self._action_queue = action_queue
        self._step_queue = step_queue
        self._current_action = 0
        self._episode_over = threading.Event()
        self._last_score = 0.0
        self._last_iteration = 0
        self._abort_requested = False
        self._abort_payload_sent = False
        self._leave_sent = False

    def request_stop(self) -> None:
        self._abort_requested = True
        self._episode_over.set()

    async def on_step(self, iteration: int) -> None:
        self._last_iteration = iteration
        if self._episode_over.is_set():
            await self._handle_abort(iteration)
            return

        self._drain_actions()
        if self._episode_over.is_set():
            await self._handle_abort(iteration)
            return

        await self._execute_action(self._current_action)

        observation = self._gather_observation()
        score = self._resource_score(observation)
        reward = score - self._last_score
        self._last_score = score

        payload = _StepPayload(
            observation=observation,
            reward=float(reward),
            terminated=False,
            truncated=False,
            info={
                "iteration": iteration,
                "action": int(self._current_action),
            },
        )
        self._step_queue.put(payload)

    async def on_end(self, result: Result) -> None:
        if self._episode_over.is_set():
            return
        self._episode_over.set()
        observation = self._gather_observation()
        outcome_reward = 0.0
        if result == Result.Victory:
            outcome_reward = 1000.0
        elif result == Result.Defeat:
            outcome_reward = -1000.0

        payload = _StepPayload(
            observation=observation,
            reward=float(outcome_reward),
            terminated=True,
            truncated=False,
            info={
                "iteration": self._last_iteration,
                "result": result.name.lower(),
            },
        )
        self._step_queue.put(payload)

    # ------------------------------------------------------------- Bot helpers
    def _drain_actions(self) -> None:
        try:
            while True:
                next_action = self._action_queue.get_nowait()
                if next_action is None:
                    self._abort_requested = True
                    self._episode_over.set()
                    return
                self._current_action = int(next_action)
        except queue.Empty:
            return

    async def _handle_abort(self, iteration: int) -> None:
        if self._abort_requested and hasattr(self, "client") and not self._leave_sent:
            try:
                await self.client.leave()
            except Exception as exc:
                logger.debug("Failed to leave game during abort: %s", exc)
            finally:
                self._leave_sent = True
        if self._abort_requested and not self._abort_payload_sent:
            observation = self._gather_observation()
            payload = _StepPayload(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=True,
                info={
                    "iteration": iteration,
                    "aborted": True,
                },
            )
            self._step_queue.put(payload)
            self._abort_payload_sent = True

    async def _execute_action(self, action: int) -> None:
        if action == 0:
            return
        if action == 1:
            self._train_unit(UnitTypeId.DRONE)
            return
        if action == 2:
            self._train_unit(UnitTypeId.OVERLORD)
            return
        if action == 3:
            await self._build_structure(UnitTypeId.SPAWNINGPOOL)

    def _train_unit(self, unit_type: UnitTypeId) -> None:
        if not self.can_afford(unit_type):
            return
        if not self.larva:
            return
        larva = self.larva.random
        larva.train(unit_type)

    async def _build_structure(self, structure_type: UnitTypeId) -> None:
        if self.structures(structure_type).amount > 0:
            return
        if not self.can_afford(structure_type):
            return
        try:
            await self.build(
                structure_type,
                near=self.start_location.towards(self.game_info.map_center, 6),
            )
        except Exception:
            # If the build fails (e.g., no placement position), we ignore the error.
            return

    def _resource_score(self, observation: np.ndarray) -> float:
        minerals, vespene, supply_used, _, workers, army, _ = observation
        return (
            float(minerals)
            + float(vespene)
            + 5.0 * float(workers)
            + 10.0 * float(army)
            + 0.5 * float(supply_used)
        )

    def _gather_observation(self) -> np.ndarray:
        minerals = float(self.minerals)
        vespene = float(self.vespene)
        supply_used = float(self.supply_used)
        supply_cap = float(self.supply_cap)
        workers = float(self.workers.amount)
        army = float(
            sum(
                1
                for unit in self.units
                if not unit.is_structure
                and unit.type_id
                not in {
                    UnitTypeId.DRONE,
                    UnitTypeId.OVERLORD,
                    UnitTypeId.OVERSEER,
                }
            )
        )
        larva = float(self.larva.amount)

        observation = np.array(
            [
                minerals,
                vespene,
                supply_used,
                supply_cap,
                workers,
                army,
                larva,
            ],
            dtype=np.float32,
        )
        return observation


# ---------------------------------------------------------------- Utilities
def _load_run_config(config_path: Path) -> RunConfig:
    config_dict = _read_yaml_dict(config_path)
    if not config_dict:
        raise ValueError(f"Run configuration at {config_path} is empty.")

    race = _parse_race(config_dict.get("race", "zerg"))
    map_name = str(config_dict.get("map", "AcropolisLE"))
    ai_difficulty = _parse_difficulty(config_dict.get("ai_difficulty", "medium"))
    seed = config_dict.get("seed")
    if isinstance(seed, float):
        seed = int(seed)
    elif seed is not None and not isinstance(seed, int):
        raise TypeError(f"Seed must be int or None, got {type(seed)!r}")

    opponent_race = _parse_race(config_dict.get("opponent_race", "protoss"))
    realtime = bool(config_dict.get("realtime", False))
    save_replay = bool(config_dict.get("save_replay", True))
    replay_prefix = str(config_dict.get("replay_prefix", "episode"))

    return RunConfig(
        race=race,
        map_name=map_name,
        ai_difficulty=ai_difficulty,
        seed=seed,
        opponent_race=opponent_race,
        realtime=realtime,
        save_replay=save_replay,
        replay_prefix=replay_prefix,
    )


def _read_yaml_dict(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import]
    except ModuleNotFoundError:
        return _fallback_parse_yaml(text)
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration in {path} must be a mapping.")
    return data


def _fallback_parse_yaml(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value_str = line.partition(":")
        key = key.strip()
        value_str = value_str.strip()
        if not value_str:
            result[key] = None
            continue
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str.strip('"').strip("'")
        result[key] = value
    return result


def _parse_race(value: Any) -> Race:
    if isinstance(value, Race):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _RACE_LOOKUP:
            return _RACE_LOOKUP[key]
    raise ValueError(f"Unknown race value: {value!r}")


def _parse_difficulty(value: Any) -> Difficulty:
    if isinstance(value, Difficulty):
        return value
    if isinstance(value, int):
        if value in _DIFFICULTY_BY_INT:
            return _DIFFICULTY_BY_INT[value]
        raise ValueError(f"Unsupported difficulty index: {value}")
    if isinstance(value, str):
        key = value.strip().replace(" ", "_").replace("-", "_").upper()
        try:
            return Difficulty[key]
        except KeyError as exc:
            raise ValueError(f"Unknown difficulty: {value!r}") from exc
    raise TypeError(f"Difficulty must be int, str, or Difficulty enum, got {type(value)!r}")


@contextmanager
def _suppress_non_main_thread_signals():
    """
    Temporarily ignore signal handler registration when invoked off the main thread.

    python-sc2 installs SIGINT handlers as part of launching the SC2 client, but
    Python's signal module forbids registering handlers outside the main thread.
    When running games from within a Gym environment thread we shim the handler
    registration to avoid raising ValueError while keeping Ctrl+C behaviour intact
    for the main interpreter thread.
    """

    if threading.current_thread() is threading.main_thread():
        yield
        return

    previous_signal = signal.signal
    saved_handlers: Dict[int, Any] = {}

    def _noop_signal(signum, handler):
        logger.debug(
            "Skipping signal handler registration for signum %s on non-main thread",
            signum,
        )
        previous = saved_handlers.get(signum, handler)
        saved_handlers[signum] = handler
        return previous

    signal.signal = _noop_signal
    try:
        yield
    finally:
        signal.signal = previous_signal
