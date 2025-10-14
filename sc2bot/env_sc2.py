from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer

from .macros import MacroResult, drone_5, expand, ling_rush, overlord, queen_inject
from .telemetry.logging_setup import EventWriter
from .telemetry.schemas import log_episode_end, log_macro_decision, log_tick

MacroCallable = Callable[[BotAI], Awaitable[MacroResult]]


@dataclass(frozen=True)
class MacroSpec:
    name: str
    func: MacroCallable


MACROS: tuple[MacroSpec, ...] = (
    MacroSpec("drone_5", drone_5),
    MacroSpec("overlord", overlord),
    MacroSpec("queen_inject", queen_inject),
    MacroSpec("expand", expand),
    MacroSpec("ling_rush", ling_rush),
)


@dataclass
class StepPayload:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    is_initial: bool = False


class MacroRuntime:
    """Runtime bridge between the Gym interface thread and the SC2 game loop thread."""

    def __init__(
        self,
        env: "SC2MacroEnv",
        episode_id: int,
        *,
        macros: tuple[MacroSpec, ...],
        replay_path: Path,
        max_steps: Optional[int],
    ) -> None:
        self.env = env
        self.episode_id = episode_id
        self.macros = macros
        self.replay_path = replay_path
        self.max_steps = max_steps

        self.pending_actions: queue.Queue[int] = queue.Queue()
        self.results: "queue.Queue[StepPayload]" = queue.Queue()

        self.bot: Optional["MacroBot"] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.episode_end_event: Optional[asyncio.Event] = None
        self.episode_done = threading.Event()

        self.initial_observation_sent = False
        self.prev_metric: Optional[float] = None
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.final_result: Optional[Result] = None
        self.truncated: bool = False
        self.error: Optional[Exception] = None
        self.final_observation: Optional[dict[str, np.ndarray]] = None

        self._lock = threading.Lock()

    def register_bot(self, bot: "MacroBot") -> None:
        self.bot = bot
        self.loop = asyncio.get_running_loop()
        self.episode_end_event = asyncio.Event()

    def enqueue_action(self, action_index: int) -> None:
        self.pending_actions.put(action_index)

    def pop_action_nowait(self) -> Optional[int]:
        try:
            return self.pending_actions.get_nowait()
        except queue.Empty:
            return None

    def push_initial_observation(self, bot: BotAI, iteration: int) -> None:
        if self.initial_observation_sent:
            return
        observation = self.env._build_observation(bot)
        metric = self.env._compute_reward_metric(bot)
        self.prev_metric = metric
        info = {
            "episode_id": self.episode_id,
            "iteration": iteration,
            "game_loop": bot.state.game_loop,
        }
        payload = StepPayload(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            is_initial=True,
        )
        self.results.put(payload)
        self.initial_observation_sent = True

    async def execute_action(self, bot: BotAI, iteration: int, action_index: int) -> None:
        macro_spec = self.macros[action_index]
        macro_result = await macro_spec.func(bot)

        observation = self.env._build_observation(bot)
        metric = self.env._compute_reward_metric(bot)
        reward = 0.0 if self.prev_metric is None else metric - self.prev_metric
        self.prev_metric = metric
        self.total_reward += reward

        truncated = False
        if self.max_steps is not None and (self.step_count + 1) >= self.max_steps:
            truncated = True
            self.truncated = True
            try:
                await bot.client.leave()
            except Exception:
                pass

        info = {
            "episode_id": self.episode_id,
            "step": self.step_count,
            "iteration": iteration,
            "game_loop": bot.state.game_loop,
            "macro": macro_spec.name,
            "macro_success": macro_result.success,
            "macro_latency": macro_result.latency,
            "macro_details": macro_result.details,
        }

        payload = StepPayload(
            observation=observation,
            reward=reward,
            terminated=False,
            truncated=truncated,
            info=info,
            is_initial=False,
        )
        self.results.put(payload)
        self.step_count += 1

    async def handle_episode_end(self, bot: BotAI, result: Result) -> None:
        with self._lock:
            self.final_result = result
            if self.truncated:
                # Prefer truncated flag when we forced a leave.
                pass
        if self.episode_end_event is not None:
            self.episode_end_event.set()

        try:
            self.final_observation = self.env._build_observation(bot)
        except Exception:
            self.final_observation = None
        self.episode_done.set()

    def record_error(self, exc: Exception) -> None:
        self.error = exc
        info = {"episode_id": self.episode_id, "error": repr(exc)}
        payload = StepPayload(
            observation={"vec": np.zeros(64, dtype=np.float32), "minimap": np.zeros((4, 64, 64), dtype=np.float32)},
            reward=0.0,
            terminated=True,
            truncated=True,
            info=info,
            is_initial=False,
        )
        self.results.put(payload)
        if self.episode_end_event is not None:
            self.episode_end_event.set()
        self.episode_done.set()


class MacroBot(BotAI):
    """BotAI wrapper that receives macro commands from the Gym environment."""

    def __init__(self, runtime: MacroRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    async def on_start(self) -> None:
        await super().on_start()
        self.runtime.register_bot(self)

    async def on_step(self, iteration: int) -> None:
        self.runtime.push_initial_observation(self, iteration)
        action_index = self.runtime.pop_action_nowait()
        if action_index is None:
            return
        await self.runtime.execute_action(self, iteration, action_index)

    async def on_end(self, game_result: Result) -> None:
        await self.runtime.handle_episode_end(self, game_result)


class SC2MacroEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment that exposes high-level Zerg macros as discrete actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        map_name: str = "AcropolisLE",
        opponent_race: Race = Race.Terran,
        opponent_difficulty: Difficulty = Difficulty.VeryEasy,
        run_dir: Path | str = Path("data") / "runs",
        max_steps: Optional[int] = None,
        step_timeout: float = 30.0,
        startup_timeout: float = 60.0,
        terminal_wait_timeout: float = 0.05,
    ) -> None:
        super().__init__()
        self.map_name = map_name
        self.opponent_race = opponent_race
        self.opponent_difficulty = opponent_difficulty
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.startup_timeout = startup_timeout
        self.terminal_wait_timeout = terminal_wait_timeout

        self.action_space = spaces.Discrete(len(MACROS))
        self.observation_space = spaces.Dict(
            {
                "vec": spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32),
                "minimap": spaces.Box(low=0.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            }
        )

        run_root = Path(run_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = run_root / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir = self.run_dir / "replays"
        self.replay_dir.mkdir(parents=True, exist_ok=True)

        self._writer = EventWriter(self.run_dir)

        self._runtime: Optional[MacroRuntime] = None
        self._game_thread: Optional[threading.Thread] = None
        self._closed = False
        self._episode_id = 0
        self._seed: Optional[int] = None

        self._minimap_scale_x: float = 1.0
        self._minimap_scale_y: float = 1.0

    # Gym API -----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        if self._runtime is not None:
            self._shutdown_runtime()

        self._episode_id += 1
        replay_path = self.replay_dir / f"episode_{self._episode_id:05d}.SC2Replay"
        runtime = MacroRuntime(
            env=self,
            episode_id=self._episode_id,
            macros=MACROS,
            replay_path=replay_path,
            max_steps=self.max_steps,
        )
        self._runtime = runtime
        self._writer.start_episode(self._episode_id)

        self._game_thread = threading.Thread(
            target=self._run_game,
            args=(runtime,),
            name=f"SC2GameThread-episode-{self._episode_id}",
            daemon=True,
        )
        self._game_thread.start()

        payload = self._wait_for_initial_observation(runtime)
        observation = payload.observation
        info = payload.info | {"episode_id": self._episode_id}
        self._log_tick(payload)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._runtime is None:
            raise RuntimeError("Environment not reset before calling step().")
        runtime = self._runtime

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the discrete action space.")

        runtime.enqueue_action(int(action))
        payload = self._wait_for_step_payload(runtime)
        observation = payload.observation
        reward = float(payload.reward)
        terminated = bool(payload.terminated)
        truncated = bool(payload.truncated)
        info = dict(payload.info)

        macro_name = info.get("macro")
        if macro_name is not None:
            self._writer.write_event(
                log_macro_decision(
                    name=macro_name,
                    priority=None,
                    extra={
                        "episode_id": runtime.episode_id,
                        "step": info.get("step"),
                        "success": info.get("macro_success"),
                        "latency": info.get("macro_latency"),
                        "details": info.get("macro_details"),
                    },
                )
            )

        self._log_tick(payload)

        if terminated or truncated:
            self._log_episode_end(runtime, payload)

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._shutdown_runtime()
        self._writer.close()
        self._closed = True
        super().close()

    # Internal helpers --------------------------------------------------------
    def _run_game(self, runtime: MacroRuntime) -> None:
        try:
            map_obj = maps.get(self.map_name)
            players = [
                Bot(Race.Zerg, MacroBot(runtime)),
                Computer(self.opponent_race, self.opponent_difficulty),
            ]
            run_game(
                map_obj,
                players,
                realtime=False,
                save_replay_path=str(runtime.replay_path),
                random_seed=self._seed,
            )
        except Exception as exc:
            runtime.record_error(exc)

    def _wait_for_initial_observation(self, runtime: MacroRuntime) -> StepPayload:
        try:
            payload = runtime.results.get(timeout=self.startup_timeout)
        except queue.Empty as exc:
            raise RuntimeError("Timed out waiting for initial observation from SC2.") from exc
        if not payload.is_initial:
            # Put back and raise for clarity.
            runtime.results.put(payload)
            raise RuntimeError("Received non-initial payload while waiting for reset observation.")
        self._update_minimap_scale(runtime)
        return payload

    def _wait_for_step_payload(self, runtime: MacroRuntime) -> StepPayload:
        deadline = time.time() + self.step_timeout
        payload: Optional[StepPayload] = None
        while time.time() < deadline:
            timeout = max(0.0, deadline - time.time())
            try:
                payload = runtime.results.get(timeout=timeout)
                break
            except queue.Empty:
                continue
        if payload is None:
            raise TimeoutError("Timed out waiting for step result from SC2.")

        payload = self._drain_additional_payloads(runtime, payload)

        if not runtime.episode_done.is_set() and self.terminal_wait_timeout > 0:
            runtime.episode_done.wait(timeout=self.terminal_wait_timeout)
            payload = self._drain_additional_payloads(runtime, payload)

        if runtime.episode_done.is_set():
            payload.terminated = True
            if runtime.truncated:
                payload.truncated = True
            if runtime.final_result is not None:
                payload.info.setdefault("episode_result", runtime.final_result.name)
            if runtime.final_observation is not None:
                payload.observation = runtime.final_observation
        return payload

    def _shutdown_runtime(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is None:
            return
        if runtime.bot is not None and runtime.loop is not None:
            async def _leave() -> None:
                try:
                    await runtime.bot.client.leave()
                except Exception:
                    pass

            future = asyncio.run_coroutine_threadsafe(_leave(), runtime.loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                pass
        if self._game_thread is not None:
            self._game_thread.join(timeout=15.0)
            self._game_thread = None

    def _update_minimap_scale(self, runtime: MacroRuntime) -> None:
        if runtime.bot is None:
            return
        map_size = runtime.bot.game_info.map_size
        width = max(1.0, float(map_size[0]))
        height = max(1.0, float(map_size[1]))
        self._minimap_scale_x = (64 - 1) / width
        self._minimap_scale_y = (64 - 1) / height

    def _build_observation(self, bot: BotAI) -> dict[str, np.ndarray]:
        vec = np.zeros(64, dtype=np.float32)
        vec[0] = float(bot.state.game_loop)
        vec[1] = float(bot.minerals)
        vec[2] = float(bot.vespene)
        vec[3] = float(bot.supply_cap)
        vec[4] = float(bot.supply_used)
        vec[5] = float(bot.supply_left)
        vec[6] = float(bot.workers.amount)
        vec[7] = float(bot.army_count)
        vec[8] = float(bot.structures.amount)
        vec[9] = float(bot.units.amount)
        vec[10] = float(
            bot.already_pending(UnitTypeId.DRONE)
            + bot.already_pending(UnitTypeId.OVERLORD)
            + bot.already_pending(UnitTypeId.HATCHERY)
        )
        vec[11] = float(bot.larva.amount)
        vec[12] = float(bot.townhalls.amount)
        vec[13] = float(bot.state.score.score)
        vec[14] = float(bot.state.score.killed_minerals_army)
        vec[15] = float(bot.state.score.lost_minerals_army)

        minimap = np.zeros((4, 64, 64), dtype=np.float32)
        for structure in bot.townhalls:
            x, y = self._minimap_coords(structure.position)
            minimap[0, y, x] = 1.0
        for unit in bot.units:
            x, y = self._minimap_coords(unit.position)
            minimap[1, y, x] = 1.0
        for enemy in bot.known_enemy_units:
            x, y = self._minimap_coords(enemy.position)
            minimap[2, y, x] = 1.0
        creep_map = bot.state.creep.data_numpy
        creep_height, creep_width = creep_map.shape
        y_bins = np.linspace(0, creep_height, 65, dtype=int)
        x_bins = np.linspace(0, creep_width, 65, dtype=int)
        for y in range(64):
            for x in range(64):
                patch = creep_map[y_bins[y] : y_bins[y + 1], x_bins[x] : x_bins[x + 1]]
                if patch.size and np.any(patch):
                    minimap[3, y, x] = 1.0

        return {"vec": vec, "minimap": minimap}

    def _drain_additional_payloads(self, runtime: MacroRuntime, base_payload: StepPayload) -> StepPayload:
        payload = base_payload
        while True:
            try:
                next_payload = runtime.results.get_nowait()
            except queue.Empty:
                break
            payload = self._merge_payloads(payload, next_payload)
        return payload

    def _merge_payloads(self, current: StepPayload, incoming: StepPayload) -> StepPayload:
        if incoming.is_initial:
            return incoming
        if incoming.terminated or incoming.truncated:
            incoming.reward += current.reward
            merged_info = dict(current.info)
            merged_info.update(incoming.info)
            incoming.info = merged_info
            if "macro" not in incoming.info and current.info.get("macro"):
                incoming.info["macro"] = current.info["macro"]
                incoming.info["macro_success"] = current.info.get("macro_success")
                incoming.info["macro_latency"] = current.info.get("macro_latency")
                incoming.info["macro_details"] = current.info.get("macro_details")
            if incoming.observation is None:
                incoming.observation = current.observation
            return incoming
        return incoming

    def _minimap_coords(self, position) -> tuple[int, int]:
        x = int(np.clip(position.x * self._minimap_scale_x, 0, 63))
        y = int(np.clip(position.y * self._minimap_scale_y, 0, 63))
        return x, y

    def _compute_reward_metric(self, bot: BotAI) -> float:
        score = bot.state.score
        return float(score.score) + 0.1 * float(bot.minerals + bot.vespene)

    def _log_tick(self, payload: StepPayload) -> None:
        observation = payload.observation
        vec_snapshot = observation["vec"][:8].tolist()
        minimap = observation["minimap"]
        minimap_stats = {
            "mean": float(minimap.mean()),
            "max": float(minimap.max()),
        }
        info = dict(payload.info)
        actions: list[str] = []
        macro_name = info.get("macro")
        if macro_name:
            actions.append(str(macro_name))
        self._writer.write_event(
            log_tick(
                step=int(info.get("step", 0)),
                game_loop=int(info.get("game_loop", 0)),
                reward=float(payload.reward),
                actions=actions or None,
                observation={"vec": vec_snapshot, "minimap_stats": minimap_stats},
                extra={"episode_id": info.get("episode_id")},
            )
        )

    def _log_episode_end(self, runtime: MacroRuntime, payload: StepPayload) -> None:
        result_name = payload.info.get("episode_result", "unknown")
        self._writer.write_event(
            log_episode_end(
                episode_id=runtime.episode_id,
                result=result_name,
                total_reward=float(runtime.total_reward),
                frames=int(payload.info.get("game_loop", 0)),
            )
        )
