from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sc2.data import Difficulty, Race

from sc2bot.agent.scripted_zerg import ScriptedZergAgent
from sc2bot.env_sc2 import SC2MacroEnv
from sc2bot.telemetry.logging_setup import EventWriter
from sc2bot.telemetry.overlay import LiveStats
from sc2bot.telemetry.schemas import log_episode_end
from sc2bot.utils.paths import RUN_CONFIG_PATH, ensure_runs_dir

DEFAULT_DECISION_INTERVAL = 8
DEFAULT_OVERLAY_REFRESH_HZ = 2.0
DECISION_SLEEP_CAP_SECONDS = 1.0


@dataclass
class RunnerConfig:
    map_name: str
    opponent_race: Race
    opponent_difficulty: Difficulty
    seed: int | None
    decision_interval_loops: int
    overlay_refresh_hz: float
    max_steps: int | None
    step_timeout: float | None
    startup_timeout: float | None
    terminal_wait_timeout: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a single SC2 macro proof-of-concept episode and feed live telemetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=RUN_CONFIG_PATH, help="YAML configuration file to load.")
    parser.add_argument("--map", type=str, help="Override the battleground defined in the config file.")
    parser.add_argument("--difficulty", type=str, help="Override the opponent difficulty (name or integer).")
    parser.add_argument("--opponent-race", type=str, help="Override the opponent race.")
    parser.add_argument("--seed", type=int, help="Override the RNG seed.")
    parser.add_argument("--decision-interval", type=int, help="Macro decision cadence in game loops.")
    parser.add_argument("--max-steps", type=int, help="Maximum number of macro steps before truncation.")
    parser.add_argument("--step-timeout", type=float, help="Environment step timeout in seconds.")
    parser.add_argument("--startup-timeout", type=float, help="Environment startup timeout in seconds.")
    parser.add_argument("--terminal-wait-timeout", type=float, help="Environment terminal wait timeout in seconds.")
    parser.add_argument("--overlay-refresh-hz", type=float, help="Refresh rate for OBS overlay metrics.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level (e.g. INFO, DEBUG).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        log_level = _parse_log_level(args.log_level)
    except ValueError as exc:
        print(f"Invalid log level: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("run_poc")

    try:
        raw_config = _load_run_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load configuration from %s: %s", args.config, exc)
        return 1
    try:
        config = _build_config(args, raw_config)
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    env: SC2MacroEnv | None = None
    writer: EventWriter | None = None
    live_stats: LiveStats | None = None

    try:
        env_kwargs: dict[str, Any] = {
            "map_name": config.map_name,
            "opponent_race": config.opponent_race,
            "opponent_difficulty": config.opponent_difficulty,
            "run_dir": ensure_runs_dir(),
        }
        if config.max_steps is not None:
            env_kwargs["max_steps"] = int(config.max_steps)
        if config.step_timeout is not None:
            env_kwargs["step_timeout"] = float(config.step_timeout)
        if config.startup_timeout is not None:
            env_kwargs["startup_timeout"] = float(config.startup_timeout)
        if config.terminal_wait_timeout is not None:
            env_kwargs["terminal_wait_timeout"] = float(config.terminal_wait_timeout)

        env = SC2MacroEnv(**env_kwargs)
        agent = ScriptedZergAgent(
            decision_interval_loops=config.decision_interval_loops,
            seed=config.seed,
        )

        observation, info = env.reset(seed=config.seed)
        agent.reset(observation, info)

        run_dir = env.run_dir
        writer = EventWriter(run_dir / "runner")
        episode_id = int(info.get("episode_id", 0))
        writer.start_episode(episode_id)

        live_stats = LiveStats(run_dir, refresh_hz=config.overlay_refresh_hz)
        macro_counts: dict[str, int] = {name: 0 for name in agent.macro_names}
        total_reward = 0.0
        decision_summary: dict[str, Any] | None = None
        start_time = time.time()

        live_stats.update(
            _build_stats_payload(
                observation,
                info,
                reward=0.0,
                total_reward=total_reward,
                macro_counts=macro_counts,
                decision_summary=None,
                status="starting",
            ),
            force=True,
        )

        terminated = False
        truncated = False

        while not (terminated or truncated):
            sleep_seconds = agent.cooldown_seconds(observation, info)
            if sleep_seconds > 0:
                time.sleep(min(sleep_seconds, DECISION_SLEEP_CAP_SECONDS))

            action, decision_summary = agent.select_action(observation, info)
            logger.info(
                "loop=%s step=%s macro=%s reason=%s stage=%s",
                info.get("game_loop"),
                info.get("step"),
                decision_summary.get("macro"),
                decision_summary.get("reason"),
                decision_summary.get("stage"),
            )

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            macro_name = info.get("macro")
            if macro_name:
                macro_counts[macro_name] = macro_counts.get(macro_name, 0) + 1

            live_stats.update(
                _build_stats_payload(
                    observation,
                    info,
                    reward=reward,
                    total_reward=total_reward,
                    macro_counts=macro_counts,
                    decision_summary=decision_summary,
                    status="running",
                )
            )

        result = info.get("episode_result", "unknown")
        duration = time.time() - start_time
        live_stats.update(
            _build_stats_payload(
                observation,
                info,
                reward=0.0,
                total_reward=total_reward,
                macro_counts=macro_counts,
                decision_summary=decision_summary,
                status="finished",
                extra={
                    "result": result,
                    "terminated": terminated,
                    "truncated": truncated,
                    "duration_seconds": duration,
                },
            ),
            force=True,
        )

        writer.write_event(
            log_episode_end(
                episode_id=episode_id,
                result=result,
                total_reward=float(total_reward),
                frames=int(info.get("game_loop", 0) or 0),
                duration_seconds=duration,
            )
        )
        writer.flush()
        live_stats.flush()
        logger.info(
            "Episode %s finished with result=%s reward=%.2f duration=%.1fs",
            episode_id,
            result,
            total_reward,
            duration,
        )
        return 0
    except KeyboardInterrupt:
        logging.getLogger("run_poc").warning("Interrupted by user.")
        return 130
    finally:
        if live_stats is not None:
            live_stats.flush()
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def _build_config(args: argparse.Namespace, raw: Mapping[str, Any]) -> RunnerConfig:
    map_name = str(args.map or raw.get("map") or "AcropolisLE")
    difficulty_value = args.difficulty or raw.get("ai_difficulty")
    opponent_race_value = args.opponent_race or raw.get("opponent_race") or raw.get("enemy_race") or "Terran"
    seed_value = args.seed if args.seed is not None else raw.get("seed")

    overlay_refresh_hz = args.overlay_refresh_hz or raw.get("overlay_refresh_hz", DEFAULT_OVERLAY_REFRESH_HZ)
    decision_interval = args.decision_interval or raw.get("decision_interval_loops") or raw.get(
        "decision_interval", DEFAULT_DECISION_INTERVAL
    )
    max_steps = args.max_steps if args.max_steps is not None else raw.get("max_steps")
    step_timeout = args.step_timeout if args.step_timeout is not None else raw.get("step_timeout")
    startup_timeout = args.startup_timeout if args.startup_timeout is not None else raw.get("startup_timeout")
    terminal_wait_timeout = (
        args.terminal_wait_timeout if args.terminal_wait_timeout is not None else raw.get("terminal_wait_timeout")
    )

    difficulty = _parse_difficulty(difficulty_value)
    opponent_race = _parse_race(opponent_race_value)
    seed = int(seed_value) if seed_value is not None else None

    try:
        decision_interval_loops = max(1, int(decision_interval))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid decision interval: {decision_interval!r}") from None

    return RunnerConfig(
        map_name=map_name,
        opponent_race=opponent_race,
        opponent_difficulty=difficulty,
        seed=seed,
        decision_interval_loops=decision_interval_loops,
        overlay_refresh_hz=float(overlay_refresh_hz),
        max_steps=int(max_steps) if max_steps is not None else None,
        step_timeout=float(step_timeout) if step_timeout is not None else None,
        startup_timeout=float(startup_timeout) if startup_timeout is not None else None,
        terminal_wait_timeout=float(terminal_wait_timeout) if terminal_wait_timeout is not None else None,
    )


def _load_run_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text)  # type: ignore[no-untyped-call]
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("Run configuration must be a mapping at the top level.")
    return dict(data)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not value:
            result[key] = None
            continue
        if value.lower() in {"null", "none"}:
            result[key] = None
            continue
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
            continue
        if value.startswith(("'", '"')) and value.endswith(value[0]):
            result[key] = value[1:-1]
            continue
        try:
            result[key] = int(value)
            continue
        except ValueError:
            try:
                result[key] = float(value)
                continue
            except ValueError:
                result[key] = value
    return result


def _parse_difficulty(value: Any) -> Difficulty:
    if isinstance(value, Difficulty):
        return value
    if value is None:
        return Difficulty.VeryEasy
    if isinstance(value, int):
        return Difficulty(value)
    text = str(value).strip()
    if not text:
        return Difficulty.VeryEasy
    if text.isdigit():
        return Difficulty(int(text))
    key = text.upper()
    if key in Difficulty.__members__:
        return Difficulty[key]
    raise ValueError(f"Unknown difficulty value: {value!r}")


def _parse_race(value: Any) -> Race:
    if isinstance(value, Race):
        return value
    if value is None:
        return Race.Terran
    text = str(value).strip()
    if not text:
        return Race.Terran
    key = text.upper()
    if key in Race.__members__:
        return Race[key]
    raise ValueError(f"Unknown race value: {value!r}")


def _parse_log_level(value: str) -> int:
    try:
        return getattr(logging, value.upper())
    except AttributeError:
        raise ValueError(f"Unsupported log level: {value}") from None


def _build_stats_payload(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    reward: float,
    total_reward: float,
    macro_counts: Mapping[str, int],
    decision_summary: Mapping[str, Any] | None,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vec = observation.get("vec") if observation is not None else None
    payload: dict[str, Any] = {
        "status": status,
        "timestamp": time.time(),
        "episode_id": info.get("episode_id"),
        "step": info.get("step"),
        "game_loop": info.get("game_loop") if info.get("game_loop") is not None else _vec_value(vec, 0, 0.0),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "macro": info.get("macro"),
        "macro_success": info.get("macro_success"),
        "macro_latency": info.get("macro_latency"),
        "macro_details": info.get("macro_details"),
        "macro_counts": dict(macro_counts),
        "decision": dict(decision_summary) if decision_summary else None,
        "resources": {
            "minerals": _vec_value(vec, 1, 0.0),
            "vespene": _vec_value(vec, 2, 0.0),
            "supply_cap": _vec_value(vec, 3, 0.0),
            "supply_used": _vec_value(vec, 4, 0.0),
            "supply_left": _vec_value(vec, 5, 0.0),
            "workers": _vec_value(vec, 6, 0.0),
            "army": _vec_value(vec, 7, 0.0),
            "townhalls": _vec_value(vec, 12, 0.0),
            "score": _vec_value(vec, 13, 0.0),
        },
        "flags": {
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
        },
    }
    if "episode_result" in info:
        payload["episode_result"] = info["episode_result"]
    if extra:
        payload.update(dict(extra))
    return payload


def _vec_value(vec: Any, index: int, default: float) -> float:
    if vec is None:
        return default
    try:
        return float(vec[index])
    except (TypeError, ValueError, IndexError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
