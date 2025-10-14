from __future__ import annotations

import time
from typing import Any, Mapping, MutableMapping, Sequence


def log_tick(
    *,
    step: int,
    game_loop: int,
    timestamp: float | None = None,
    reward: float | None = None,
    actions: Sequence[str] | None = None,
    observation: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    extra: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured payload for a per-tick event."""
    payload: dict[str, Any] = {
        "type": "tick",
        "step": step,
        "game_loop": game_loop,
        "timestamp": _timestamp(timestamp),
    }
    if reward is not None:
        payload["reward"] = reward
    if actions:
        payload["actions"] = list(actions)
    if observation:
        payload["observation"] = dict(observation)
    if extra:
        payload.update(dict(extra))
    return payload


def log_macro_decision(
    *,
    name: str,
    timestamp: float | None = None,
    priority: float | None = None,
    tags: Sequence[str] | None = None,
    context: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    extra: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured payload for high-level macro decisions."""
    payload: dict[str, Any] = {
        "type": "macro_decision",
        "name": name,
        "timestamp": _timestamp(timestamp),
    }
    if priority is not None:
        payload["priority"] = priority
    if tags:
        payload["tags"] = list(tags)
    if context:
        payload["context"] = dict(context)
    if extra:
        payload.update(dict(extra))
    return payload


def log_episode_end(
    *,
    episode_id: str | int,
    result: str,
    total_reward: float,
    duration_seconds: float | None = None,
    frames: int | None = None,
    timestamp: float | None = None,
    extra: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured payload for the final episode summary."""
    payload: dict[str, Any] = {
        "type": "episode_end",
        "episode_id": str(episode_id),
        "result": result,
        "total_reward": total_reward,
        "timestamp": _timestamp(timestamp),
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    if frames is not None:
        payload["frames"] = frames
    if extra:
        payload.update(dict(extra))
    return payload


def _timestamp(value: float | None) -> float:
    return float(value if value is not None else time.time())

