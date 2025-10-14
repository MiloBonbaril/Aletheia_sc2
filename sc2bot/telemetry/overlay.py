from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

import orjson


class LiveStats:
    """Throttle writes of live metrics to a JSON file for overlay consumption."""

    def __init__(
        self,
        path: Path | str,
        *,
        refresh_hz: float = 2.0,
    ) -> None:
        candidate = Path(path)
        suffix = candidate.suffix.lower()
        self._path = candidate if suffix == ".json" else candidate / "metrics" / "live" / "stats.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._refresh_interval = 1.0 / max(refresh_hz, 0.001)
        self._last_write_ts: float = 0.0
        self._latest_payload: Optional[dict] = None

    @property
    def path(self) -> Path:
        return self._path

    def update(
        self,
        payload: Mapping[str, object] | MutableMapping[str, object],
        *,
        force: bool = False,
    ) -> None:
        self._latest_payload = dict(payload)
        if force or self._should_write():
            self._write()

    def flush(self) -> None:
        if self._latest_payload is None:
            return
        self._write()

    def _should_write(self) -> bool:
        now = time.monotonic()
        return (now - self._last_write_ts) >= self._refresh_interval

    def _write(self) -> None:
        if self._latest_payload is None:
            return

        data = orjson.dumps(self._latest_payload, option=orjson.OPT_INDENT_2)
        self._path.write_bytes(data)
        self._last_write_ts = time.monotonic()
