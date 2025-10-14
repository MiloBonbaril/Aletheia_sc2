from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

import orjson
import pyarrow as pa
import pyarrow.parquet as pq


class EventWriter:
    """Stream structured episode events to JSONL and rotate Parquet chunks."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        parquet_chunk_size: int = 1024,
        parquet_compression: str = "zstd",
    ) -> None:
        self._run_dir = Path(run_dir)
        self._episodes_dir = self._run_dir / "events" / "episodes"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)

        self._parquet_chunk_size = max(1, parquet_chunk_size)
        self._parquet_compression = parquet_compression

        self._jsonl_handle: Optional[BinaryIO] = None
        self._current_episode_id: Optional[str] = None
        self._parquet_buffer: list[dict[str, Any]] = []
        self._parquet_chunk_index: int = 0

    @property
    def episodes_dir(self) -> Path:
        return self._episodes_dir

    def start_episode(self, episode_id: str | int) -> None:
        """Begin logging a new episode, closing any previously open episode."""
        if self._jsonl_handle is not None:
            self.close_episode()

        self._current_episode_id = str(episode_id)
        jsonl_path = self._episodes_dir / f"{self._current_episode_id}.jsonl"
        self._jsonl_handle = open(jsonl_path, "ab")

        self._parquet_buffer.clear()
        self._parquet_chunk_index = 0

    def write_event(self, payload: dict[str, Any]) -> None:
        if self._jsonl_handle is None or self._current_episode_id is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        self._append_jsonl(payload)
        self._append_parquet(payload)

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        assert self._jsonl_handle is not None
        line = orjson.dumps(payload)
        self._jsonl_handle.write(line)
        self._jsonl_handle.write(b"\n")
        self._jsonl_handle.flush()

    def _append_parquet(self, payload: dict[str, Any]) -> None:
        self._parquet_buffer.append(payload)
        if len(self._parquet_buffer) >= self._parquet_chunk_size:
            self._flush_parquet_buffer()

    def _flush_parquet_buffer(self, *, force: bool = False) -> None:
        if not self._parquet_buffer:
            return
        if not force and len(self._parquet_buffer) < self._parquet_chunk_size:
            return

        assert self._current_episode_id is not None
        chunk_path = self._episodes_dir / f"{self._current_episode_id}_{self._parquet_chunk_index}.parquet"
        table = pa.Table.from_pylist(self._parquet_buffer)
        pq.write_table(
            table,
            chunk_path,
            compression=self._parquet_compression,
        )
        self._parquet_buffer.clear()
        self._parquet_chunk_index += 1

    def flush(self) -> None:
        if self._jsonl_handle is None:
            return
        self._jsonl_handle.flush()
        self._flush_parquet_buffer(force=True)

    def close_episode(self) -> None:
        if self._jsonl_handle is None:
            return

        try:
            self._flush_parquet_buffer(force=True)
        finally:
            self._jsonl_handle.close()
            self._jsonl_handle = None
            self._current_episode_id = None
            self._parquet_chunk_index = 0
            self._parquet_buffer.clear()

    def close(self) -> None:
        self.close_episode()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_many(self, payloads: Iterable[dict[str, Any]]) -> None:
        for payload in payloads:
            self.write_event(payload)
