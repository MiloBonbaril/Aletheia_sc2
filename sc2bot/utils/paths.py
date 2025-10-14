from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
RUN_CONFIG_PATH: Path = CONFIGS_DIR / "run.yaml"

DATA_DIR: Path = PROJECT_ROOT / "data"
RUNS_DIR: Path = DATA_DIR / "runs"
REPLAYS_DIR: Path = DATA_DIR / "replays"

_SC2_ENV_VARS: tuple[str, ...] = ("SC2PATH", "STARCRAFT2PATH", "SC2_DIR")
_DEFAULT_SC2_DIRECTORIES: tuple[Path, ...] = (
    Path("C:/Program Files (x86)/StarCraft II"),
    Path("C:/Program Files/StarCraft II"),
    Path.home() / "StarCraft II",
    Path.home() / "Documents" / "StarCraft II",
    Path.home() / "Library" / "Application Support" / "Blizzard" / "StarCraft II",
    Path("/Applications/StarCraft II"),
)


def ensure_runs_dir(create: bool = True) -> Path:
    """Return the run artifacts directory, optionally creating it."""
    return _ensure_dir(RUNS_DIR) if create else RUNS_DIR


def ensure_replays_dir(create: bool = True) -> Path:
    """Return the replay storage directory, optionally creating it."""
    return _ensure_dir(REPLAYS_DIR) if create else REPLAYS_DIR


def ensure_data_subdir(name: str, create: bool = True) -> Path:
    """Return a named data subdirectory under data/, optionally creating it."""
    candidate = DATA_DIR / name
    return _ensure_dir(candidate) if create else candidate


def resolve_sc2_dir(explicit_path: Optional[Path | str] = None, ensure_exists: bool = True) -> Path:
    """
    Resolve the StarCraft II installation directory.

    Search order:
        1. An explicit path passed to the function.
        2. Environment variables: SC2PATH, STARCRAFT2PATH, SC2_DIR.
        3. Common installation locations for Windows, Linux, and macOS.
    """
    candidates = list(_iter_sc2_candidates(explicit_path))
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not ensure_exists or candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the StarCraft II installation directory. "
        f"Searched: {searched or 'no paths supplied'}"
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iter_sc2_candidates(explicit_path: Optional[Path | str]) -> Iterable[Path]:
    if explicit_path:
        yield Path(explicit_path)
    for env_var in _SC2_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            yield Path(value)
    yield from _DEFAULT_SC2_DIRECTORIES
