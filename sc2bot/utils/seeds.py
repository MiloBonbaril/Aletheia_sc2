from __future__ import annotations

import os
import random
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional at runtime.
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at runtime.
    torch = None  # type: ignore[assignment]

__all__ = ["set_global_seed"]


def set_global_seed(seed: Optional[int]) -> Optional[int]:
    """
    Seed Python's RNG along with numpy and torch if they are available.

    The function safely handles optional dependencies and returns the seed
    that was applied so callers can log or reuse it.
    """
    if seed is None:
        return None
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer or None, got {type(seed)!r}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed
