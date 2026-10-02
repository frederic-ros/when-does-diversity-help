from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class ALState:
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    X_unlabeled: np.ndarray
    model: Any
    rng: np.random.Generator
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
