from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from alframework.core.state import ALState

class QueryStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def select(self, state: ALState, budget: int) -> np.ndarray:
        """Return indices (0..len(state.X_unlabeled)-1) to query."""
        raise NotImplementedError
