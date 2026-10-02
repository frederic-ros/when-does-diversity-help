from __future__ import annotations
import numpy as np
from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy

@register("random")
class RandomSampling(QueryStrategy):
    def select(self, state: ALState, budget: int) -> np.ndarray:
        n = len(state.X_unlabeled)
        k = min(budget, n)
        return state.rng.choice(n, size=k, replace=False)
