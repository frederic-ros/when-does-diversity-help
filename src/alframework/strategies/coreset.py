from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


def greedy_k_center(X: np.ndarray, k: int, rng: np.random.Generator,
                    X_labeled: np.ndarray = None) -> np.ndarray:
    """Greedy k-center (Sener & Savarese 2018), O(n*k), no n^2 matrix.

    Faithful version: the cover includes the ALREADY labeled points. The
    min-distance vector d is initialized from distances to X_labeled, so the
    first (and every) pick is the unlabeled point farthest from the current
    labeled+selected cover. Falls back to a random first pick only when no
    labeled points are available.
    """
    n = X.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int)

    selected = np.empty(k, dtype=int)
    d = np.full(n, np.inf)

    if X_labeled is not None and len(X_labeled) > 0:
        # distance of each unlabeled point to the nearest labeled point
        for c in X_labeled:
            d = np.minimum(d, np.linalg.norm(X - c, axis=1))
        start = 0  # first pick = farthest from labeled cover
    else:
        selected[0] = int(rng.integers(0, n))
        d = np.minimum(d, np.linalg.norm(X - X[selected[0]], axis=1))
        d[selected[0]] = -np.inf
        start = 1

    for i in range(start, k):
        nxt = int(np.argmax(d))
        selected[i] = nxt
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
        # Never re-pick a selected point (with duplicated rows all
        # distances can reach 0 and argmax returned an already-selected index)
        d[selected[: i + 1]] = -np.inf
    return selected


@register("coreset_greedy")
class CoresetGreedy(QueryStrategy):
    def select(self, state: ALState, budget: int) -> np.ndarray:
        return greedy_k_center(
            state.X_unlabeled, budget, state.rng,
            X_labeled=getattr(state, "X_labeled", None),
        )


@register("coreset_kmeanspp")
class CoresetKMeansPP(QueryStrategy):
    """Coreset selection using kmeans++ seeding (fast, decent diversity)."""
    def select(self, state: ALState, budget: int) -> np.ndarray:
        n = len(state.X_unlabeled)
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=int)
        _, indices = kmeans_plusplus(state.X_unlabeled, n_clusters=k,
                                     random_state=int(state.rng.integers(2**31 - 1)))  # per-run seed
        return np.asarray(indices, dtype=int)
