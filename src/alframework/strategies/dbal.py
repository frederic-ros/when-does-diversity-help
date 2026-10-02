from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy
from alframework.strategies.uncertainty import margin_uncertainty, entropy


@register("dbal")
class DBAL(QueryStrategy):
    """DBAL variant (as in your DBAL_E):
    - compute uncertainty (margin or entropy)
    - take top (budget*dbal_factor) most uncertain
    - run weighted KMeans into `budget` clusters (weights = uncertainty)
    - select closest point to each center within the top-uncertain subset
    """
    def __init__(self, method: str = "margin", dbal_factor: int = 5, random_state: int = 0):
        self.method = method
        self.dbal_factor = int(dbal_factor)
        self.random_state = int(random_state)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        proba = state.model.predict_proba(state.X_unlabeled)
        method_s=False
        
        if self.method == "margin":
            u = margin_uncertainty(proba)
            method_s=True
        if self.method == "entropy":
            u = entropy(proba)
            method_s=True
        if self.method == "least-confidence":
            u = 1 - np.max(proba, axis=1)
            method_s=True
        
        if method_s==False: u = 1 - np.max(proba, axis=1)
        
        n = len(u)
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=int)

        m = min(n, k * max(1, self.dbal_factor))
        top_idx = np.argsort(-u)[:m]

        # weighted kmeans on subset
        weights = u[top_idx] + 1e-8
        X_sub = state.X_unlabeled[top_idx]
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        km.fit(X_sub, sample_weight=weights)

        centers = km.cluster_centers_
        # One distinct point per center: nearest point not already taken.
        # (Taking np.unique of the nearest points would return fewer than k
        #  points when two centers share their nearest point, e.g. on datasets
        #  with duplicate rows.)
        taken = np.zeros(len(top_idx), bool)
        selected = []
        for c in centers:
            d = np.linalg.norm(X_sub - c, axis=1)
            d[taken] = np.inf
            j = int(np.argmin(d))
            taken[j] = True
            selected.append(int(top_idx[j]))
        return np.asarray(selected[:k], dtype=int)
