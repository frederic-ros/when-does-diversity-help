from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


@register("typiclust")
class TypiClustSampling(QueryStrategy):
    """TypiClust: cluster into `budget` clusters and pick the most 'typical' point per cluster."""
    def __init__(self, neighbors: int = 5, random_state: int = 0):
        self.neighbors = int(neighbors)
        self.random_state = int(random_state)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X = state.X_unlabeled
        n = len(X)
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=int)

        # TypiClust (Hacohen et al. 2022): partition into (|labeled| + budget)
        # clusters, then query the most TYPICAL point of each of the `budget`
        # largest uncovered clusters. Typicality of a point = inverse of the
        # mean distance to its K nearest neighbors WITHIN the same cluster
        # (local density), not over the whole pool.
        Xl = getattr(state, "X_labeled", None)
        n_labeled = 0 if Xl is None else len(Xl)
        n_clusters = min(n_labeled + k, n)
        n_clusters = max(n_clusters, k)
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                    n_init="auto").fit(X)
        labels = km.labels_

        # rank clusters by size (descending); query the budget largest ones
        cluster_ids, sizes = np.unique(labels, return_counts=True)
        order = cluster_ids[np.argsort(-sizes)]

        selected = []
        for c in order:
            if len(selected) >= k:
                break
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            if members.size == 1:
                selected.append(int(members[0]))
                continue
            # KNN density computed WITHIN the cluster
            Xc = X[members]
            kk = min(self.neighbors, len(members) - 1)
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)  # +1: self included
            dists, _ = nn.kneighbors(Xc)
            mean_d = dists[:, 1:].mean(axis=1)          # drop self-distance
            typicality = 1.0 / (mean_d + 1e-12)
            best_local = members[int(np.argmax(typicality))]
            selected.append(int(best_local))

        return np.asarray(selected[:k], dtype=int)
