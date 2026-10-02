from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy
from alframework.strategies.uncertainty import margin_uncertainty, entropy, least_confident


@register("rank2022")
class RANK2022(QueryStrategy):
    """Refactor of the selection logic in RANK2022.py (loop-based script).

    Steps:
    1) Train on labeled
    2) Compute uncertainty on unlabeled
    3) Take top-s most uncertain (s = min(n_unlabeled, budget*s_factor))
    4) Agglomerative clustering into `budget` clusters
    5) Pick, per cluster, the most uncertain point
    """
    def __init__(self, s_factor: int = 10, linkage: str = "ward", loss: str = "margin"):
        self.s_factor = int(s_factor)
        self.linkage = linkage
        self.loss = loss

    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        proba = state.model.predict_proba(state.X_unlabeled)

        if self.loss == "margin":
            u = margin_uncertainty(proba)
        elif self.loss == "entropy":
            u = entropy(proba)
        elif self.loss == "least_confident":
            u = least_confident(proba)
        else:
            raise ValueError("loss must be 'margin', 'entropy', or 'least_confident'")

        n = len(u)
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=int)

        s = min(n, k * max(1, self.s_factor))
        top_s_local = np.argsort(-u)[:s]  # indices within unlabeled

        X_top = state.X_unlabeled[top_s_local]
        if k == 1 or len(X_top) == 1:
            return top_s_local[:1].astype(int)

        clustering = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
        labels = clustering.fit_predict(X_top)

        selected = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            best_local = members[np.argmax(u[top_s_local[members]])]
            selected.append(int(top_s_local[best_local]))

        return np.asarray(selected, dtype=int)
