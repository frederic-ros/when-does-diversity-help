from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


@register("badge_approx")
class BADGEApprox(QueryStrategy):
    """Classifier-agnostic BADGE approximation.

    The original BADGE (Ash et al., ICLR 2020) builds last-layer gradient
    embeddings  g_x = (p - onehot(y_hat)) (x) h(x),  where h(x) is the penultimate
    hidden representation of a neural network, then applies k-means++ seeding to
    pick a batch that is both high-magnitude (uncertain) and diverse.

    Here we DELIBERATELY drop the h(x) factor and use g_x = p - onehot(y_hat),
    which requires only predict_proba. This makes the method applicable to ANY
    probabilistic classifier (RandomForest, LogisticRegression, ...) rather than
    being tied to a differentiable network with an accessible hidden layer.
    We therefore register it as "badge_approx": it preserves BADGE's mechanism
    (uncertainty-magnitude (x) directional diversity via k-means++) in the
    label/gradient-direction space, without the feature-space scaling.
    """
    def __init__(self, random_state: int = 0):
        self.random_state = int(random_state)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        proba = np.asarray(state.model.predict_proba(state.X_unlabeled))
        n = proba.shape[0]
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=int)

        preds = np.argmax(proba, axis=1)
        onehot = np.zeros_like(proba)
        onehot[np.arange(n), preds] = 1.0
        grad_embed = proba - onehot  # (n, C)

        _, idx = kmeans_plusplus(grad_embed, n_clusters=k, random_state=self.random_state)
        # With duplicated gradient embeddings (e.g. duplicated rows, or a
        # random forest giving identical probability vectors to many points),
        # k-means++ seeding can return the same index twice. Keep the distinct
        # seeds and top up with the largest-magnitude gradients, which is BADGE's
        # uncertainty criterion.
        sel = list(dict.fromkeys(int(i) for i in idx))
        if len(sel) < k:
            mag = np.linalg.norm(grad_embed, axis=1)
            chosen = set(sel)
            for j in np.argsort(-mag, kind="stable"):
                j = int(j)
                if j not in chosen:
                    sel.append(j); chosen.add(j)
                    if len(sel) == k:
                        break
        return np.asarray(sel, dtype=int)
