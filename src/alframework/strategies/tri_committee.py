from __future__ import annotations

import numpy as np

from alframework.utils.bootstrap import stratified_bootstrap
from sklearn.base import clone

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


def _bootstrap_indices(rng: np.random.Generator, n: int, ratio: float) -> np.ndarray:
    m = max(1, int(np.ceil(ratio * n)))
    return rng.choice(n, size=m, replace=True)


def _vote_entropy_from_votes(votes: np.ndarray, n_classes: int) -> np.ndarray:
    n_models, n_samples = votes.shape
    counts = np.zeros((n_samples, n_classes), dtype=float)
    for c in range(n_classes):
        counts[:, c] = np.sum(votes == c, axis=0)
    p = counts / max(1, n_models)
    eps = 1e-12
    return -np.sum(p * np.log(p + eps), axis=1)


@register("tri_committee")
class TriCommitteeDisagreement(QueryStrategy):
    """Tri-training-inspired acquisition (A-only): build 3 bootstrap clones and query
    where disagreement is highest.

    Modes:
    - "two_vs_one": prioritize points with 2-vs-1 split (strong signal in tri-training)
    - "vote_entropy": vote entropy among 3

    Parameters
    ----------
    bootstrap_ratio : float
    mode : {"two_vs_one","vote_entropy"}
    """

    def __init__(self, bootstrap_ratio: float = 1.0, mode: str = "two_vs_one"):
        self.bootstrap_ratio = float(bootstrap_ratio)
        self.mode = str(mode)


    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_L, y_L, X_U = state.X_labeled, state.y_labeled, state.X_unlabeled
        nU = len(X_U)
        k = min(int(budget), nU)
        if k <= 0:
            return np.array([], dtype=int)

        # ✅ n_classes connu en entrée (robuste unbalanced)
        if hasattr(state, "n_classes") and state.n_classes is not None:
            n_classes = int(state.n_classes)
        elif hasattr(state, "labels") and state.labels is not None:
            n_classes = int(len(state.labels))
        else:
            # fallback si labels sont 0..C-1 dans ton synth
            n_classes = int(np.max(y_L)) + 1

        # garde-fou si labeled vide
        if len(X_L) == 0:
            return state.rng.choice(nU, size=k, replace=False)

        votes = []
        for _ in range(3):
            m = clone(state.model)
            idx = stratified_bootstrap(state.rng, y_L, self.bootstrap_ratio)  # class-stratified
            m.fit(X_L[idx], y_L[idx])
            votes.append(m.predict(X_U))

        votes = np.stack(votes, axis=0)  # (3, nU)

        if self.mode == "two_vs_one":
            a, b, c = votes[0], votes[1], votes[2]
            two_vs_one = ((a == b) & (a != c)) | ((a == c) & (a != b)) | ((b == c) & (b != a))
            score = two_vs_one.astype(float)
            score += 1e-3 * _vote_entropy_from_votes(votes, n_classes=n_classes)
        else:
            score = _vote_entropy_from_votes(votes, n_classes=n_classes)

        return np.argsort(-score)[:k]