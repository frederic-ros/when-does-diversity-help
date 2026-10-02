from __future__ import annotations

import numpy as np

from alframework.utils.bootstrap import stratified_bootstrap
from sklearn.base import clone

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy

def _align_proba(proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    """
    proba: (n_samples, C') associé à classes_
    return: (n_samples, n_classes) avec 0 sur les classes absentes
    """
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, c in enumerate(classes_):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = proba[:, j]
    return out

def _bootstrap_indices(rng: np.random.Generator, n: int, ratio: float) -> np.ndarray:
    m = max(1, int(np.ceil(ratio * n)))
    return rng.choice(n, size=m, replace=True)


def _vote_entropy(votes: np.ndarray, n_classes: int) -> np.ndarray:
    """
    votes: (n_models, n_samples) integer labels
    returns: (n_samples,) vote-entropy
    """
    n_models, n_samples = votes.shape
    counts = np.zeros((n_samples, n_classes), dtype=float)
    for c in range(n_classes):
        counts[:, c] = np.sum(votes == c, axis=0)
    p = counts / max(1, n_models)
    eps = 1e-12
    return -np.sum(p * np.log(p + eps), axis=1)


def _mean_kl_from_proba(probas: np.ndarray) -> np.ndarray:
    """
    probas: (n_models, n_samples, n_classes)
    returns: (n_samples,) mean KL divergence to mean-proba
    """
    eps = 1e-12
    p_bar = np.mean(probas, axis=0)  # (n_samples, n_classes)
    p_bar = np.clip(p_bar, eps, 1.0)
    p_bar = p_bar / np.sum(p_bar, axis=1, keepdims=True)

    kl = np.zeros(p_bar.shape[0], dtype=float)
    for p in probas:
        p = np.clip(p, eps, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        kl += np.sum(p * (np.log(p) - np.log(p_bar)), axis=1)
    return kl / probas.shape[0]


@register("qbc")
class QueryByCommittee(QueryStrategy):
    """Query-by-Committee with bootstrap clones.

    Parameters
    ----------
    n_committee : int
        Number of committee members.
    bootstrap_ratio : float
        Fraction of labeled set used (with replacement) to train each clone.
    metric : {"vote_entropy", "kl"}
        Disagreement metric. "kl" requires predict_proba on the model.
    """

    def __init__(self, n_committee: int = 5, bootstrap_ratio: float = 1.0, metric: str = "vote_entropy"):
        self.n_committee = int(n_committee)
        self.bootstrap_ratio = float(bootstrap_ratio)
        self.metric = str(metric)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_L, y_L, X_U = state.X_labeled, state.y_labeled, state.X_unlabeled
        nU = len(X_U)
        k = min(int(budget), nU)
        if k <= 0:
            return np.array([], dtype=int)

        # n_classes connu en entrée
        if hasattr(state, "n_classes") and state.n_classes is not None:
            n_classes = int(state.n_classes)
        elif hasattr(state, "labels") and state.labels is not None:
            n_classes = int(len(state.labels))
        else:
            raise ValueError(
            "QBC: n_classes inconnu. Ajoute state.n_classes ou state.labels."
            )

        votes = []
        probas = []

        if len(X_L) == 0:
            return state.rng.choice(nU, size=k, replace=False)

        for _ in range(self.n_committee):
            m = clone(state.model)
            idx = stratified_bootstrap(state.rng, y_L, self.bootstrap_ratio)  # class-stratified
            m.fit(X_L[idx], y_L[idx])

            votes.append(m.predict(X_U))

        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X_U)
            p = _align_proba(p, m.classes_, n_classes)
            probas.append(p)

        votes = np.stack(votes, axis=0)

        if self.metric == "kl" and len(probas) == self.n_committee:
            probas = np.stack(probas, axis=0)
            score = _mean_kl_from_proba(probas)
        else:
            score = _vote_entropy(votes, n_classes=n_classes)

        return np.argsort(-score)[:k]

    
    