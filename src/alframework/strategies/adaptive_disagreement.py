from __future__ import annotations

import numpy as np
from sklearn.base import clone
from alframework.utils.bootstrap import stratified_bootstrap

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


def _get_n_classes(state: ALState, y_L: np.ndarray) -> int:
    """Nombre de classes global (robuste)."""
    if hasattr(state, "n_classes") and state.n_classes is not None:
        return int(state.n_classes)
    if hasattr(state, "labels") and state.labels is not None:
        return int(len(state.labels))
    # fallback (OK si labels = 0..C-1)
    return int(np.max(y_L)) + 1


def _align_proba(p: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Aligne une proba sklearn (n_samples, C_presentes) vers (n_samples, n_classes),
    en remettant 0 sur les classes absentes, puis renormalise.
    """
    n = p.shape[0]
    out = np.zeros((n, n_classes), dtype=float)
    for j, c in enumerate(classes_):
        c_int = int(c)
        if 0 <= c_int < n_classes:
            out[:, c_int] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return out / s


@register("adaptive_disagreement")
class AdaptiveDisagreementSelection(QueryStrategy):
    """
    Adaptive Disagreement Selection (ADS).
    Dynamically adjusts the disagreement threshold based on committee confidence.
    """

    def __init__(self, n_committee: int = 5, min_disagreement: float = 0.2):
        self.n_committee = int(n_committee)
        self.min_disagreement = float(min_disagreement)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X_L, y_L, X_U = state.X_labeled, state.y_labeled, state.X_unlabeled
        nU = len(X_U)
        k = min(int(budget), nU)
        if k <= 0:
            return np.array([], dtype=int)

        # garde-fou si labeled vide
        if X_L is None or len(X_L) == 0:
            return state.rng.choice(nU, size=k, replace=False).astype(int)

        n_classes = _get_n_classes(state, y_L)

        # Train committee (bootstrap clones)
        committee = []
        for _ in range(self.n_committee):
            seed = int(state.rng.integers(0, 2**31 - 1))
            idx = stratified_bootstrap(np.random.default_rng(seed), y_L, 1.0)  # class-stratified bootstrap: class-stratified
            X_s, y_s = X_L[idx], y_L[idx]

            # ✅ IMPORTANT : on clone le "vrai" modèle, pas le wrapper (AlignedProbaClassifier)
            base = getattr(state, "base_model", state.model)
            m = clone(base)
            m.fit(X_s, y_s)
            committee.append(m)

        # Predictions (votes)
        votes = np.stack([m.predict(X_U) for m in committee], axis=0)  # (M, nU)

        # Disagreement: 1 - (max vote fraction). Vectorized (no per-point np.unique):
        # build a (nU, n_classes) histogram of committee votes via bincount over a
        # flattened (sample, class) index, then take the max count per sample.
        # Identical result to the per-point loop, ~100x faster on large pools.
        votes_t = votes.T  # (nU, M)
        flat = (np.arange(nU)[:, None] * n_classes + votes_t).ravel()
        counts = np.bincount(flat, minlength=nU * n_classes).reshape(nU, n_classes)
        disagreement = 1.0 - (counts.max(axis=1) / self.n_committee)

        # Mean probability and uncertainty (aligned)
        if not all(hasattr(m, "predict_proba") for m in committee):
            # fallback si pas de proba: utiliser disagreement seul
            score = disagreement
            return np.argsort(-score)[:k].astype(int)

        probas = []
        for m in committee:
            p = m.predict_proba(X_U)
            p = _align_proba(p, m.classes_, n_classes)  # ✅ essentiel si classes absentes dans bootstrap
            probas.append(p)

        proba_committee = np.stack(probas, axis=0)      # (M, nU, C)
        mean_proba = np.mean(proba_committee, axis=0)   # (nU, C)
        uncertainty = 1.0 - np.max(mean_proba, axis=1)  # (nU,)

        # Adaptive threshold: disagreement must exceed uncertainty * threshold
        adaptive_threshold = uncertainty * (1.0 - self.min_disagreement)
        valid = np.where(disagreement > adaptive_threshold)[0]

        if len(valid) < k:
            # Fallback: select most uncertain if not enough high-disagreement points
            return np.argsort(-uncertainty)[:k].astype(int)

        # Select points with highest disagreement/uncertainty ratio
        scores = disagreement[valid] / (uncertainty[valid] + 1e-12)
        return valid[np.argsort(-scores)[:k]].astype(int)