# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 08:24:35 2026

@author: frederic.ros
"""

# robust_qbc.py
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
    en remettant 0 sur les classes absentes.

    classes_ : ex array([0,2]) si la classe 1 n'a pas été vue.
    """
    n = p.shape[0]
    out = np.zeros((n, n_classes), dtype=float)

    # map classe->colonne
    for j, c in enumerate(classes_):
        c_int = int(c)
        if 0 <= c_int < n_classes:
            out[:, c_int] = p[:, j]

    # sécurité: renormaliser (évite somme != 1 quand classes manquantes)
    s = out.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    out = out / s
    return out


@register("robust_qbc")
class RobustQBC(QueryStrategy):
    """
    Robust Query-by-Committee (RQBC).
    Disagreement pondéré par la confiance.

    metric: {"vote_entropy","confidence_weighted"}
    """

    def __init__(self, n_committee: int = 5, bootstrap_ratio: float = 0.8, metric: str = "confidence_weighted"):
        self.n_committee = int(n_committee)
        self.bootstrap_ratio = float(bootstrap_ratio)
        self.metric = str(metric)

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

        # taille bootstrap
        n_samples = max(1, int(np.ceil(self.bootstrap_ratio * len(X_L))))

        probas = []
        for _ in range(self.n_committee):
            # resample reproductible avec state.rng
            seed = int(state.rng.integers(0, 2**31 - 1))
            idx = stratified_bootstrap(np.random.default_rng(seed), y_L, self.bootstrap_ratio)  # class-stratified bootstrap
            X_s, y_s = X_L[idx], y_L[idx]

            base = getattr(state, "base_model", state.model)
            m = clone(base)
            m.fit(X_s, y_s)

            if not hasattr(m, "predict_proba"):
                raise ValueError("robust_qbc requiert un modèle avec predict_proba().")

            p = m.predict_proba(X_U)
            p = _align_proba(p, m.classes_, n_classes)  # ✅ clé pour unbalanced
            probas.append(p)

        proba_committee = np.stack(probas, axis=0)      # (M, nU, C)
        mean_proba = np.mean(proba_committee, axis=0)   # (nU, C)

        if self.metric == "vote_entropy":
            scores = -np.sum(mean_proba * np.log(mean_proba + 1e-12), axis=1)

        elif self.metric == "confidence_weighted":
            member_confidence = np.max(proba_committee, axis=2)         # (M, nU)
            disagreement = np.std(proba_committee, axis=0).mean(axis=1) # (nU,)
            scores = disagreement * np.mean(member_confidence, axis=0)  # (nU,)

        else:
            raise ValueError("metric must be 'vote_entropy' or 'confidence_weighted'")

        return np.argsort(-scores)[:k].astype(int)