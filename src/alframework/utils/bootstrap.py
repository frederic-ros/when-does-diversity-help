"""Class-stratified bootstrap for committee strategies.

A plain bootstrap of a small, imbalanced labeled set can miss a minority class
entirely (with n_init=20 and one minority example, the probability is about
(1-1/20)^20 = 36%). Classifiers such as logistic regression then cannot be fit,
and committee members disagree only because some of them never saw a class.
Resampling with replacement *within each class* keeps every labeled class in
every bootstrap sample and preserves class proportions.
"""
from __future__ import annotations

import numpy as np


def stratified_bootstrap(rng: np.random.Generator, y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """Indices of a bootstrap sample of y, drawn with replacement within each class.

    Each class c with n_c labeled points contributes max(1, round(ratio * n_c))
    points, so every class present in y is present in the sample.
    """
    y = np.asarray(y)
    parts = []
    for c in np.unique(y):
        idx_c = np.flatnonzero(y == c)
        m_c = max(1, int(round(ratio * len(idx_c))))
        parts.append(rng.choice(idx_c, size=m_c, replace=True))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return idx
