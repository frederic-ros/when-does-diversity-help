from __future__ import annotations

import numpy as np

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


def _check_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError("predict_proba must return array of shape (n_samples, n_classes)")
    return proba


def least_confident(proba: np.ndarray) -> np.ndarray:
    proba = _check_proba(proba)
    return 1.0 - np.max(proba, axis=1)


def entropy(proba: np.ndarray) -> np.ndarray:
    proba = _check_proba(proba)
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    proba = _check_proba(proba)
    sorted_p = -np.sort(-proba, axis=1)
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    return 1.0 - margin  # higher => more uncertain


@register("least_confident")
class LeastConfidentSampling(QueryStrategy):
    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        u = least_confident(state.model.predict_proba(state.X_unlabeled))
        k = min(budget, len(u))
        return np.argsort(-u)[:k]


@register("entropy")
class EntropySampling(QueryStrategy):
    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        u = entropy(state.model.predict_proba(state.X_unlabeled))
        k = min(budget, len(u))
        return np.argsort(-u)[:k]


@register("margin")
class MarginUncertainty(QueryStrategy):
    def select(self, state: ALState, budget: int) -> np.ndarray:
        state.model.fit(state.X_labeled, state.y_labeled)
        u = margin_uncertainty(state.model.predict_proba(state.X_unlabeled))
        k = min(budget, len(u))
        return np.argsort(-u)[:k]
