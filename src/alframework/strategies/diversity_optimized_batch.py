# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 08:39:11 2026

@author: frederic.ros
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy
from alframework.strategies.uncertainty import margin_uncertainty, entropy

@register("diversity_optimized_batch")
class DiversityOptimizedBatchSelection(QueryStrategy):
    """
    Diversity-Optimized Batch Selection (DOBS).
    Selects a batch that maximizes both uncertainty and diversity using a greedy submodular approach.

    Parameters:
    -----------
    uncertainty_metric: str
        Uncertainty metric ('margin' or 'entropy').
    lambda_diversity: float
        Trade-off between uncertainty and diversity (0-1).
    """

    def __init__(self, uncertainty_metric: str = "margin", lambda_diversity: float = 0.5):
        self.uncertainty_metric = uncertainty_metric
        self.lambda_diversity = lambda_diversity

    def select(self, state: ALState, budget: int) -> np.ndarray:
        # Fit model and compute uncertainty
        state.model.fit(state.X_labeled, state.y_labeled)
        proba = state.model.predict_proba(state.X_unlabeled)

        # Compute uncertainty scores
        if self.uncertainty_metric == "margin":
            uncertainty = margin_uncertainty(proba)
        elif self.uncertainty_metric == "entropy":
            uncertainty = entropy(proba)
        else:
            raise ValueError("uncertainty_metric must be 'margin' or 'entropy'")

        # Normalize uncertainty to [0, 1]
        uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-10)

        # Compute pairwise distances (for diversity)
        distances = pairwise_distances(state.X_unlabeled, metric="euclidean")
        max_distance = np.max(distances)
        similarity = 1 - (distances / (max_distance + 1e-10))  # Convert to similarity [0, 1]

        # Greedy selection
        selected_indices = []
        remaining_indices = set(range(len(state.X_unlabeled)))

        for _ in range(min(budget, len(state.X_unlabeled))):
            best_score = -1
            best_index = -1

            for i in remaining_indices:
                # Uncertainty term (normalized)
                u_score = uncertainty[i]

                # Diversity term: min similarity to already selected points
                if selected_indices:
                    d_score = 1 - np.max(similarity[i, selected_indices])
                else:
                    d_score = 1.0  # Maximum diversity if no points selected yet

                # Combined score: weighted sum of uncertainty and diversity
                score = (1 - self.lambda_diversity) * u_score + self.lambda_diversity * d_score

                if score > best_score:
                    best_score = score
                    best_index = i

            if best_index >= 0:
                selected_indices.append(best_index)
                remaining_indices.remove(best_index)

        return np.array(selected_indices, dtype=int)
