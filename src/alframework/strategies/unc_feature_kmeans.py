# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 08:22:06 2026

@author: frederic.ros
"""

# bait_simple.py
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy
from alframework.strategies.uncertainty import margin_uncertainty, entropy

@register("unc_feature_kmeans")
class UncFeatureKMeans(QueryStrategy):
    """
    Uncertainty-weighted feature clustering (NOT the Fisher-based BAIT of Ash et al. 2021).
    Augments features with a scaled uncertainty channel, runs KMeans, and selects the
    most uncertain point per cluster. A diversity-by-clustering baseline closely related
    to the DBAL/WALC family.

    Parameters:
    -----------
    uncertainty_metric: str
        Metric for uncertainty ('margin' or 'entropy').
    n_clusters_factor: int
        Number of clusters = budget * n_clusters_factor.
    uncertainty_weight: float
        Weight for uncertainty in the clustering space.
    """

    def __init__(self, uncertainty_metric: str = "margin", n_clusters_factor: int = 3,
                 uncertainty_weight: float = 1.0, random_state: int = 0):
        self.random_state = int(random_state)
        self.uncertainty_metric = uncertainty_metric
        self.n_clusters_factor = n_clusters_factor
        self.uncertainty_weight = uncertainty_weight

    def select(self, state: ALState, budget: int) -> np.ndarray:
        # Fit the model
        state.model.fit(state.X_labeled, state.y_labeled)
        proba = state.model.predict_proba(state.X_unlabeled)

        # Compute uncertainty
        if self.uncertainty_metric == "margin":
            uncertainty = margin_uncertainty(proba)
        elif self.uncertainty_metric == "entropy":
            uncertainty = entropy(proba)
        else:
            raise ValueError("uncertainty_metric must be 'margin' or 'entropy'")

        # Combine features and uncertainty into a single embedding
        X_augmented = np.hstack([
            state.X_unlabeled,
            uncertainty.reshape(-1, 1) * self.uncertainty_weight
        ])

        # Cluster and select
        n_clusters = min(budget * self.n_clusters_factor, len(state.X_unlabeled))
        clustering = KMeans(n_clusters=n_clusters, n_init=10,
                            random_state=self.random_state).fit(X_augmented)
        labels = clustering.labels_

        selected_indices = []
        for c in np.unique(labels):
            cluster_points = np.where(labels == c)[0]
            if len(cluster_points) > 0:
                # Pick the most uncertain point in the cluster
                best_in_cluster = cluster_points[np.argmax(uncertainty[cluster_points])]
                selected_indices.append(best_in_cluster)

        # Keep the `budget` MOST UNCERTAIN cluster representatives (rather than
        # an arbitrary subset of the budget*n_clusters_factor representatives).
        selected_indices = np.asarray(selected_indices, dtype=int)
        order = np.argsort(-uncertainty[selected_indices], kind="stable")
        selected_indices = selected_indices[order][:budget]
        if len(selected_indices) < budget:          # fewer non-empty clusters than budget
            rest = [i for i in np.argsort(-uncertainty, kind="stable") if i not in set(selected_indices)]
            selected_indices = np.concatenate([selected_indices, rest[: budget - len(selected_indices)]])
        return np.asarray(selected_indices, dtype=int)
