# src/alframework/strategies/probcover.py
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from alframework.core.registry import register
from alframework.core.state import ALState
from alframework.strategies.base import QueryStrategy


@register("probcover")
class ProbCoverSampling(QueryStrategy):
    """
    ProbCover (covering lens) — implémentation autonome (sans skactiveml).

    Idée :
      - Fixer un rayon delta
      - Construire pour chaque point i l'ensemble N(i) des points dans la boule (radius neighbors)
      - Greedy max-coverage : à chaque étape, choisir i qui couvre le plus de points encore non couverts.

    Réfs : "Active Learning Through a Covering Lens" (NeurIPS 2022) + blogpost associé. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        *,
        delta: float | None = None,
        # si delta=None, on le déduit via un quantile sur la distance au k-ième voisin
        delta_quantile: float = 0.10,
        knn_k: int = 10,
        metric: str = "euclidean",
        random_state: int = 0,
    ):
        self.delta = delta
        self.delta_quantile = float(delta_quantile)
        self.knn_k = int(knn_k)
        self.metric = str(metric)
        self.random_state = int(random_state)

    def _estimate_delta(self, X: np.ndarray) -> float:
        """
        Heuristique pratique :
          - calcule la distance au k-ième plus proche voisin (k=knn_k)
          - prend un quantile (delta_quantile) de cette distribution
        """
        n = X.shape[0]
        if n <= 2:
            return 0.0

        k = max(2, min(self.knn_k + 1, n))  # +1 car le point lui-même est voisin #0
        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(X)
        dists, _ = nn.kneighbors(X, return_distance=True)
        dk = dists[:, -1]  # distance au k-ième voisin
        q = np.clip(self.delta_quantile, 0.0, 1.0)
        delta = float(np.quantile(dk, q))
        # sécurité : éviter delta=0 si tout est identique
        return max(delta, 1e-12)

    def select(self, state: ALState, budget: int) -> np.ndarray:
        X = np.asarray(state.X_unlabeled)
        n = X.shape[0]
        k = min(int(budget), n)
        if k <= 0 or n == 0:
            return np.array([], dtype=int)

        delta = float(self.delta) if self.delta is not None else self._estimate_delta(X)

        # radius neighbors (inclut le point lui-même)
        nn = NearestNeighbors(radius=delta, metric=self.metric)
        nn.fit(X)
        neigh = nn.radius_neighbors(X, radius=delta, return_distance=False)

        # Greedy max-coverage
        uncovered = np.ones(n, dtype=bool)
        selected: list[int] = []

        # Pour éviter O(n^2) “bête”, on maintient un score = nb de non-couverts couverts par i
        # et on le met à jour localement (simple et suffisamment rapide pour n ~ quelques milliers).
        scores = np.array([uncovered[idxs].sum() for idxs in neigh], dtype=int)

        rng = np.random.default_rng(self.random_state)

        for _ in range(k):
            best_score = scores.max()
            if best_score <= 0:
                break  # plus rien à couvrir

            # tie-break aléatoire pour éviter biais
            candidates = np.flatnonzero(scores == best_score)
            choice = int(rng.choice(candidates))
            selected.append(choice)

            # marquer couverts
            covered_idx = np.asarray(neigh[choice], dtype=int)
            newly_covered = covered_idx[uncovered[covered_idx]]
            uncovered[newly_covered] = False

            # mise à jour “simple” des scores (on recalcule seulement les points touchés)
            # Ici on fait une MAJ conservative : recalc complet si tu veux ultra-safe.
            # Pour rester simple et robuste, on recalc tous les scores — OK pour n ~ 1e4 max.
            scores = np.array([uncovered[idxs].sum() for idxs in neigh], dtype=int)

            # optionnel: éviter de re-sélectionner
            scores[selected] = -1

        return np.asarray(selected, dtype=int)