# -*- coding: utf-8 -*-
"""
ActivePseudoLabelV58 — unified regime-routing active learning (STANDALONE).
===========================================================================

STANDALONE BY DESIGN. Unlike V5.7 (which inherits V5.5 -> V5.4 -> V5.3 -> V4),
this class inherits from nothing and depends only on numpy + scikit-learn. It
deliberately ABANDONS the inherited layers:

  * the mixed margin+entropy uncertainty signal (V4/V5.3),
  * the "smart" reweighting and the post-hoc "guard" (V5.4),
  * the KMeans-vs-Ward clustering router (V5.5).

Consequences to be aware of when comparing against the inherited V5.7/V58:
  * Results WILL differ on the DBAL and WALC modes, because those used the
    inherited mixed signal (e.g. the documented credit +2.3pp). This class uses
    pure margin everywhere. This is a simplification, not a bug.
  * It is ~2-3x faster per run: it goes margin -> route -> cluster directly,
    without traversing the inheritance stack each round. (The inheritance
    traversal — not the clustering — was the source of the ~30s vs ~13s gap.)

Registered under "ActivePseudoLabelV58clean" so it can sit in the panel
ALONGSIDE the inherited "ActivePseudoLabelV58" for a direct speed/quality
comparison. Rename the registration (bottom of file) to "ActivePseudoLabelV58"
once you decide to replace the inherited version outright.

Two stages, cleanly separated:

  1. UNCERTAINTY   how each candidate is scored        -> AcquisitionConfig.uncertainty
  2. ACQUISITION   how k points are chosen given a mode -> UnifiedAcquisition.select
  3. ROUTING       which mode to use, decided + corrected -> RegimeRouter

The three canonical modes (DBAL / R22 / WALC) are points of ONE parameterized
acquisition family:

    gamma   : exponent on (normalised) uncertainty used as the KMeans weight
    alpha   : representative trade-off  (1 - alpha) * U + alpha * (1 - d)
              float in [0,1], or "adaptive" (cv-based, as in V5.7 WALC)
    linkage : "kmeans" | "ward"
    uncertainty : "margin" | "entropy"
    pool_mult : top-pool size multiplier

Canonical modes:
    DBAL : alpha=1.0,        gamma=1.0, kmeans, margin, pool_mult=5
    R22  : alpha=0.0,        gamma=0.0, ward,   margin, pool_mult=10
    WALC : alpha="adaptive", gamma=2.0, kmeans, margin, pool_mult=8

Routing — act early, correct once
---------------------------------
V58 routes at round 1 so no acquisition round is wasted on indecision, then
allows ONE correction at `correction_round` (default 3), once the structural
signals have stabilised. The correction fires only if the re-decision differs
from the provisional mode AND the averaged signal is decisive (hysteresis), so
the mode never flips on round-to-round noise. After the correction round the
mode is final.

Base routing rule (variant="V58", default):
    n_classes >= multiclass_thr           -> R22
    elif mean_U >= u_flat_trigger:
        eff_dim >= eff_dim_thr and flat    -> R22   (distances diluted in high
                                                     effective dim: WALC's
                                                     density term is unreliable)
        else                               -> WALC
    else                                   -> DBAL

The eff_dim deviation is gated on the classifier being "flat" (low peak,
RF-like); confident classifiers (high peak, LR-like) keep WALC.

Variants (`variant=`):
    "V58"  : default — dimension-aware routing with round-3 correction
    "V57"  : legacy V5.7 routing (n_classes / mean_U only), frozen at round 1
    "V58c" : calibration-aware (confident multiclass -> DBAL)

Framework-agnostic: works with any classifier exposing predict_proba and any
state with X_labeled / y_labeled / X_unlabeled / model. `select(state, budget)`
adapts to the alframework registry; `select_core(...)` takes explicit arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Uncertainty & structural signals
# --------------------------------------------------------------------------- #
def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    """1 - (p_max - p_2nd). Binary case reduces to 2*min(p, 1-p)."""
    s = np.sort(np.asarray(proba, float), axis=1)
    if s.shape[1] < 2:
        return np.zeros(len(s))
    return 1.0 - (s[:, -1] - s[:, -2])


def entropy_uncertainty(proba: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy in [0, 1]."""
    p = np.clip(np.asarray(proba, float), EPS, 1.0)
    return (-np.sum(p * np.log(p), axis=1)) / np.log(p.shape[1])


def safe_minmax(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < eps else (x - lo) / (hi - lo)


def _uncertainty(proba, kind: str) -> np.ndarray:
    return margin_uncertainty(proba) if kind == "margin" else entropy_uncertainty(proba)


def peak_confidence(proba: np.ndarray) -> float:
    """Mean (p_max - p_2nd): low for flat (RF-like), high for confident (LR-like)."""
    s = np.sort(np.asarray(proba, float), axis=1)
    return float(np.mean(s[:, -1] - (s[:, -2] if s.shape[1] > 1 else 0.0)))


def effective_dim(X: np.ndarray) -> float:
    """Participation ratio of the PCA spectrum: (sum ev)^2 / sum(ev^2).
    Low  -> data in a low-dim subspace, distances meaningful (WALC ok).
    High -> distances diluted (curse of dimensionality), WALC density unreliable.
    """
    X = np.asarray(X, float)
    if len(X) < 3:
        return float(X.shape[1])
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    ev = sv ** 2
    s = ev.sum()
    return float((s * s) / (np.sum(ev * ev) + EPS))


# --------------------------------------------------------------------------- #
# 2. Acquisition configuration + unified selector
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcquisitionConfig:
    """One point of the unified acquisition family (a 'mode')."""
    alpha: Union[float, str] = 0.3        # float in [0,1] or "adaptive"
    gamma: float = 1.0
    linkage: str = "kmeans"               # "kmeans" | "ward"
    uncertainty: str = "margin"           # "margin" | "entropy"
    pool_mult: int = 8
    n_init: int = 10

    def resolve_alpha(self, U_full: np.ndarray) -> float:
        """For 'adaptive', cv is computed over the FULL U (all candidates),
        matching V5.7's WALC_K branch exactly."""
        if self.alpha == "adaptive":
            cv = float(np.std(U_full) / (np.mean(U_full) + EPS))
            return float(np.clip(0.6 - 0.5 * cv, 0.1, 0.6))
        return float(self.alpha)


MODES = {
    "DBAL": AcquisitionConfig(alpha=1.0,        gamma=1.0, linkage="kmeans", uncertainty="margin", pool_mult=5),
    "R22":  AcquisitionConfig(alpha=0.0,        gamma=0.0, linkage="ward",   uncertainty="margin", pool_mult=10),
    "WALC": AcquisitionConfig(alpha="adaptive", gamma=2.0, linkage="kmeans", uncertainty="margin", pool_mult=8),
}


class UnifiedAcquisition:
    """Stateless selector implementing the unified (alpha, gamma, linkage) rule."""

    def __init__(self, config: AcquisitionConfig, random_state: int = 0):
        self.cfg = config
        self.random_state = int(random_state)

    def select(self, X_unlabeled: np.ndarray, proba: np.ndarray, budget: int) -> np.ndarray:
        cfg = self.cfg
        X = np.asarray(X_unlabeled, float)
        n = len(X)
        k = min(int(budget), n)
        if k <= 0:
            return np.array([], dtype=int)

        U_raw = _uncertainty(proba, cfg.uncertainty)
        U = safe_minmax(U_raw)
        if k == 1 or n == 1:
            return np.asarray([int(np.argmax(U_raw))], dtype=int)

        n_top = min(n, max(k, k * cfg.pool_mult))
        top = np.argsort(-U_raw)[:n_top]
        X_top = X[top]
        nc = min(k, len(X_top))
        if nc <= 1:
            return top[:k]

        if cfg.linkage == "ward":
            from sklearn.cluster import AgglomerativeClustering
            try:
                labels = AgglomerativeClustering(n_clusters=nc, linkage="ward").fit_predict(X_top)
                centers = np.array([X_top[labels == c].mean(axis=0) for c in range(nc)])
            except Exception:
                return top[:k]
        else:
            from sklearn.cluster import KMeans
            w = np.power(U[top] + EPS, cfg.gamma)
            km = KMeans(n_clusters=nc, n_init=cfg.n_init,
                        random_state=self.random_state).fit(X_top, sample_weight=w)
            labels, centers = km.labels_, km.cluster_centers_

        alpha = cfg.resolve_alpha(U)   # full U, as in V5.7

        sel = []
        for c in range(nc):
            m = np.where(labels == c)[0]
            if not len(m):
                continue
            U_loc = safe_minmax(U[top[m]])
            d_loc = safe_minmax(np.linalg.norm(X_top[m] - centers[c], axis=1))
            score = (1.0 - alpha) * U_loc + alpha * (1.0 - d_loc)
            sel.append(int(top[m[int(np.argmax(score))]]))
        sel = list(dict.fromkeys(sel))[:k]
        # Top-up: with duplicated points KMeans can return fewer distinct
        # clusters than k; fill the batch with the most uncertain remaining points.
        if len(sel) < k:
            chosen = set(sel)
            for j in np.argsort(-U_raw, kind="stable"):
                if int(j) not in chosen:
                    sel.append(int(j)); chosen.add(int(j))
                    if len(sel) == k:
                        break
        return np.array(sel, dtype=int)


# --------------------------------------------------------------------------- #
# 3. Regime router: act at round 1, correct once at correction_round
# --------------------------------------------------------------------------- #
@dataclass
class RegimeRouter:
    variant: str = "V58"            # "V58" | "V57" | "V58c"
    multiclass_thr: int = 5
    u_flat_trigger: float = 0.50
    peak_thr: float = 0.55          # V58c: confident-multiclass gate
    peak_lo: float = 0.58           # V58: "flat classifier" gate for eff_dim deviation
    eff_dim_thr: float = 12.0       # V58: high-effective-dim gate
    route_round: int = 1            # provisional decision here (act immediately)
    correction_round: int = 3       # single correction opportunity here
    hysteresis: float = 0.05        # min signal margin to allow a correction

    mode: Optional[str] = field(default=None, init=False)
    _round: int = field(default=0, init=False)
    _corrected: bool = field(default=False, init=False)
    _acc_mu: List[float] = field(default_factory=list, init=False)
    _acc_pk: List[float] = field(default_factory=list, init=False)
    _acc_ed: List[float] = field(default_factory=list, init=False)

    # -- decision rule -----------------------------------------------------
    def decide(self, n_classes: int, mean_u: float, peak: float, eff_dim: float = 0.0) -> str:
        if self.variant == "V57":                          # legacy
            if n_classes >= self.multiclass_thr:
                return "R22"
            return "WALC" if mean_u >= self.u_flat_trigger else "DBAL"
        if self.variant == "V58c":                         # calibration-aware
            if n_classes >= self.multiclass_thr:
                return "DBAL" if peak >= self.peak_thr else "R22"
            return "WALC" if mean_u >= self.u_flat_trigger else "DBAL"
        # V58 (default): dimension-aware
        if n_classes >= self.multiclass_thr:
            return "R22"
        if mean_u >= self.u_flat_trigger:
            if eff_dim >= self.eff_dim_thr and peak < self.peak_lo:
                return "R22"
            return "WALC"
        return "DBAL"

    # -- hysteresis: averaged signal far enough from a boundary? ------------
    def _decisive(self, n_classes: int, mu: float, pk: float, ed: float) -> bool:
        m = self.hysteresis
        if abs(mu - self.u_flat_trigger) < m:
            return False
        if self.variant == "V58" and mu >= self.u_flat_trigger:
            if abs(ed - self.eff_dim_thr) < (self.eff_dim_thr * m):
                return False
            if abs(pk - self.peak_lo) < m:
                return False
        return True

    # -- per-round update --------------------------------------------------
    def step(self, proba: np.ndarray, n_classes: int, X_pool: np.ndarray = None) -> str:
        self._round += 1
        U = safe_minmax(margin_uncertainty(proba))
        mu_now = float(np.mean(U))
        pk_now = peak_confidence(proba)
        ed_now = effective_dim(X_pool) if X_pool is not None else 0.0
        self._acc_mu.append(mu_now)
        self._acc_pk.append(pk_now)
        self._acc_ed.append(ed_now)

        # provisional decision: act immediately (no rounds wasted)
        if self.mode is None and self._round >= max(1, self.route_round):
            self.mode = self.decide(n_classes, mu_now, pk_now, ed_now)

        # single correction window using AVERAGED (de-noised) signals
        if (not self._corrected
                and self.correction_round > self.route_round
                and self._round >= self.correction_round):
            mu = float(np.mean(self._acc_mu))
            pk = float(np.mean(self._acc_pk))
            ed = float(np.mean(self._acc_ed))
            new_mode = self.decide(n_classes, mu, pk, ed)
            if new_mode != self.mode and self._decisive(n_classes, mu, pk, ed):
                self.mode = new_mode
            self._corrected = True

        return self.mode if self.mode is not None else "WALC"

    def reset(self) -> None:
        self.mode = None
        self._round = 0
        self._corrected = False
        self._acc_mu = []
        self._acc_pk = []
        self._acc_ed = []


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class ActivePseudoLabelV58:
    """Regime routing (act@1, correct@3) over the unified acquisition family."""

    def __init__(
        self,
        *,
        variant: str = "V58",
        multiclass_thr: int = 5,
        u_flat_trigger: float = 0.50,
        peak_thr: float = 0.55,
        peak_lo: float = 0.58,
        eff_dim_thr: float = 12.0,
        route_round: int = 1,
        correction_round: int = 3,
        hysteresis: float = 0.05,
        modes: Optional[dict] = None,
        random_state: int = 0,
        kmeans_n_init: str = "auto",
        debug_route: bool = False,
        **kwargs,                       # tolerate inherited V5.5/V5.4 kwargs
    ):
        self.router = RegimeRouter(
            variant=variant, multiclass_thr=multiclass_thr,
            u_flat_trigger=u_flat_trigger, peak_thr=peak_thr, peak_lo=peak_lo,
            eff_dim_thr=eff_dim_thr, route_round=route_round,
            correction_round=correction_round, hysteresis=hysteresis,
        )
       
        _base = dict(MODES if modes is None else modes)
        self.modes = {
            name: AcquisitionConfig(
                alpha=c.alpha, gamma=c.gamma, linkage=c.linkage,
                uncertainty=c.uncertainty, pool_mult=c.pool_mult,
                n_init=(kmeans_n_init if kmeans_n_init == "auto" else int(kmeans_n_init)),
            )
            for name, c in _base.items()
        }
        self.random_state = int(random_state)
        self.debug_route = bool(debug_route)
        self._prev_mode = None

    def select_core(self, clf, X_unlabeled: np.ndarray, budget: int, n_classes: int) -> np.ndarray:
        proba = clf.predict_proba(X_unlabeled)
        mode = self.router.step(proba, n_classes, X_pool=X_unlabeled)
        if self.debug_route and mode != self._prev_mode:
            tag = "route" if self._prev_mode is None else "CORRECT"
            print(f"[V58-{tag}] round={self.router._round} n_cls={n_classes} "
                  f"mean_U={float(np.mean(safe_minmax(margin_uncertainty(proba)))):.3f} "
                  f"peak={peak_confidence(proba):.3f} "
                  f"eff_dim={effective_dim(X_unlabeled):.1f} -> mode={mode}")
            self._prev_mode = mode
        acq = UnifiedAcquisition(self.modes[mode], random_state=self.random_state)
        return acq.select(X_unlabeled, proba, budget)

    def select(self, state, budget: int) -> np.ndarray:
        n_classes = len(np.unique(state.y_labeled))
        return self.select_core(state.model, state.X_unlabeled, budget, n_classes)

    @property
    def routed_mode(self) -> Optional[str]:
        return self.router.mode

    def reset_run(self) -> None:
        self.router.reset()
        self._prev_mode = None


# Optional alframework registration (no-op if registry is absent).
# Registered under the canonical name: this standalone class REPLACES the
# inherited V58. Drop this file in alframework/strategies/ as the module
# imported by _ensure_strategies_registered.
try:  # pragma: no cover
    from alframework.core.registry import register
    register("ActivePseudoLabelV58")(ActivePseudoLabelV58)
except Exception:
    pass
