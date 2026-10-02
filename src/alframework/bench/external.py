"""Adapters for acquisition strategies released in other packages.

DualSelect (Ros et al., Knowledge-Based Systems, accepted 2026) is released as
the `dualselect` package (github.com/frederic-ros/dualselect). Its strategies
use their own state object (indices relative to the full pool, oracle labels in
`y_pool`) and their own initialization. The adapter below runs the *unmodified*
DualSelectStrategy inside the benchmark engine, with three rules:

1. Shared initial set. The engine draws the initial labeled set (paired across
   strategies); DualSelect's own initialization is skipped. Every other part of
   the acquisition rule is the released code with the paper's hyper-parameters.
2. No label leakage. DualSelect reads `state.y_pool` but only at labeled indices,
   plus `np.unique(y_pool)` to get the set of classes. The adapter therefore
   passes a *masked* label vector: true labels on labeled points, and on
   unlabeled points a label already present in the labeled set. The class set is
   unchanged (all classes are in the initial set by construction) and no
   unlabeled label is visible. Test data are not passed.
3. Exact batch size. The released `_select_diverse_topk` skips empty k-means
   clusters and can return fewer than `batch` points when the pool contains
   duplicated rows. The protocol requires exactly `k` distinct points, so a
   short batch is topped up with the most uncertain remaining points (smallest
   margin of the current model). The number of topped-up points is logged.
"""
from __future__ import annotations

import numpy as np


def _dualselect_version() -> dict:
    try:
        import dualselect
        import subprocess
        from pathlib import Path
        root = Path(dualselect.__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    return {"package": "dualselect", "commit": commit}


class DualSelectAdapter:
    """Benchmark-engine strategy wrapping dualselect.strategy.dual_select.DualSelectStrategy."""

    def __init__(self, random_state: int = 0, batch: int = 10, n_init: int = 10,
                 cls: str = "dualselect.strategy.dual_select.DualSelectStrategy",
                 **params):
        import importlib
        mod, name = cls.rsplit(".", 1)
        Strategy = getattr(importlib.import_module(mod), name)
        self._rng = np.random.default_rng(int(random_state))
        self._inner = Strategy(batch=int(batch), n_init=int(n_init),
                               rng=self._rng, **params)
        self._inner._initialized = True          # rule 1: shared initial set
        self._pool_ids = None
        self.topped_up = 0
        self.version = dict(_dualselect_version(), cls=cls)

    def bind(self, X_pool: np.ndarray, y_pool: np.ndarray):
        """Called once by the engine with the full training pool."""
        self._X_pool = X_pool
        self._y_true = y_pool          # kept private; never exposed unmasked

    def select_global(self, labeled_ids: np.ndarray, model, batch: int) -> np.ndarray:
        from dualselect.core.state import ALState as DSState
        labeled_ids = np.asarray(labeled_ids, dtype=int)
        y_masked = np.full(len(self._y_true), self._y_true[labeled_ids[0]])
        y_masked[labeled_ids] = self._y_true[labeled_ids]           # rule 2
        st = DSState(X_pool=self._X_pool, y_pool=y_masked, X_test=None, y_test=None,
                     model=model, rng=self._rng, labeled_idx=labeled_ids.tolist())
        idx, _meta = self._inner.select(st)
        lab = set(labeled_ids.tolist())
        sel = [int(i) for i in dict.fromkeys(idx) if int(i) not in lab][:batch]
        if len(sel) < batch:                                          # rule 3
            unl = np.array([i for i in range(len(self._X_pool)) if i not in lab
                            and i not in set(sel)], dtype=int)
            p = np.sort(model.predict_proba(self._X_pool[unl]), axis=1)
            margin = p[:, -1] - p[:, -2] if p.shape[1] > 1 else np.zeros(len(unl))
            fill = unl[np.argsort(margin, kind="stable")[: batch - len(sel)]]
            self.topped_up += int(len(fill))
            sel += fill.tolist()
        return np.asarray(sel, dtype=int)
