"""One run = (regime, dataset, classifier, split, strategy) -> one JSON record.

Loop semantics:
  round 0 : fit on the initial set (n_init points), evaluate, log indicators
  round t : select `batch_size` points, label, refit, evaluate
  stop when n_labeled == max_budget (the final budget IS evaluated)

Failures are never silent: any exception produces a record with
status="failed" and the traceback, and the checker refuses incomplete cells.
"""
from __future__ import annotations

import importlib
import inspect
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from alframework.core.state import ALState

from .data import dataset_n_init, prepare_split, split_seeds
from .provenance import environment, git_info, hash_array, hash_obj

SCHEMA_VERSION = 2

STRATEGY_MODULES = [
    "random", "uncertainty", "coreset", "typiclust", "probcover", "badge",
    "unc_feature_kmeans", "diversity_optimized_batch", "qbc", "robust_qbc",
    "tri_committee", "adaptive_disagreement", "dbal", "rank2022", "strategy_v58",
]


def _registry():
    from alframework.core.registry import STRATEGIES
    for m in STRATEGY_MODULES:
        importlib.import_module(f"alframework.strategies.{m}")
    return STRATEGIES


class WALCFixed:
    """The WALC acquisition mode of the router, used as a fixed strategy."""

    def __init__(self, random_state: int = 0):
        from alframework.strategies.strategy_v58 import MODES, UnifiedAcquisition
        c = MODES["WALC"]
        self._acq = UnifiedAcquisition(c, random_state=int(random_state))

    def select(self, state, budget: int) -> np.ndarray:
        proba = state.model.predict_proba(state.X_unlabeled)
        return self._acq.select(state.X_unlabeled, proba, budget)


def build_strategy(reg_name: str, params: dict, seed: int, batch: int = None,
                   n_init: int = None):
    if reg_name == "bench.walc_fixed":
        return WALCFixed(random_state=seed)
    if reg_name == "external.dualselect":
        from .external import DualSelectAdapter
        return DualSelectAdapter(random_state=seed, batch=batch, n_init=n_init, **params)
    reg = _registry()
    if reg_name not in reg:
        raise KeyError(f"strategy '{reg_name}' not registered")
    cls = reg[reg_name]
    kw = dict(params)
    try:
        if "random_state" in inspect.signature(cls.__init__).parameters:
            kw["random_state"] = int(seed)
    except (TypeError, ValueError):
        pass
    return cls(**kw)


def build_model(ccfg: dict, seed: int):
    mod, name = ccfg["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod), name)
    return cls(random_state=int(seed), **ccfg["params"])


# --------------------------------------------------------------------------- #
# Indicators of the uncertain band
# --------------------------------------------------------------------------- #
def _margins(proba: np.ndarray) -> np.ndarray:
    p = np.sort(proba, axis=1)
    return p[:, -1] - p[:, -2] if p.shape[1] > 1 else np.ones(len(p))


def _eff_dim(X: np.ndarray) -> float:
    if len(X) < 3:
        return float("nan")
    ev = np.linalg.eigvalsh(np.cov(X.T))
    ev = ev[ev > 1e-12]
    return float(ev.sum() ** 2 / np.square(ev).sum()) if len(ev) else float("nan")


def band_indicators(proba: np.ndarray, X_pool: np.ndarray, bcfg: dict,
                    geometry: bool, seed: int) -> Dict[str, float]:
    m = _margins(proba)
    out = {"margin_mean": float(m.mean()), "margin_median": float(np.median(m))}
    for t in bcfg["taus"]:
        out[f"band_width_tau{t:g}"] = float((m < t).mean())
    if geometry:
        k = max(10, int(bcfg["geometry_top_fraction"] * len(m)))
        U = X_pool[np.argsort(m, kind="stable")[:k]]
        out["unc_eff_dim"] = _eff_dim(U)
        try:
            nk = min(4, max(2, len(U) // 50))
            km = KMeans(n_clusters=nk, n_init=3, random_state=int(seed) % (2**31)).fit(U)
            tot = np.square(U - U.mean(0)).sum() / len(U)
            out["unc_clusterability"] = float(1 - (km.inertia_ / len(U)) / tot) if tot > 0 else 0.0
        except Exception:
            out["unc_clusterability"] = float("nan")
    return out


def evaluate(model, X, y, n_classes) -> Dict[str, float]:
    yp = model.predict(X)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "f1_macro": float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, yp, labels=labels, average="weighted", zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_key(regime, dataset, clf, split, strategy) -> dict:
    return {"regime": regime, "dataset": dataset, "classifier": clf,
            "split": int(split), "strategy": strategy}


def execute(protocol: dict, protocol_hash: str, regime: str, dataset: str, clf: str,
            split: int, strategy: str, reg_name: str, sparams: dict,
            data_root: Optional[str], cache_dir: Optional[str] = None) -> Dict[str, Any]:
    rcfg = protocol["regimes"][regime]
    bcfg = protocol["uncertain_band"]
    seeds = split_seeds(protocol["seeds"]["base"], protocol["seeds"]["stride"], split)
    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key": run_key(regime, dataset, clf, split, strategy),
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "strategy_impl": {"registry_name": reg_name, "params": sparams},
        "classifier_impl": protocol["classifiers"][clf],
        "seeds": seeds,
        "code": git_info(),
        "environment": environment(),
        "status": "running",
    }
    t_start = time.perf_counter()
    try:
        sd = prepare_split(regime, dataset, rcfg, data_root, seeds, cache_dir)
        rec["data"] = sd.provenance

        n_init, bsz, bmax = dataset_n_init(rcfg, dataset), rcfg["batch_size"], rcfg["max_budget"]
        if (bmax - n_init) % bsz:
            raise ValueError("max_budget - n_init must be a multiple of batch_size")
        n_rounds = (bmax - n_init) // bsz

        pool_ids = np.arange(len(sd.y_pool))           # stable ids within the pool
        lab = np.zeros(len(pool_ids), bool)
        lab[sd.init_idx] = True

        base_model = build_model(protocol["classifiers"][clf], seeds["model"])
        strat = build_strategy(reg_name, sparams, seeds["strategy"], batch=bsz, n_init=n_init)
        if hasattr(strat, "bind"):
            strat.bind(sd.X_pool, sd.y_pool)
            rec["strategy_impl"]["external"] = strat.version
        rng = np.random.default_rng(seeds["strategy"])
        np.random.seed(seeds["strategy"] % (2**32))   # guard against legacy global-RNG use

        history = []
        for r in range(n_rounds + 1):
            L, U = pool_ids[lab], pool_ids[~lab]
            model = clone(base_model)
            t0 = time.perf_counter()
            model.fit(sd.X_pool[L], sd.y_pool[L])
            t_fit = time.perf_counter() - t0

            step = {"round": r, "n_labeled": int(lab.sum()), "t_fit": t_fit,
                    **evaluate(model, sd.X_test, sd.y_test, sd.n_classes)}
            if len(U) >= 2:
                proba_pool = model.predict_proba(sd.X_pool[U])
                step.update(band_indicators(proba_pool, sd.X_pool[U], bcfg,
                                            geometry=r in bcfg["geometry_rounds"],
                                            seed=seeds["strategy"]))
            if r == n_rounds:
                history.append(step)
                break

            state = ALState(X_labeled=sd.X_pool[L], y_labeled=sd.y_pool[L],
                            X_unlabeled=sd.X_pool[U], model=model, rng=rng,
                            X_test=None, y_test=None)      # no test data for strategies
            state.n_classes = sd.n_classes
            state.labels = np.arange(sd.n_classes)
            state.all_labels = np.arange(sd.n_classes)
            # deliberately NOT set: state.y_pool (oracle labels of the pool)

            t0 = time.perf_counter()
            if hasattr(strat, "select_global"):      # external strategy, pool ids
                g = strat.select_global(L, model, bsz)
                pos = {int(u): i for i, u in enumerate(U)}
                if any(int(x) not in pos for x in g):
                    raise RuntimeError("external strategy selected a labeled point")
                sel = np.asarray([pos[int(x)] for x in g], dtype=int)
                step["topped_up_total"] = int(strat.topped_up)
            else:
                sel = np.asarray(strat.select(state, budget=bsz), dtype=int).ravel()
            step["t_select"] = time.perf_counter() - t0
            if len(sel) != bsz or len(np.unique(sel)) != bsz or sel.min() < 0 or sel.max() >= len(U):
                raise RuntimeError(f"invalid selection at round {r}: size={len(sel)}, "
                                   f"unique={len(np.unique(sel))}, range=[{sel.min()},{sel.max()}]")
            chosen = U[sel]
            step["selected_hash"] = hash_array(np.sort(chosen))
            step["selected"] = chosen.tolist()
            if hasattr(strat, "routed_mode"):
                step["router_mode"] = strat.routed_mode
            lab[chosen] = True
            history.append(step)

        rec["history"] = history
        rec["status"] = "ok"
    except Exception as e:  # never silent
        rec["status"] = "failed"
        rec["error"] = {"type": type(e).__name__, "message": str(e),
                        "traceback": traceback.format_exc()}
    rec["wall_time_s"] = time.perf_counter() - t_start
    rec["result_hash"] = hash_obj([h.get("f1_macro") for h in rec.get("history", [])]
                                  + [h.get("selected_hash") for h in rec.get("history", [])])
    return rec
