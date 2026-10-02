"""Fast unit tests for the benchmark engine (synthetic only, no data needed)."""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import dualselect  # noqa: F401
    _HAS_DUALSELECT = True
except ImportError:
    _HAS_DUALSELECT = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alframework.bench.campaign import enumerate_jobs, load_protocol  # noqa: E402
from alframework.bench.data import prepare_split, split_seeds, stratified_init  # noqa: E402
from alframework.bench.run import execute  # noqa: E402

PROTO, PHASH = load_protocol(str(ROOT / "configs" / "protocol.json"))
RC = PROTO["regimes"]["synthetic"]


def test_seed_streams_independent_and_stable():
    a, b = split_seeds(42, 17, 3), split_seeds(42, 17, 3)
    assert a == b and len({a[k] for k in ("data", "split", "init", "model", "strategy")}) == 5


def test_stratified_init_covers_all_classes():
    y = np.repeat(np.arange(5), [500, 50, 20, 5, 2])
    idx = stratified_init(y, 20, np.random.default_rng(0))
    assert len(idx) == 20 and set(y[idx]) == set(range(5))


def test_pairing_independent_of_strategy_and_classifier():
    s = split_seeds(42, 17, 0)
    d1 = prepare_split("synthetic", "imbalanced", RC, None, s)
    d2 = prepare_split("synthetic", "imbalanced", RC, None, s)
    assert d1.provenance["pairing_key"] == d2.provenance["pairing_key"]


def test_run_full_grid_deterministic_and_no_leak():
    reg, params = PROTO["panel"]["main"]["margin"]
    r1 = execute(PROTO, PHASH, "synthetic", "easy", "lr", 0, "margin", reg, params, None)
    r2 = execute(PROTO, PHASH, "synthetic", "easy", "lr", 0, "margin", reg, params, None)
    assert r1["status"] == "ok", r1.get("error")
    grid = [h["n_labeled"] for h in r1["history"]]
    assert grid == list(range(RC["n_init"], RC["max_budget"] + 1, RC["batch_size"]))
    assert r1["result_hash"] == r2["result_hash"]
    sel = np.concatenate([h["selected"] for h in r1["history"] if "selected" in h])
    assert len(sel) == len(set(sel))


def test_failure_is_recorded_not_raised():
    r = execute(PROTO, PHASH, "synthetic", "easy", "lr", 0, "x", "does_not_exist", {}, None)
    assert r["status"] == "failed" and "traceback" in r["error"]


def test_job_count():
    assert len(enumerate_jobs(PROTO, ["synthetic"])) == 16 * 2 * 20 * 21


def test_every_strategy_returns_exact_distinct_batch_on_duplicated_pool():
    """Every strategy returns exactly k distinct points, even on duplicated pools."""
    import warnings
    warnings.filterwarnings("ignore")
    from alframework.bench.campaign import panel
    from alframework.bench.run import build_model, build_strategy
    from alframework.core.state import ALState
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 5))
    X = np.repeat(base, 30, axis=0)                  # heavy duplication
    y = (base[:, 0] > 0).astype(int).repeat(30)
    y[::7] = 2
    lab = np.concatenate([np.flatnonzero(y == c)[:10] for c in range(3)])
    U = np.setdiff1d(np.arange(len(y)), lab)
    for name, (reg, params) in panel(PROTO, ["main", "extras"]).items():
        if reg.startswith("external.") and not _HAS_DUALSELECT:
            continue  # optional external package (pip install dualselect, see README)
        for clf in ("lr", "rf"):
            m = build_model(PROTO["classifiers"][clf], 0).fit(X[lab], y[lab])
            st = ALState(X[lab], y[lab], X[U], m, np.random.default_rng(1))
            st.n_classes, st.labels, st.all_labels = 3, np.arange(3), np.arange(3)
            strat = build_strategy(reg, params, 0, batch=50, n_init=30)
            if hasattr(strat, "select_global"):
                strat.bind(X, y)
                g = strat.select_global(lab, m, 50)
                pos = {int(u): i for i, u in enumerate(U)}
                s = np.asarray([pos[int(x)] for x in g])
            else:
                s = np.asarray(strat.select(st, 50)).ravel()
            assert len(s) == 50 and len(np.unique(s)) == 50, (name, clf, len(s), len(np.unique(s)))


def test_every_strategy_is_deterministic():
    """Two executions of the same run must give identical curves and selections."""
    from alframework.bench.campaign import panel
    for name, (reg, params) in panel(PROTO, ["main", "extras"]).items():
        if reg.startswith("external.") and not _HAS_DUALSELECT:
            continue  # optional external package (pip install dualselect, see README)
        for clf in ("lr", "rf"):
            a = execute(PROTO, PHASH, "synthetic", "medium5c", clf, 1, name, reg, params, None)
            b = execute(PROTO, PHASH, "synthetic", "medium5c", clf, 1, name, reg, params, None)
            assert a["status"] == "ok", (name, a.get("error"))
            assert a["result_hash"] == b["result_hash"], (name, clf)


def test_dualselect_adapter_no_label_leakage():
    """Permuting the labels of UNLABELED pool points must not change DualSelect's choice."""
    import pytest
    pytest.importorskip("dualselect")
    from alframework.bench.external import DualSelectAdapter
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    lab = np.concatenate([np.flatnonzero(y == 0)[:10], np.flatnonzero(y == 1)[:10]])
    unl = np.setdiff1d(np.arange(400), lab)
    y2 = y.copy(); y2[unl] = rng.permutation(y2[unl])
    out = []
    for yy in (y, y2):
        m = RandomForestClassifier(50, random_state=0).fit(X[lab], yy[lab])
        a = DualSelectAdapter(random_state=3, batch=20, n_init=20)
        a.bind(X, yy)
        out.append(a.select_global(lab, m, 20))
    assert np.array_equal(out[0], out[1])
    assert len(out[0]) == 20 and len(set(out[0])) == 20 and not set(out[0]) & set(lab)


def test_badge_returns_distinct_points_with_duplicated_probabilities():
    """Regression test: k-means++ seeding can return an index twice when many
    unlabeled points share the same gradient embedding."""
    from alframework.bench.run import build_strategy
    from alframework.core.state import ALState

    class ConstantProba:
        classes_ = np.arange(2)
        def fit(self, X, y): return self
        def predict_proba(self, X):
            p = np.tile(np.array([0.5, 0.5]), (len(X), 1))
            p[: len(X) // 2] = [0.9, 0.1]        # two distinct embeddings only
            return p
    X = np.zeros((300, 4)); y = np.zeros(300, int)
    st = ALState(X[:20], y[:20], X[20:], ConstantProba(), np.random.default_rng(0))
    st.n_classes, st.labels, st.all_labels = 2, np.arange(2), np.arange(2)
    s = np.asarray(build_strategy("badge_approx", {}, 0).select(st, 20)).ravel()
    assert len(s) == 20 and len(np.unique(s)) == 20


def test_committee_strategies_with_single_minority_example():
    """Regression test: with logistic regression, a plain
    bootstrap of 20 labeled points could miss a minority class and crash the fit."""
    import warnings
    warnings.filterwarnings("ignore")
    from alframework.bench.campaign import panel
    from alframework.bench.run import build_model, build_strategy
    from alframework.core.state import ALState
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)); y = (X[:, 0] > 1.9).astype(int)
    lab = np.concatenate([np.flatnonzero(y == 0)[:19], np.flatnonzero(y == 1)[:1]])
    U = np.setdiff1d(np.arange(600), lab)
    for name in ("qbc", "robust_qbc", "tri_committee", "adaptive_disagreement"):
        reg, params = panel(PROTO, ["main"])[name]
        for seed in range(10):
            m = build_model(PROTO["classifiers"]["lr"], 0).fit(X[lab], y[lab])
            st = ALState(X[lab], y[lab], X[U], m, np.random.default_rng(seed))
            st.n_classes, st.labels, st.all_labels = 2, np.arange(2), np.arange(2)
            s = np.asarray(build_strategy(reg, params, seed).select(st, 20)).ravel()
            assert len(s) == 20 and len(np.unique(s)) == 20, (name, seed)
