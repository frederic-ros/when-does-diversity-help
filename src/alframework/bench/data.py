"""Data preparation for the benchmark.

Everything that defines *which* points a run sees is computed here, from the
split's seed streams only. It never depends on the strategy or the classifier,
so all strategies and both classifiers of a (dataset, split) share exactly the
same pool, test set and initial labelled set (checked by check.py via hashes).

Per split s (root seed = base + stride*s):
  1. synthetic : regenerate the dataset with make_classification(data_seed)
     tabular/latent : stratified subsample of at most `max_samples` points
  2. stratified train/test split (test_size)
  3. preprocessing fit on TRAIN only: median imputation, standard scaling
  4. stratified initial labelled set of size n_init, >= 1 point per class,
     remaining slots allocated proportionally to class frequency
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .provenance import file_sha256, hash_array, hash_obj

STREAMS = ("data", "split", "init", "model", "strategy")
PARSER_VERSION = 2  # bump whenever _parse_txt changes (invalidates the disk cache)


def split_seeds(base: int, stride: int, split: int) -> Dict[str, int]:
    """Independent, reproducible 32-bit seeds for each random stream of a split."""
    root = base + stride * split
    children = np.random.SeedSequence(root).spawn(len(STREAMS))
    seeds = {name: int(c.generate_state(1)[0]) for name, c in zip(STREAMS, children)}
    seeds["root"] = root
    return seeds


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
def load_txt(path: str, cache_dir: Optional[str] = None):
    """Parsed arrays, cached on disk (keyed by the file's SHA-256) so that the
    text file is parsed once per campaign rather than once per run."""
    if cache_dir:
        cp = Path(cache_dir) / f"{file_sha256(path)}_p{PARSER_VERSION}.npz"
        if cp.exists():
            z = np.load(cp)
            return z["X"], z["y"]
        X, y = _parse_txt(path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(cp.stem + ".tmp.npz")
        np.savez(tmp, X=X, y=y)
        import os
        os.replace(tmp, cp)
        return X, y
    return _parse_txt(path)


@lru_cache(maxsize=2)
def _parse_txt(path: str):
    """Tab-separated file, label = last column. Returns float X (NaN kept), int y.

    Non-numeric columns (<80% parseable) are ordinal-encoded; missing values are
    left as NaN and imputed later on the training split only.
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="c",
                     keep_default_na=False)
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.apply(lambda c: c.str.strip())
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan,
                     "NULL": np.nan, "?": np.nan})
    lab = df.shape[1] - 1
    keep = df.iloc[:, lab].notna()
    df = df.loc[keep].reset_index(drop=True)
    y = LabelEncoder().fit_transform(df.iloc[:, lab].to_numpy()).astype(int)
    cols = []
    for c in range(lab):
        s = df.iloc[:, c].str.replace(",", ".", regex=False)
        num = pd.to_numeric(s, errors="coerce")
        present = s.notna().sum()
        if present == 0 or num.notna().sum() / present >= 0.80:
            cols.append(num.to_numpy(float))
        else:
            codes, _ = pd.factorize(s, sort=True)  # NaN -> -1
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
    X = np.column_stack(cols).astype(float)
    # drop columns with no observed value (e.g. trailing tab in the file)
    X = X[:, ~np.all(np.isnan(X), axis=0)]
    return X, y


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def stratified_subsample(y: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a stratified subsample without replacement (>=1 per class)."""
    n = len(y)
    if size is None or size >= n:
        return np.arange(n)
    classes, counts = np.unique(y, return_counts=True)
    alloc = np.maximum(1, np.floor(counts / n * size).astype(int))
    alloc = np.minimum(alloc, counts)
    # distribute the remainder to the largest fractional parts
    rest = size - alloc.sum()
    if rest > 0:
        frac = counts / n * size - np.floor(counts / n * size)
        for j in np.argsort(-frac):
            if rest == 0:
                break
            if alloc[j] < counts[j]:
                alloc[j] += 1
                rest -= 1
    idx = [rng.choice(np.flatnonzero(y == c), size=a, replace=False)
           for c, a in zip(classes, alloc)]
    return np.sort(np.concatenate(idx))


def stratified_init(y: np.ndarray, n_init: int, rng: np.random.Generator) -> np.ndarray:
    """>=1 point per class, remaining slots proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    if n_init < len(classes):
        raise ValueError(f"n_init={n_init} < n_classes={len(classes)}")
    alloc = np.ones(len(classes), int)
    extra = n_init - len(classes)
    if extra > 0:
        share = counts / counts.sum() * extra
        add = np.floor(share).astype(int)
        rem = extra - add.sum()
        add[np.argsort(-(share - add))[:rem]] += 1
        alloc += add
    alloc = np.minimum(alloc, counts)
    idx = [rng.choice(np.flatnonzero(y == c), size=a, replace=False)
           for c, a in zip(classes, alloc)]
    return np.sort(np.concatenate(idx))


def dataset_n_init(rcfg: dict, dataset: str) -> int:
    return int(rcfg.get("dataset_overrides", {}).get(dataset, {}).get("n_init", rcfg["n_init"]))


@dataclass
class SplitData:
    X_pool: np.ndarray          # full training pool (init + unlabelled), preprocessed
    y_pool: np.ndarray          # oracle labels (used by the simulated labeller only)
    X_test: np.ndarray
    y_test: np.ndarray
    init_idx: np.ndarray        # indices into X_pool
    n_classes: int
    provenance: dict


def prepare_split(regime: str, dataset: str, rcfg: dict, data_root: Optional[Path],
                  seeds: Dict[str, int], cache_dir: Optional[str] = None) -> SplitData:
    prov = {}
    if regime == "synthetic":
        params = dict(rcfg["datasets"][dataset])
        X, y = make_classification(random_state=seeds["data"], **params)
        prov["data_source"] = {"generator": "make_classification", "params": params,
                               "random_state": seeds["data"]}
        sub = np.arange(len(y))
    else:
        rel = rcfg["datasets"][dataset]
        path = Path(data_root) / rcfg["data_dir"] / rel
        if not path.exists():
            raise FileNotFoundError(path)
        X_all, y_all = load_txt(str(path), cache_dir)
        prov["data_source"] = {"file": str(Path(rcfg["data_dir"]) / rel),
                               "parser_version": PARSER_VERSION,
                               "sha256": file_sha256(str(path)),
                               "n_rows": int(len(y_all))}
        sub = stratified_subsample(y_all, rcfg.get("max_samples"),
                                   np.random.default_rng(seeds["data"]))
        X, y = X_all[sub], y_all[sub]

    # re-encode labels to 0..C-1 on the subsample (all classes kept by construction)
    y = LabelEncoder().fit_transform(y)
    n_classes = int(len(np.unique(y)))

    tr, te = train_test_split(np.arange(len(y)), test_size=rcfg["test_size"],
                              random_state=seeds["split"] % (2**32 - 1), stratify=y)
    tr, te = np.sort(tr), np.sort(te)

    imp = SimpleImputer(strategy="median").fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    Xtr = sc.transform(imp.transform(X[tr]))
    Xte = sc.transform(imp.transform(X[te]))
    ytr, yte = y[tr], y[te]

    init_idx = stratified_init(ytr, dataset_n_init(rcfg, dataset), np.random.default_rng(seeds["init"]))

    prov.update({
        "subsample_hash": hash_array(sub),
        "train_hash": hash_array(sub[tr]),
        "test_hash": hash_array(sub[te]),
        "init_hash": hash_array(sub[tr][init_idx]),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": int(X.shape[1]), "n_classes": n_classes,
        "class_counts_train": np.bincount(ytr, minlength=n_classes).tolist(),
    })
    prov["pairing_key"] = hash_obj([prov["train_hash"], prov["test_hash"], prov["init_hash"]])
    return SplitData(Xtr, ytr, Xte, yte, init_idx, n_classes, prov)
