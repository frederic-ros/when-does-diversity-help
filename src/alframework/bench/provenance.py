"""Provenance helpers: stable hashes, environment capture, code version."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(stable_json(obj).encode()).hexdigest()[:n]


def hash_array(a: np.ndarray, n: int = 16) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:n]


@lru_cache(maxsize=None)
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def git_info() -> dict:
    root = Path(__file__).resolve().parents[3]
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


@lru_cache(maxsize=1)
def environment() -> dict:
    import scipy
    import sklearn
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
