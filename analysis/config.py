"""
config.py — paths, constants and strategy metadata of the analysis pipeline.

Paths are set by environment variables (or edit the defaults below):
    AL_RECORDS   directory of the run records: <regime>/<dataset>/<lr|rf>/splitNN/<strategy>.json
                 (default: <repository>/records)
    AL_PROTOCOL  protocol file (default: <repository>/configs/protocol.json)
    AL_OUT       output directory (default: ./out)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = Path(os.environ.get("AL_RECORDS", REPO / "records")).resolve()
PROTOCOL = Path(os.environ.get("AL_PROTOCOL", REPO / "configs" / "protocol.json")).resolve()

OUT_ROOT = Path(os.environ.get("AL_OUT", "./out")).resolve()
CACHE = OUT_ROOT / "cache"        # intermediate parquet tables
FIGDIR = OUT_ROOT / "figures"     # PNG figures
TABDIR = OUT_ROOT / "tables"      # LaTeX / CSV / text outputs
for _d in (CACHE, FIGDIR, TABDIR):
    _d.mkdir(parents=True, exist_ok=True)

METRICS = ["f1_macro", "balanced_accuracy"]
PRIMARY_METRIC = "f1_macro"

# Strategies run in the "extras" group of the protocol. They are excluded from
# the main ranking (steps 2-10) and analysed separately in step 11, each one
# inserted alone into the main panel.
EXTRAS = ["V58", "walc_fixed", "dualselect", "dbal_dualselect"]

STRAT_MAP = {
    "badge": "badge_approx", "router": "V58",
    "V58B": "V58", "ActivePseudoLabelV58": "V58",
    "diversity_opt_batch": "diversity_optimized_batch",
}

def norm_strat(s: str) -> str:
    return STRAT_MAP.get(s, s)

# Strategy -> family.
FAMILY = {
    "random": "random",
    "margin": "uncertainty", "entropy": "uncertainty", "least_confident": "uncertainty",
    "coreset_greedy": "diversity", "coreset_kmeanspp": "diversity",
    "typiclust": "diversity", "probcover": "diversity",
    "badge_approx": "hybrid", "unc_feature_kmeans": "hybrid",
    "diversity_optimized_batch": "hybrid",
    "qbc": "committee", "robust_qbc": "committee",
    "tri_committee": "committee", "adaptive_disagreement": "committee",
    "dbal": "clustering", "rank2022": "clustering", "V58": "router",
    "walc_fixed": "clustering", "dualselect": "hybrid", "dbal_dualselect": "clustering",
}

# Pretty names + short family labels (for tables/figures).
PRETTY = {
    "margin": "Margin", "entropy": "Entropy", "least_confident": "Least-confident",
    "random": "Random", "coreset_greedy": "Core-set (greedy)",
    "coreset_kmeanspp": "Core-set (k-means++)", "typiclust": "TypiClust",
    "probcover": "ProbCover", "badge_approx": "BADGE",
    "unc_feature_kmeans": "Unc.+feat. k-means", "diversity_optimized_batch": "Div.-opt. batch",
    "qbc": "QBC", "robust_qbc": "Robust QBC", "tri_committee": "Tri-committee",
    "adaptive_disagreement": "Adapt. disagree", "dbal": "DBAL",
    "rank2022": "Rank-2022", "V58": "Router (ours)", "walc_fixed": "WALC (fixed)",
    "dualselect": "DualSelect", "dbal_dualselect": "DBAL (DualSelect impl.)",
}
FAM_SHORT = {
    "uncertainty": "Unc.", "diversity": "Div.", "hybrid": "Hybrid",
    "committee": "Comm.", "clustering": "Clust.", "router": "Router", "random": "Rand.",
}

# The two diversity branches compared against margin in the structural analysis.
CLUSTERING_BRANCH = ["dbal", "rank2022"]   # the router is analysed separately (step 11)
PUREDIV_BRANCH = ["coreset_greedy", "coreset_kmeanspp", "typiclust", "probcover"]

