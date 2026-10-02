#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper
#   "When does diversity help in active learning? A large-scale empirical study of
#    uncertainty, diversity and worst-case risk"
# from the archived run records (Zenodo). See REPRODUCE.md for the mapping
# between outputs and paper artifacts.
#
# usage: ./reproduce_paper.sh <records dir> [data dir] [--skip-replay]
#   <records dir> : contains synthetic/ tabular/ latent/ (real folders, not symlinks)
#   [data dir]    : contains tab/reeltabulaire/ and latent/latent_pca*H3/ (default: data)
# Outputs go to $OUT (default: paper_outputs/).
set -euo pipefail
RECORDS=$(realpath "${1:?usage: reproduce_paper.sh <records dir> [data dir] [--skip-replay]}")
DATA=$(realpath "${2:-data}")
SKIP_REPLAY=${3:-}
OUT=$(realpath -m "${OUT:-paper_outputs}")
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

# single-threaded numerics, as in the campaign (bit-exact replays)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export AL_RECORDS="$RECORDS" AL_OUT="$OUT"

echo "== 1/2  Tables, figures and in-text statistics (about 10 min)"
(cd "$ROOT" && python analysis/run_all.py)

if [ "$SKIP_REPLAY" = "--skip-replay" ]; then
  echo "== 2/2  skipped (model tests of Section 4.4)"; exit 0
fi
echo "== 2/2  Model tests of Section 4.4 (replay: about 2 h + 15 min on 2 cores)"
cd "$ROOT"
python -m alframework.bench.replay --records "$RECORDS" --data-root "$DATA" \
       --out "$OUT/replay.parquet" --cache "$OUT/.cache_txt"
python -m alframework.bench.replay_imbalance --records "$RECORDS" --data-root "$DATA" \
       --out "$OUT/replay_imbalance.parquet" --cache "$OUT/.cache_txt"
python tools/analyze_model_tests.py --records "$RECORDS" --replay "$OUT/replay.parquet" \
       --out "$OUT/model_tests.txt"
python tools/analyze_imbalance.py "$RECORDS" "$OUT/replay_imbalance.parquet" \
       "$OUT/D_robustness.csv"
echo "done -> $OUT"
