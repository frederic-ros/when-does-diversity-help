# Reproducing the paper

This page explains how to regenerate every table, figure and number of

> E. Bertrand-Bonvillain, Y. Nasser, F. Ros. *When does diversity help in active
> learning? A large-scale empirical study of uncertainty, diversity and worst-case
> risk.* Submitted to Neurocomputing.

There are three levels, from the cheapest to the most complete.

| Level | What it does | Inputs | Time |
|---|---|---|---|
| A | Regenerates all tables, figures and statistics from the run records | run records | ~10 min |
| B | Adds the model tests of Section 4.4 (replay of the runs) | run records + feature files | ~2 h 15 on 2 cores |
| C | Re-runs the benchmark itself (36,120 runs) | feature files | days on a workstation |

## 1. Install

```bash
git clone https://github.com/frederic-ros/when-does-diversity-help.git
cd when-does-diversity-help
git checkout v1.0.0
pip install -e .
pip install -r analysis/requirements.txt
```

The campaign was run with Python 3.11, numpy 2.4, scipy 1.17 and scikit-learn 1.9
(`environment.txt`).

## 2. Get the data (Zenodo, DOI in the paper)

```
records/                         one JSON record per run (36,120 files)
  synthetic/<dataset>/<lr|rf>/splitNN/<strategy>.json
  tabular/<dataset>/<lr|rf>/splitNN/<strategy>.json
  latent/<dataset>/<lr|rf>/splitNN/<strategy>.json
data/
  tab/reeltabulaire/<dataset>.txt         tabular datasets (label = last column)
  latent/latent_pca{10,20,50,100}H3/*.txt image embeddings (ResNet-18 + PCA)
```

`records/` must contain real folders (not symbolic links). Synthetic datasets are
regenerated from `configs/protocol.json`.

## 3. Run

```bash
./reproduce_paper.sh records data                 # levels A + B
./reproduce_paper.sh records data --skip-replay   # level A only
```

Outputs are written to `paper_outputs/` (`OUT=... ./reproduce_paper.sh ...` to change).

## 4. Where each paper artifact comes from

Files are relative to `paper_outputs/`. Step numbers refer to
`analysis/run_all.py`.

| Paper | Content | File | Produced by |
|---|---|---|---|
| Fig. 1 | Spread between strategies vs budget | `figures/fig_spread.png` | step 6 |
| Fig. 2 | Mean rank vs failure rate | `figures/fig_risk.png` | step 6 |
| Fig. 3 | Learning curves | `figures/fig_curves.png` | step 6 |
| Fig. 4 | Six illustrative settings | `figures/fig_showcase.png` | step 6 |
| Fig. 5 | Gain of pure diversity vs indicators | `figures/fig_structure.png` | step 6 |
| Fig. B.6 | Critical-difference diagram | `figures/fig_cd.png` | step 11 |
| Table 2 | Leaderboard (rank, win, regret, failure) | `tables/tab1_leaderboard.tex` | step 7 |
| Table 3 | Rank under three windows (extract of B.12) | `tables/stats_windows.csv` | step 11 |
| Table 4 | Mean rank per regime | `tables/tab2_perregime.tex` | step 7 |
| Table 5 | Failure rate per regime | `tables/tab3_collapse.tex` | step 7 |
| Table 6 | Indicators vs gain of diversity | `tables/tab6_struct_paper.tex` | step 7 |
| Table 7 | Mean rank by difficulty tier | `tables/tab4_difficulty.tex` | step 8 |
| Table 8 | LR to RF rank shifts | `tables/stats_classifier.csv` | step 11 |
| Table B.12 | Rank under five windows | `tables/stats_windows.csv` | step 11 |
| Table B.13 | Failure rate, alternative definitions | `tables/stats_collapse.csv` | step 11 |
| Table B.14 | Computation cost per strategy | `tables/tab_cost.tex` | step 7 |
| Table B.15 | Allocation-imbalance proxy | `D_robustness.tex` / `.csv` | `tools/analyze_imbalance.py` |
| Sec. 4.1 | Friedman, Wilcoxon-Holm, Nemenyi | `tables/stats_stats.txt` | step 11 |
| Sec. 4.3 | Robustness of the null result (per strategy, tau, mixed model, bootstrap) | `tables/stats_trend.txt` | step 11 |
| Sec. 4.4 | Testable implications (q, D) | `model_tests.txt` | `tools/analyze_model_tests.py` |
| Sec. 4.6 | Router, WALC, DBAL implementations | `tables/stats_router.txt`, `tables/stats_extras.txt` | step 11 |

Tables 1, 9, A.10, A.11 and D.16 are descriptive (notation, guide, protocol,
strategy definitions) and are not computed.

## 5. Checks

* `python benchmarks/run_benchmark.py check --regime synthetic tabular latent --out records`
  verifies completeness, pairing hashes and code versions of a campaign; step 1 of the
  analysis refuses an unclean campaign.
* The replay runner (level B) re-runs every recorded selection and checks that each
  replayed round reproduces the logged width of the uncertain band (99.8% of rounds
  match exactly; the rest differ by one point at the threshold, from numerical
  differences between platforms). The replay does not depend on how ties are broken,
  because it uses the recorded selections.
* `python benchmarks/run_benchmark.py repro ...` re-executes a random sample of runs
  from scratch. On the platform of the campaign (Windows 10, x86-64, Python 3.11,
  numpy 2.4.6, scikit-learn 1.9.1) re-execution is bit-exact. On another platform the
  initial model and its predictions are identical, but ties between equal scores
  (frequent with random forests, whose probabilities are multiples of 1/50, and with
  committee votes) can be broken in a different order by the sorting routines; the
  trajectory then diverges after the first tie. On Linux with the same library
  versions, 26 of 40 sampled runs were identical and the 14 others were all such
  tie cases. This affects individual trajectories, not the distribution of results.

## 6. Image embeddings

`tools/extract_embeddings.py` is the script that produced the image embeddings
(ResNet-18, torchvision `IMAGENET1K_V1` weights, PCA fitted on the official training
split). With `--bench-layout data/latent` it writes the files under the names used by
the protocol. The archived files are the reference inputs: their SHA-256 is stored in
every run record (`data.data_source.sha256`); re-extraction with other library
versions or on GPU may differ in the last digits.

## 7. Provenance of the run records

Each record stores the commit of the code that produced it (`code.commit`):
`9391c90` for all strategies except BADGE, `adb99b8` for BADGE. These commits
belong to the development repository of the project. This repository is its clean
public release: the benchmark engine (`src/alframework/bench/`), the strategies
(`src/alframework/strategies/`) and the protocol (`configs/protocol.json`, same
hash `016e1b9c8c8ee9b8`) are those used for the runs; only comments, file names
and the analysis/plotting code were reorganized. The replay runner (level B)
checks this directly, by re-running the recorded selections with the code of this
repository and comparing the logged band widths.
