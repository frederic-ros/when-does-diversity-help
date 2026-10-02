# Benchmark engine

The benchmark is run by **one engine** (`src/alframework/bench/`) driven by **one
protocol file** (`configs/protocol.json`). The engine sets up, executes, logs and
checks the runs; the acquisition strategies live in `src/alframework/strategies/`.

## Guarantees

| Property | How it is enforced |
|---|---|
| One protocol | Every setting lives in `configs/protocol.json`. Its hash is stored in every record, and records with another hash are re-run and never mixed. |
| One code version | Every record stores the git commit and a *dirty* flag. `check` refuses a campaign spanning several commits or uncommitted code. |
| True pairing | Subsample, train/test split and initial set depend only on the split seed. Their hashes form a `pairing_key` that must be identical for all strategies and both classifiers of a (dataset, split). |
| Independent randomness | Each split derives separate `SeedSequence` streams for data, split, init, model and strategy, so a strategy's internal randomness cannot shift the data. |
| Full budget grid | Evaluation runs at `n_init`, after every batch, **and** at `max_budget`. `check` verifies the exact grid. |
| No silent failure | Exceptions are stored in the record with their traceback (`status="failed"`). Invalid selections (wrong size, duplicates, out of range) raise errors. |
| No label leakage | Strategies receive `n_classes` and the labelled set only. Pool labels (`y_pool`) and test data are not exposed. |
| Preprocessing without leakage | Imputation and scaling are fit on the training split only. |
| Bit-exact reproducibility | Numerics are single-threaded (`OMP/MKL/OPENBLAS_NUM_THREADS=1`) and random forests use `n_jobs=1`. `repro` re-executes a sample and compares `result_hash` (curves plus selections). Bit-exactness holds on the same platform; across platforms, ties between equal scores may be broken differently (see `REPRODUCE.md`, Section 5). |
| In-run structural indicators | At every round the runner logs the margin mean/median and the uncertain-band width \|U_τ\|/\|pool\| for τ ∈ {0.05, 0.1, 0.2, 0.3}. At round 0 it also logs the effective dimension and clusterability of the lowest-margin 20 %. |
| Cost measured | Fit and selection time are logged per round. |
| Resumable | Records are written atomically, and completed records are skipped on relaunch. |

## Definitions (as implemented)

- **Margin**: m(x) = p₍₁₎(x) − p₍₂₎(x), the gap between the two largest predicted
  class probabilities.
- **Uncertain band**: U_τ = {x ∈ pool : m(x) < τ}, with primary τ = 0.1.
- **Band width**: |U_τ| / |pool|.

## Protocol summary (`configs/protocol.json`)

| Regime | Datasets | Splits | Test | n_init | Batch | Budget | Subsample |
|---|---|---|---|---|---|---|---|
| synthetic | 16 | 20 | 0.30 | 20 | 20 | 480 | — (regenerated per split) |
| tabular | 15 | 20 | 0.30 | 20 (letter: 40) | 20 | 400 | 3000 stratified |
| latent | 12 | 20 | 0.30 | 50 | 50 | 1200 | 5000 stratified |

- **Classifiers**: logistic regression (`max_iter=2000`) and random forest (50 trees).
- **Panel**: `main` holds the 17 strategies of the paper. `extras` holds the
  router (standalone V58), WALC as a fixed mode, **DualSelect** (Ros et al.,
  KBS 2026), and the DBAL implementation released with DualSelect (to reconcile
  the two papers' rankings). DualSelect runs the released `dualselect` package, pinned to the
  commit recorded in the protocol file, through `src/alframework/bench/external.py`.
  The adapter uses the shared initial set, masks the labels of unlabeled pool
  points (a unit test checks that permuting them does not change the selection),
  and enforces the exact batch size.
- **Size**: 43 datasets × 2 classifiers × 20 splits × 21 strategies = **36 120 runs**.
- **Seeds**: split `s` uses root seed `42 + 17·s`.
- **Letter recognition**: it has 26 classes, so it uses `n_init = 40`.
- **Latent splits**: 20 splits. To reduce cost, lower `regimes.latent.splits`
  *before* launching; it is part of the protocol hash.

## Workflow

```bash
pip install -e .
pip install "git+https://github.com/frederic-ros/dualselect@4f09ff6c16f65d061bd672d52d02f7206db75bb6"
git status            # must be clean: commit (and tag) before launching

# 0. size of the campaign
python benchmarks/run_benchmark.py plan

# 1. run (resumable; can be sharded across machines with --regime/--datasets/--splits;
#    all shards must use the same commit and protocol)
python benchmarks/run_benchmark.py run --regime synthetic tabular latent \
       --data-root data --out records --jobs 8

# 2. integrity check: must print "STATUS: CLEAN"
python benchmarks/run_benchmark.py check --regime synthetic tabular latent --out records

# 3. reproducibility spot-check (bit-exact re-execution of a random sample)
python benchmarks/run_benchmark.py repro --regime synthetic tabular latent \
       --out records --data-root data -n 30

# 4. analysis (steps 1-11)
AL_RECORDS=records AL_OUT=paper_outputs python analysis/run_all.py
```

`--data-root` must contain `tab/reeltabulaire/*.txt` and
`latent/latent_pca{10,20,50,100}H3/*.txt`; synthetic datasets are regenerated from
the protocol.

### Cost

Measured on one core:

| Regime | Classifier | Seconds per run |
|---|---|---|
| synthetic | LR | ~0.6 |
| synthetic | RF | ~5 |
| tabular | LR | ~1 |
| tabular | RF | ~4 |
| latent (1200 labels) | LR | ~1–20 (clustering and core-set methods are the slow ones) |
| latent (1200 labels) | RF | ~5–20 |

The full campaign takes a few tens of CPU-hours; with `--jobs 8` it runs in a few
hours on a workstation. Per-round selection times are stored in the records and
summarized in Table B.14 of the paper.

### Smoke test

`configs/protocol_smoke.json` is a tiny version (5 datasets, 3 splits, reduced
latent budget) for validating an installation end to end in about 30 minutes on
one core. Never use it for results.

```bash
python benchmarks/run_benchmark.py run --protocol configs/protocol_smoke.json \
       --regime synthetic tabular latent --data-root data --out smoke --jobs 2
python benchmarks/run_benchmark.py check --protocol configs/protocol_smoke.json \
       --regime synthetic tabular latent --out smoke
```

Step 1 of the analysis refuses to load a campaign that is not clean. Step 5 reads
the structural indicators logged by the runs.

## Record format (one JSON per run)

Path: `records/<regime>/<dataset>/<classifier>/splitNN/<strategy>.json`

Each record contains:

- the run `key`;
- `protocol_hash`, the strategy registry name and parameters, and the classifier parameters;
- all seeds;
- `code` (commit, branch, dirty flag) and `environment` (Python, NumPy, SciPy,
  scikit-learn, platform);
- `data`: the source file's SHA-256 or the generator parameters, the hashes of
  the subsample/train/test/initial set, the `pairing_key`, sizes and class counts;
- `history`: one entry per budget point with the metrics, band indicators,
  timings, selected pool ids and their hash, and the router mode where applicable;
- `status`, `wall_time_s` and `result_hash`.

## Archiving

After a clean `check`, archive `records/` (without `.data_cache/`),
`check_report.json`, the protocol file and the commit tag as a **new version**
of the Zenodo record. In the paper, cite the tag and the Zenodo version DOI.
