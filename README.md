# When does diversity help in active learning?

Code, benchmark engine and analysis pipeline for

> E. Bertrand-Bonvillain, Y. Nasser, F. Ros.
> *When does diversity help in active learning? A large-scale empirical study of
> uncertainty, diversity and worst-case risk.* Submitted to *Neurocomputing*, 2026.

The study compares 16 batch active-learning strategies and random sampling on
86 settings (43 datasets × 2 classifiers) in three regimes: synthetic problems,
real tabular data and image embeddings (ResNet-18 + PCA). Each setting is run on
20 paired splits, for 36,120 runs in total including the additional strategies
discussed in the paper.

This repository contains everything needed to

1. **reproduce every table, figure and number of the paper** from the archived
   run records (one command, about 10 minutes);
2. **replay** the archived runs to compute the quantities of the allocation model
   (Section 4.4);
3. **re-run the benchmark** itself from a single protocol file.

The run records and the input data are archived on Zenodo
(DOI: [10.5281/zenodo.23114601](https://doi.org/10.5281/zenodo.23114601)).
See [`REPRODUCE.md`](REPRODUCE.md) for the step-by-step procedure and the mapping
between output files and paper artifacts.

## Quick start

```bash
git clone https://github.com/frederic-ros/when-does-diversity-help.git
cd when-does-diversity-help
pip install -e .
pip install -r analysis/requirements.txt

# records/ and data/ downloaded from Zenodo
./reproduce_paper.sh records data --skip-replay   # tables, figures, statistics
./reproduce_paper.sh records data                 # + model tests (replay, ~2 h)
```

Outputs are written to `paper_outputs/`.

## Repository layout

```
src/alframework/
  strategies/        the acquisition strategies (one file per family)
  core/              strategy registry and active-learning state
  bench/             benchmark engine: protocol, data preparation, runner,
                     integrity checks, replay runners
  utils/             bootstrap helpers
benchmarks/
  run_benchmark.py   command-line interface: plan | run | check | repro | export
configs/
  protocol.json      the protocol of the paper (single source of truth)
  protocol_smoke.json  tiny protocol to validate an installation
analysis/            pipeline that turns the records into tables and figures
  run_all.py         runs steps 1-11 in order
tools/
  analyze_model_tests.py   tests of the allocation model (Section 4.4)
  analyze_imbalance.py     robustness of the imbalance proxy (Table B.15)
  extract_embeddings.py    ResNet-18 + PCA image embeddings
tests/               unit tests of the engine and strategies
docs/BENCHMARK.md    benchmark design, guarantees and record format
reproduce_paper.sh   one-command reproduction
```

## Strategies

| Family | Strategies (name in the protocol and records) |
|---|---|
| Uncertainty | margin, entropy, least_confident |
| Pure diversity | coreset_greedy, coreset_kmeanspp, typiclust, probcover |
| Hybrid | badge, unc_feature_kmeans, diversity_opt_batch |
| Committee / disagreement | qbc, robust_qbc, tri_committee, adaptive_disagreement |
| Clustering of uncertain points | dbal, rank2022 |
| Baseline | random |

These 17 form the main panel of the paper. The protocol also runs four additional
strategies discussed in Section 4.6: the router (`router`), its fixed WALC mode
(`walc_fixed`), DualSelect (`dualselect`) and the DBAL implementation released with
DualSelect (`dbal_dualselect`). The last two use the external
[`dualselect`](https://github.com/frederic-ros/dualselect) package, pinned in
`configs/protocol.json`:

```bash
pip install "git+https://github.com/frederic-ros/dualselect@4f09ff6c16f65d061bd672d52d02f7206db75bb6"
```

It is needed only to re-run those two strategies; the analysis does not need it.

## Running the benchmark

```bash
python benchmarks/run_benchmark.py plan                       # size of the campaign
python benchmarks/run_benchmark.py run --regime synthetic tabular latent \
       --data-root data --out records --jobs 8                # resumable
python benchmarks/run_benchmark.py check --regime synthetic tabular latent \
       --out records                                          # must print STATUS: CLEAN
python benchmarks/run_benchmark.py repro --out records --data-root data -n 30
```

Every record stores the protocol hash, the code commit, all seeds, the data hashes
and the points queried at each round. Details in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Requirements

Python ≥ 3.10. The campaign was run with Python 3.11, numpy 2.4, scipy 1.17 and
scikit-learn 1.9 (full list in `environment.txt`). PyTorch is needed only to
re-extract the image embeddings (`tools/extract_embeddings.py`).

## Citation

```bibtex
@article{bertrandbonvillain2026diversity,
  title   = {When does diversity help in active learning? A large-scale empirical
             study of uncertainty, diversity and worst-case risk},
  author  = {Bertrand-Bonvillain, Erwan and Nasser, Yassine and Ros, Fr{\'e}d{\'e}ric},
  journal = {Neurocomputing (submitted)},
  year    = {2026}
}
```

Software citation metadata: [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Run records and data on Zenodo: CC-BY-4.0.
