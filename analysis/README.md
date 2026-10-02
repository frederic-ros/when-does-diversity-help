# Analysis pipeline

Turns the run records into every table, figure and in-text statistic of the paper.
The usual entry point is `./reproduce_paper.sh` at the repository root; see
[`../REPRODUCE.md`](../REPRODUCE.md) for the mapping between output files and paper
artifacts.

## Run

```bash
pip install -r analysis/requirements.txt
export PYTHONPATH=src
AL_RECORDS=records AL_OUT=paper_outputs python analysis/run_all.py
AL_RECORDS=records AL_OUT=paper_outputs python analysis/run_all.py --from 6   # resume
AL_RECORDS=records AL_OUT=paper_outputs python analysis/run_all.py --only 7   # one step
```

Environment variables (defaults in `config.py`):

| Variable | Meaning | Default |
|---|---|---|
| `AL_RECORDS` | folder with `synthetic/ tabular/ latent/` records (real folders, not symlinks) | `<repo>/records` |
| `AL_PROTOCOL` | protocol file | `configs/protocol.json` |
| `AL_OUT` | output folder | `./out` |

Outputs: `<AL_OUT>/cache/` (intermediate parquet), `<AL_OUT>/figures/` (PNG, 300 dpi),
`<AL_OUT>/tables/` (LaTeX fragments, CSV and text reports).

## Steps

| Step | File | Main outputs |
|---|---|---|
| 1 | `step1_load.py` | checks the campaign (must be CLEAN); `cache/long_full.parquet`, `cache/long.parquet` (17-strategy panel) |
| 2 | `step2_saturation.py` | early-budget window n95 per setting |
| 3 | `step3_aulc.py` | windowed AULC per split |
| 4 | `step4_leaderboard.py` | ranks, win rate, regret, failure rate |
| 5 | `step5_structure.py` | structural indicators logged by the runs vs gain of diversity |
| 6 | `step6_figures.py` | `fig_spread`, `fig_risk`, `fig_curves`, `fig_showcase`, `fig_structure` |
| 7 | `step7_tables.py` | `tab1_leaderboard`, `tab2_perregime`, `tab3_collapse`, `tab6_struct_paper`, `tab_cost` |
| 8 | `step8_difficulty.py` | `tab4_difficulty` |
| 9 | `step9_classifier.py` | `tab5_classifier` |
| 10 | `step10_trend_robustness.py` | robustness of the structural trend (console) |
| 11 | `step11_statistics.py` | `fig_cd`, `stats_*` (windows, failure definitions, classifier shifts, Friedman/Wilcoxon/Nemenyi, trend robustness, router and additional strategies) |

The file names `tabN_*` are internal and do not follow the table numbering of the
paper; `REPRODUCE.md` gives the correspondence.

All steps read only the run records and the protocol file; no dataset is needed. The model tests of Section 4.4 need the feature files and are
run separately by `reproduce_paper.sh` (replay runners in `src/alframework/bench/`
and scripts in `tools/`).
