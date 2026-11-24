# CARF-3: review response (study for `latex3/`)

Successor to `carf2/` (which is unchanged and still produces `latex2/`). `carf3/` contains
`carf2/`'s full pipeline unmodified, plus `run_review.py` and `make_assets_review.py`, which add
the analyses requested by the MIKE 2026 review (`202-review-640.docx`; see
`../REVIEW_RESPONSE.md` for the full point-by-point mapping). `mike2026/`, `experiments/`, and
`latex2/` are read (data, loaders, parsers) but never modified; `carf2/results/cache/` was copied
over so the frozen experts and inner out-of-fold predictions do not need to be refit.

| File | Role |
|---|---|
| `carf2.py`, `common.py`, `experts.py`, `pipeline.py`, `fusion.py`, `models.py` | Unchanged from `carf2/` — bounded anchored fusion, experts, baselines, CARF and its ablations |
| `run_main.py`, `run_extra.py` | Unchanged from `carf2/` — main internal/external study and the input-corruption/natural-missingness/GSE25066/permutation experiments |
| `run_review.py` | **New.** Four review-response analyses on the frozen `carf2` design: (1) `anchor` — corrupts the clinical anchor itself (HR/HER2 charting errors, 5–20%, external, 30 reps); (2) `combo` — simultaneous modality failures (RNA swap + MRI noise/swap, RNA dropout + MRI dropout); (3) `calib` — post-hoc Platt recalibration (nested internal, external fit on pooled internal OOF) and a decision-curve analysis; (4) `mrigate` — exploratory I-SPY1-only MRI→T-stage verification gate |
| `make_assets.py`, `fig_architecture.py` | Unchanged from `carf2/` — tables, macros, and figures for `latex2/` |
| `make_assets_review.py` | **New.** Tables, macros, and the decision-curve figure for the new analyses, written to `latex3/generated/` and `latex3/figures/`; also reads the pre-existing, frozen `experiments/results/e8_gse20194/` (CARF v1, not rerun) to add a third external cohort (GSE20194) |

Reproduce (the main study is already cached from `carf2/`; the new analyses take a few minutes):

```bash
cd carf3
python run_main.py                                  # only if results/ was not copied from carf2/
python run_extra.py natural gse input perm           # only if results/ was not copied from carf2/
python run_review.py anchor combo calib mrigate      # new
python fig_architecture.py && python make_assets.py
python make_assets_review.py                         # new
cd ../latex3 && latexmk -pdf main.tex supplement.tex
```

All model settings are fixed in `models.py`, as in `carf2/`; the budget B is the only tuned
quantity, chosen by inner CV. The new analyses reuse the same frozen experts and fusion models
fit on all of I-SPY2 (`run_extra.frozen`), so none of them refit anything in `carf2/`'s cache.
