# CARF with anchor verification (study for `latex2/`)

New code only; `mike2026/`, `experiments/`, and `latex/` are read (data, loaders, parsers) but never modified.

| File | Role |
|---|---|
| `carf2.py` | Bounded anchored fusion for K optional modalities, anchor-verification gate (`AnchorContext`), swap/extreme robust environments, inner-CV budget selection |
| `experts.py` | Treatment-aware clinical anchor, RNA/MRI/RPPA experts, auxiliary HR/HER2 heads |
| `pipeline.py` | Study designs, cross-fitted expert predictions with caching (`results/cache/<design hash>/`) |
| `fusion.py`, `models.py` | Baselines (mean, stacking, mask-stacking, offset LR, robust offset LR, ModDrop MLP), the same baselines on gated experts (`Gated`), CARF and ablations |
| `run_main.py` | 5x5 internal CV on I-SPY2 with internal failures and semi-realistic mix-ups (accession offset, within-batch and within-subtype swaps); external I-SPY1 with output-level failures (4 designs); saves external per-scenario predictions for bootstrap intervals |
| `run_extra.py` | Input-level corruption, natural missingness, GSE25066, whole-pipeline permutation test |
| `make_assets.py`, `fig_architecture.py` | Tables, macros, and figures for `latex2/` |

Reproduce (about 1 h on 80 cores):

```bash
cd carf2
python run_main.py
python run_extra.py natural gse input perm
python fig_architecture.py && python make_assets.py
cd ../latex2 && latexmk -pdf main.tex supplement.tex
```

GSE25066 inputs (`data_gse25066/`) are downloaded from GEO (series matrix and GPL96 annotation).
All model settings are fixed in `models.py`; the budget B is the only tuned quantity, chosen by inner CV.
