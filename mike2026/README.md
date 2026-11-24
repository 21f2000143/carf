# MIKE breast-cancer multimodal prediction project

Research package for a MIKE/LNCS-style paper on **bounded fusion of pretreatment
MRI, microarray transcriptomics, and clinical features for pathological complete
response (pCR) prediction**. Historical RNA-seq experiments are retained separately.

## Current manuscript: bounded clinical-anchored fusion (revision 3)

The current `paper/main.tex` evaluates **CARF**, a clinical-anchored residual gate
for patient-linked I-SPY2/I-SPY1 MRI, microarray, and clinical inputs. Optional
modalities have explicit missingness masks and learned logit-influence budgets.
The six-coefficient convex objective combines modality dropout environments,
controlled corruption, and regularization. See `revision3/ANALYSIS_PLAN.md` for
fixed settings and `revision3/RESULTS.md` for generated results. The original
RNA-seq workflow below is a separate historical analysis.

With the existing processed inputs in `data/revision2/processed/`:

```bash
python -m pip install -r requirements-revision2.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_revision3_analysis.py
python scripts/validate_revision3.py
python scripts/make_revision3_assets.py
make paper
python -m unittest discover -s tests -v
```

The run performs 25 outer folds with inner OOF gate fitting, 13 matched models,
external corruption tests, 2,000 paired bootstrap resamples, and corrected
five-fold RPPA evaluation. Completed expert predictions are cached by source and
input hashes. Tables, figures, and numerical manuscript macros are generated
from `results/revision3/`; do not hand-edit reported values.

Revision 3 corrects missing-value propagation in expression ranks, moves gene
variance filtering inside every fit, and removes held-out fitting from the RPPA
comparison. Earlier `results/revision2/` and `results/revision3_initial_rank_bug/`
artifacts are superseded and must not be used for current claims. Gate-only
permutations in the old analysis are not a whole-pipeline significance test.
Previous external outcomes were known during methodological development, so the
new I-SPY1 comparisons are exploratory, not independent confirmation. The
influence bound guarantees stability relative to an available clinical anchor;
it does not guarantee better accuracy or clinical utility.

## Original RNA-seq contribution

The main ML contribution is **subtype-conditioned gated fusion**:
1. a transcriptomic expert predicts pCR from RNA-seq;
2. a clinical expert predicts pCR from routinely available clinical variables;
3. a leakage-safe meta-model learns how the relative value of the two experts changes
   across breast-cancer subtypes using only out-of-fold predictions from the training cohort.

The package also evaluates external generalisation, uncertainty-aware selective prediction,
permutation controls, subgroup performance, calibration, and standard baselines.

## Public datasets used

Primary discovery:
- GSE163882: 222 pretreatment breast-cancer biopsies; RNA-seq TPM; pCR/RD and clinical
  characteristics in GEO metadata.

Primary external validation:
- GSE240671: 95 pretreatment breast-cancer biopsies; bulk RNA-seq raw counts; RCB/clinical
  metadata in GEO. Post-treatment residual-tumour libraries are automatically excluded.

The scripts download processed expression matrices and GEO SOFT metadata directly from NCBI.
No controlled-access data are required for the main analysis.

## Quick start

```bash
conda env create -f environment.yml
conda activate mike-pcr
bash scripts/run_all.sh
```

For a faster smoke test:

```bash
python scripts/make_synthetic_data.py
python -m src.experiments --config configs/synthetic.yaml
```

For a reduced real-data run:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python -m src.experiments --config configs/quick.yaml
python scripts/make_paper_assets.py
```

Full analysis:

```bash
bash scripts/run_all.sh
```

## Outputs

- `results/metrics.csv` — model-level AUROC/AUPRC/MCC/balanced accuracy/Brier
- `results/predictions.csv` — sample-level predictions
- `results/permutation_test.csv` — label-permutation null
- `results/selective_prediction.csv` — uncertainty/coverage tradeoff
- `results/subgroup_metrics.csv` — performance by subtype
- `results/selected_genes.csv` — selected transcriptomic features
- `paper/generated/results_macros.tex` — LaTeX macros auto-populated from results
- `paper/generated/results_table.tex` — LaTeX table auto-populated from results
- `paper/figures/*.pdf` — diagrams and result figures
- `paper/main.tex` — initial LNCS manuscript

## MIKE / LNCS formatting

The manuscript uses Springer's `llncs` class and targets a 10-page conference paper.
The current MIKE series publishes in Springer LNCS/LNAI. Before final submission, verify
the page limit in the current MIKE call for papers, since conference-specific limits can change.

The current `paper/` directory includes the official Springer class and bibliography style.
Preserve those files unchanged. To retrieve the template in a fresh checkout, run:

```bash
bash scripts/get_lncs_template.sh
```

Then compile:

```bash
cd paper
latexmk -pdf main.tex
```

## Reproducibility notes

- All feature selection and hyperparameter choice occur inside training folds.
- The external cohort is never used for model/feature selection.
- Clinical columns are chosen from features common to both cohorts at runtime.
- Cohort-specific expression scaling is performed independently.
- The default full configuration uses 100 label permutations and 30 bootstrap models.
  Increase these for the final paper (e.g. 1000 permutations, 200 bootstraps) after the
  pipeline is verified on your server.
- pCR parsing is conservative. The preparation script writes a metadata audit CSV; inspect it
  before trusting final results.

## Suggested final-paper claim

Do not claim clinical utility from retrospective public cohorts. The defensible claim is:
**subtype-conditioned fusion improves or stabilises cross-cohort discrimination relative
to single-modality and naive-fusion baselines, if the actual experiment supports it.**

## Validation

Run regression tests and audit every experiment output table:

```bash
python -m unittest discover -s tests -v
python scripts/validate_results.py --config configs/synthetic.yaml
# After the full analysis completes:
python scripts/validate_results.py --config configs/full.yaml
```

`SMOKE_TEST.txt` records the current execution evidence. For bounded CPU use, set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8` before the run.
The full analysis includes nested tuning for every gate-training fold and can take
substantially longer than the quick configuration. Progress is recorded in `logs/`.

Expression sample names are matched to GEO titles and library descriptions. The
external count file is tab-separated despite its filename; its supplied Ensembl
identifiers are used to match the discovery matrix. Preparation retains 222 discovery
and 95 pretreatment external samples with the currently downloaded data.
