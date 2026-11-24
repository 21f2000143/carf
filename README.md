# CARF: Clinical-Anchored Robust Fusion

This repository contains the code, data-processing utilities, generated results, and
LaTeX sources for a research study of **bounded multimodal fusion** for pathological
complete response (pCR) prediction in breast cancer.

The current implementation, **CARF-3**, combines:

- pretreatment MRI-derived features;
- transcriptomic expression features;
- clinical variables used as an explicit anchor;
- modality-specific experts with explicit missingness handling; and
- a bounded residual fusion rule with learned per-modality influence budgets.

CARF is designed to make multimodal predictions auditable under missing, corrupted, or
misassigned inputs. Its guarantees concern stability relative to the clinical anchor;
they do **not** guarantee higher accuracy, clinical utility, or robustness when the
anchor itself is wrong.

## Repository status

The repository is organized around three related analysis generations:

| Directory | Purpose |
| --- | --- |
| [`carf3/`](carf3/) | Current CARF implementation and review-response analyses. |
| [`latex3/`](latex3/) | Current manuscript and supplement generated from `carf3/` outputs. |
| [`carf2/`](carf2/) | Frozen preceding implementation used to produce the original CARF study. |
| [`latex2/`](latex2/) | Manuscript and supplement corresponding to `carf2/`. |
| [`experiments/`](experiments/) | Additional robustness, calibration, external-cohort, and modality-failure experiments. |
| [`mike2026/`](mike2026/) | Earlier revision workflow, processed inputs, tests, and manuscript package. |
| [`latex/`](latex/) | Earlier manuscript, figures, generated tables, and calculation notes. |

For the current paper, use `carf3/` and `latex3/`. The older directories are retained
for provenance and reproducibility and should not be mixed with the current results.

## Main scientific contribution

CARF uses a clinical prediction as an anchor and expresses each optional modality as a
bounded residual contribution. For an anchor logit \(z_c\) and modality residuals
\(r_m\), the fused logit has the form

```text
z = z_c + sum_m alpha_m r_m
```

where the learned coefficients are constrained by an influence budget. The pipeline
also evaluates modality verification gates, dropout environments, controlled input
corruptions, natural missingness, calibration, and standard unbounded fusion baselines.

The study reports both positive and negative findings. CARF can preserve the clinical
anchor when optional modalities are absent and can limit their influence under selected
corruptions. However, it is not consistently more accurate than stacking or simpler
baselines, and it is less robust when the clinical anchor itself is corrupted. External
comparisons are exploratory because the available paired MRI, expression, and pCR data
are limited and earlier development iterations had access to I-SPY1 outcomes.

## Quick start: current CARF-3 pipeline

The current pipeline is a Python research workflow. The main and review scripts are
intended to be run from their own directories.

### Requirements

Use Python 3.11 with NumPy, pandas, SciPy, scikit-learn, matplotlib, PyYAML, joblib,
requests, tqdm, openpyxl, and GEOparse. The maintained environment specification is
[`mike2026/environment.yml`](mike2026/environment.yml).

```bash
conda env create -f mike2026/environment.yml
conda activate mike-pcr
python -m pip install -r mike2026/requirements-revision2.txt
```

If Conda is not available, install the packages listed in
[`mike2026/requirements.txt`](mike2026/requirements.txt) and
[`mike2026/requirements-revision2.txt`](mike2026/requirements-revision2.txt) in a
Python 3.11 virtual environment.

### Run the current main study

`carf3/` reuses the frozen expert predictions and cache produced by `carf2/` when those
artifacts are present. If the cache is not available, run the main study first.

```bash
cd carf3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_main.py
python run_extra.py natural gse input perm
```

The extra analyses cover natural missingness, the GSE25066 external cohort, input-level
corruption, and whole-pipeline label permutation.

### Run the review-response analyses

These analyses reuse the frozen CARF-2 expert cache and add:

- clinical-anchor corruption;
- simultaneous modality failures;
- post-hoc Platt calibration and decision-curve analysis; and
- an exploratory MRI-to-tumour-stage verification gate.

```bash
cd carf3
python run_review.py anchor combo calib mrigate
python make_assets_review.py
```

### Generate figures, tables, and the current manuscript

```bash
cd carf3
python fig_architecture.py
python make_assets.py
python make_assets_review.py

cd ../latex3
latexmk -pdf main.tex supplement.tex
```

The scripts write generated LaTeX macros and tables under `latex3/generated/` and
figures under `latex3/figures/`. Reported numerical values should be regenerated from
the result files rather than edited by hand.

## Reproducing the preceding CARF-2 study

CARF-2 is the frozen implementation corresponding to `latex2/`.

```bash
cd carf2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_main.py
python run_extra.py natural gse input perm
python fig_architecture.py
python make_assets.py

cd ../latex2
latexmk -pdf main.tex supplement.tex
```

The main run performs internal cross-validation on I-SPY2, external evaluation on
I-SPY1, controlled modality mix-ups, and cached cross-fitted expert predictions.
The `gse` analysis uses the GEO inputs in `carf2/data_gse25066/`; those files are
downloaded from GEO when absent.

## Earlier revision workflow

The `mike2026/` directory contains the earlier revision-specific data and analysis
workflow. Its current revision-3 entry points are:

```bash
cd mike2026
python -m pip install -r requirements-revision2.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/run_revision3_analysis.py
python scripts/validate_revision3.py
python scripts/make_revision3_assets.py
cd paper
latexmk -pdf main.tex
cd ..
python -m unittest discover -s tests -v
```

Equivalent Make targets are available:

```bash
cd mike2026
make analysis
make validate
make assets
make test
```

The revision-3 workflow performs nested gate fitting, external corruption tests,
paired bootstrap evaluation, and an RPPA comparison. See
[`mike2026/revision3/ANALYSIS_PLAN.md`](mike2026/revision3/ANALYSIS_PLAN.md) and
[`mike2026/revision3/RESULTS.md`](mike2026/revision3/RESULTS.md) for the frozen
settings and results.

## Data and cohorts

The repository includes processed research inputs and generated outputs, but some
raw inputs are downloaded from public repositories by the relevant scripts.

The main multimodal study uses:

- **I-SPY2** for internal development and cross-validation;
- **I-SPY1** for exploratory external evaluation;
- **GSE25066** for an RNA-only external analysis;
- **GSE20194** for an additional, partially overlapping RNA-only analysis; and
- **METABRIC** and CMU-MOSEI analyses for supplementary stress or transfer studies.

The paired I-SPY cohorts are the only part of the repository that evaluates the full
MRI-plus-expression-plus-clinical setting. GSE25066 and GSE20194 do not validate the
multimodal gate or anchor-failure behavior because they are RNA-only. GSE20194 also
substantially overlaps GSE25066.

No controlled-access data are required for the public-data workflows. Always inspect
the generated provenance, metadata-audit, and cohort-summary files before interpreting
results.

## Output locations

Important generated artifacts include:

- `carf3/results/` — current CARF-3 predictions, gates, stress analyses, and review results;
- `carf2/results/` — frozen CARF-2 predictions, caches, and tables;
- `experiments/results/` — supplementary experiment outputs;
- `mike2026/results/revision3/` — revision-3 metrics, predictions, provenance, and certificates;
- `latex3/generated/` — current manuscript macros and tables;
- `latex3/figures/` — current manuscript figures;
- `latex3/main.pdf` — current compiled manuscript;
- `latex3/supplement.pdf` — current compiled supplement.

Cache directories can be large. Do not delete or regenerate frozen caches when trying
to reproduce a manuscript unless you intend to rerun the corresponding analysis.

## Validation and tests

Run the focused tests for the maintained revision workflow with:

```bash
cd mike2026
python -m unittest discover -s tests -v
python scripts/validate_revision3.py
```

For manuscript or generated-result changes, compile the relevant paper and inspect the
generated tables and figures:

```bash
cd latex3
latexmk -pdf main.tex supplement.tex
```

The final review-response record is documented in
[`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md). It maps the review points to the new
`carf3/` analyses and confirms the compiled-PDF checks.

## Reproducibility rules

When extending the analysis:

1. Keep feature selection, model fitting, and budget selection inside training folds.
2. Never use external labels for model selection or hyperparameter tuning.
3. Preserve cohort and sample provenance in generated result files.
4. Generate manuscript numbers from result files; do not hand-type them into LaTeX.
5. Treat I-SPY1 comparisons as exploratory and report confidence intervals.
6. Distinguish simulated corruption from naturally observed missingness.
7. Treat the clinical anchor as an assumption that must be stress-tested.

The detailed calculation and reporting notes are in
[`latex/CALCULATION_FLOW.md`](latex/CALCULATION_FLOW.md),
[`latex/EXPERIMENTS_EXP_SETUP_PROTOCOL.md`](latex/EXPERIMENTS_EXP_SETUP_PROTOCOL.md),
and [`latex/VERIFIED_STATEMENTS.md`](latex/VERIFIED_STATEMENTS.md).

## Limitations

This is a research and reproducibility repository, not a clinical device. The studies
use retrospective public cohorts, simulated failure modes, and limited external
multimodal validation. The clinical anchor can be misrecorded, the external cohorts
are not fully independent, and performance estimates should not be interpreted as
evidence of clinical utility or deployment readiness.

## License and citation

No repository-wide open-source license or citation record is currently declared.
Before redistributing the code, data derivatives, or manuscript assets, verify the
applicable terms for the included datasets, Springer template files, and third-party
software.
