# E9: CARF on METABRIC — pre-specified task definition

Written 2026-09-25 02:29 IST, after inspecting only file headers / field value counts,
before fitting any model or looking at any performance number. Nothing below is changed after results.

## Data
cBioPortal datahub `brca_metabric` (S3 tarball returned HTTP 403; files taken from the GitHub datahub LFS
mirror, `https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/<file>`).
Files: `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mrna_illumina_microarray.txt`.
SHA-256 in `experiments/data_e9/SHA256SUMS` and `provenance.json`.

## Outcome
Breast-cancer death within 5 years (60 months), from `VITAL_STATUS` + `OS_MONTHS`
(the datahub has no explicit DSS_STATUS/DSS_MONTHS columns; `VITAL_STATUS` distinguishes
"Died of Disease" from "Died of Other Causes", which gives disease-specific survival).
- y = 1: `VITAL_STATUS == "Died of Disease"` and `OS_MONTHS <= 60`.
- y = 0: `OS_MONTHS > 60` (alive, or died of any cause after 60 months).
- Excluded: censored before 60 months under DSS, i.e. `Living` with `OS_MONTHS <= 60`,
  or `Died of Other Causes` with `OS_MONTHS <= 60` (competing death = DSS censoring);
  and patients with missing `VITAL_STATUS` / `OS_MONTHS`.

## Modalities (fixed column order [rna, clinical, mri], anchor = index 1; mri column is NaN for everyone)
- Anchor clinical expert: age at diagnosis (`AGE_AT_DIAGNOSIS`), tumour size (`TUMOR_SIZE`),
  grade (`GRADE`), positive lymph nodes (`LYMPH_NODES_EXAMINED_POSITIVE`), ER status (`ER_STATUS`, sample file,
  Positive=1), HER2 status (`HER2_STATUS`, sample file, Positive=1). Pipeline mirrors `fit_expert(kind='clinical')`:
  median impute -> StandardScaler -> LR(C=1, class-balanced, liblinear).
- Optional expression expert: Illumina HT-12 microarray, per-patient `rank_normalize`, then `fit_expert(kind='rna')`
  (median impute -> top-3000 variance -> SelectKBest 250 -> scaler -> LR C=0.1, class-balanced), imported unchanged.

## Split
`COHORT` field present (values 1-9; survival only for cohorts 1-5). Train = cohorts 1-2, test = cohorts 3-5
(the commonly used convention; this is not identical to the original 997/995 discovery/validation split).
Primary analysis: patients with outcome and expression (complete cases, as in the I-SPY pipeline).
Experts and every fusion model fit on train only; gates fit on inner 5-fold OOF expert probabilities
(StratifiedKFold, seed 20260923+100, as `cached_predictions` does for the external fit). Test labels are used only for evaluation.

## Models (all settings fixed, nothing tuned on test)
CARF (`AnchoredResidualFusion(anchor=1, budget=1, robust_weight=0.25)`) and ablations from `fit_fusions`;
experts; mean; stack; mask_stack; offset_lr / offset_lr_masked (tier1); RobustOffsetLR (e5, gamma=0.25);
ModDrop MLP (a7, grid chosen by inner CV on train OOF only).
stack cannot take NaN; it is fit on the train OOF matrix with the NaN mri column filled with 0.5,
exactly as `predict_fusions` fills it at prediction time. mask_stack and CARF are fit on the NaN matrix.

## Evaluation on test
AUROC, Brier, ECE (10 bins), max |logit(p) - logit(anchor)|; paired bootstrap (2000, seed 20260923) of CARF - model.
Stress with frozen models: RNA output permuted (30 seeds), RNA output + N(0, 0.25) clipped (30 seeds),
RNA missing, RNA 50% input gene dropout per patient before rank normalisation (30 seeds);
paired bootstrap intervals for permuted (AUROC averaged over the 30 realisations per resample).
Natural missingness: test patients (and, if any, train-cohort patients are not used) with clinical data and outcome but no expression,
evaluated with the true mask (RNA = NaN) as a subgroup.

---
# Results (appended after the run; the definition above was not changed)

Run 2026-09-25 ~02:32 IST, `experiments/e9_metabric.py` (single run; ~4 min wall).

## Cohorts
Cohorts 1-5: 2268 patients; 442 excluded (DSS-censored before 60 mo incl. 90 other-cause deaths, or unknown status);
1 train-cohort patient with outcome but no expression excluded.
- Train (cohorts 1-2): n = 734, events = 132 (18.0%).
- Test (cohorts 3-5): n = 1091, events = 205 (18.8%).
- Natural missingness: 0 test patients have clinical + outcome but no expression -> subgroup not evaluable.

## Experts
Train inner-OOF AUROC: clinical 0.783, RNA 0.705. Test AUROC: clinical 0.739, RNA 0.660.
The optional modality is weaker than the anchor here (on I-SPY external it was on par: 0.809 vs 0.802).

## NaN handling
Unmodified `fit_fusions` fails on the all-NaN mri column (sklearn LR in `stack`). stack and no_mri_stack
were fit with the mri column filled with 0.5 (the `predict_fusions` fill); mask_stack and CARF fit on NaN.
CARF == carf_no_mri exactly (max diff 0). All predictions finite.
CARF gate: coef = [0.200, 0.217, 0, 0, 0, 0]; RNA budget 0.417 logit.

## Test metrics (n = 1091)
| model | AUROC | Brier | ECE | max abs(z-a) |
|---|---|---|---|---|
| clinical | 0.739 | 0.197 | 0.232 | 0 |
| rna | 0.660 | 0.222 | 0.199 | 8.51 |
| mean | 0.735 | 0.183 | 0.193 | 4.67 |
| stack | 0.753 | 0.195 | 0.240 | 3.77 |
| mask_stack | 0.753 | 0.193 | 0.235 | 3.58 |
| offset_lr | 0.752 | 0.135 | 0.037 | 2.65 |
| robust_offset_lr | 0.748 | 0.135 | 0.038 | 2.11 |
| moddrop_mlp | 0.753 | 0.133 | 0.035 | 5.01 |
| carf_no_robust | 0.751 | 0.181 | 0.195 | 0.93 |
| **carf** | **0.749** | 0.189 | 0.221 | **0.28** |

Paired bootstrap, CARF - model, AUROC (95% CI): clinical +0.010 (+0.002, +0.019); stack -0.004 (-0.015, +0.006);
mask_stack -0.004 (-0.012, +0.003); offset_lr -0.003 (-0.009, +0.004);
robust_offset_lr +0.001 (-0.004, +0.006); moddrop -0.004 (-0.013, +0.004).
Brier: CARF is worse than every free-intercept model (offset/robust offset/ModDrop by ~0.055) because the class-balanced
anchor is miscalibrated at an 19% event rate and CARF has no intercept; better than clinical by 0.007.

## Stress (frozen models; mean AUROC over 30 realisations)
| scenario | clinical | stack | mask_stack | offset_lr | robust_offset | moddrop | carf_no_robust | carf |
|---|---|---|---|---|---|---|---|---|
| intact | 0.739 | 0.753 | 0.753 | 0.751 | 0.748 | 0.753 | 0.751 | 0.749 |
| rna_permuted | 0.739 | 0.718 | 0.726 | 0.729 | 0.736 | 0.723 | 0.708 | 0.735 |
| rna_noisy (SD .25) | 0.739 | 0.747 | 0.749 | 0.743 | 0.746 | 0.748 | 0.741 | 0.746 |
| rna_missing | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 | 0.739 |
| rna_dropout50 (input) | 0.739 | 0.744 | 0.746 | 0.746 | 0.744 | 0.745 | 0.740 | 0.743 |

Max abs(z - a) under stress: CARF <= 0.28 in every scenario; stack 4.4, mask_stack 4.1, offset_lr 2.7, robust offset 2.2, ModDrop 5.7.
RNA permuted, CARF - model AUROC (95% CI): clinical -0.004 (-0.006, -0.002); stack +0.017 (+0.013, +0.021);
mask_stack +0.009 (+0.006, +0.011); offset_lr +0.006 (+0.004, +0.008);
robust_offset_lr -0.001 (-0.002, -0.000); moddrop +0.012 (+0.009, +0.015); carf_no_robust +0.027 (+0.021, +0.033).
Intervals for noisy / dropout / missing are in bootstrap.csv (all CARF-vs-fusion-baseline CIs include 0).

## Comparison with I-SPY (candid)
Qualitatively the same shape: CARF gives a small, significant gain over the anchor, stays within its displacement bound
(0.28 logit vs 3-6 for learned fusions), and degrades least among the non-offset fusions when RNA is permuted.
Differences: (1) all effects are much smaller because RNA is weaker than the anchor here; (2) intact CARF does not beat
stack/mask_stack/ModDrop (point -0.004, CIs include 0); (3) under permutation the robust offset LR matches or slightly
beats CARF (-0.001); (4) CARF still loses to the anchor under permutation (-0.004), as on I-SPY; (5) calibration is poor
without recalibration. The natural-missingness check could not be run (no eligible test patients).
