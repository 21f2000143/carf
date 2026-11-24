# Revision 3: bounded clinical-anchored residual fusion

Written before running the new architecture (2026-09-23). Earlier I-SPY1 results
have already been inspected; this is an exploratory methodological revision,
not a fresh confirmatory external test. No model or hyperparameter will be
selected using the new I-SPY1 outcomes.

## Hypothesis and contribution
CARF (clinical-anchored residual fusion) adds optional molecular/MRI log-odds
residuals to a clinical anchor. Residuals are clipped to B=1 and multiplied by
patient-local agreement/confidence basis functions in [0,1]. Nonnegative
coefficients have total mass at most one. This gives exact clinical fallback
when both optional modalities are absent, bounded logit displacement, and a
per-modality perturbation bound without claiming guaranteed AUROC improvement.
Coefficients minimize mean loss over all optional-modality masks plus a smooth
worst-corruption loss and L2 regularization. This is a convex program in six
coefficients. Confidence is a heuristic feature, not a verified reliability score.

## Fixed protocol
Use the existing participant-linked MRI, expression, and clinical cohorts.
Move variance-based 3,000-gene filtering into every expert fit; ANOVA then selects
250 genes. Use the existing fixed expert hyperparameters and 5 inner folds for
OOF gate training, 5 outer folds x 5 repeats for internal evaluation. No tuning
against outer or external labels. All comparisons share expert predictions.

Fixed CARF settings: B=1, robust weight=0.25, log-sum-exp temperature=0.05,
L2=0.01, clinical anchor index=1. Train over all four optional masks. Corruption
environments replace one optional expert at a time by 0.01 or 0.99, without
using labels to construct perturbations. Ablate robust training, context basis,
MRI, and the residual bound. Compare clinical/RNA/MRI alone, mean, standard
stacking, and mask-augmented logistic stacking. Missing data are explicit
masks; 0.5 is neutralization, not actual absence.

Report clean AUROC/AUPRC/Brier/log loss, paired bootstrap intervals (2,000
patients resamples), all outer fold results and out-of-fold predictions,
missing-modality and corruption stress for each optional branch. Repeat random
noise/permutation stress 30 times with fixed seeds. Report both discrimination
and displacement from the clinical anchor; a smaller displacement alone is not
an accuracy benefit. Report learned budgets and optimization diagnostics.

## Corrections to prior analyses
Prior internal gene filtering used all discovery rows before splitting. Prior
RPPA evaluation constructed held-out predictions by fitting within the test
partition. Prior gate-only label permutations are not whole-pipeline null tests.
Do not reuse those internal/RPPA/permutation findings as valid new evidence.
Correct RPPA held-out expert prediction, rerun its five matched folds, and label
it secondary. Keep historical artifacts separate. Retain negative findings;
state missing independent confirmation and lack of neural/SOTA comparison.

## Data-integrity amendment, before the corrected run
The initial revision-3 run exposed constant external RNA probabilities. Inspection
showed partially missing gene measurements and scipy rankdata's default NaN
propagation: one missing gene erased every observed value in that patient row.
Correct patient-local ranks to rank finite genes only, divide by the number of
observed genes, and retain missing entries for training-fitted imputation. This
changes no fusion hyperparameters. Rerun all comparisons under corrected source
hashes. Old revision-2 and preliminary revision-3 RNA-dependent results are
superseded, not biological evidence of assay-transfer failure.
