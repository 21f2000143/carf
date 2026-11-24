# Contribution and evidence map

## What changed

CARF (clinical-anchored residual fusion) replaces an unconstrained probability
remapping with an explicit clinical logit plus optional molecular/MRI corrections.
Six nonnegative coefficients allocate a total correction budget. Patient-local
agreement and confidence modify the corrections without turning gate fitting
into a nonconvex optimization problem. Missing optional experts contribute zero.

The gate learns from inner out-of-fold probabilities with an average loss over
all four availability masks, a smooth worst-corruption loss, and regularization.
All choices are fixed in `ANALYSIS_PLAN.md`; outer or external outcomes do not
select numerical hyperparameters. Clinical anchoring is a disclosed operational
assumption developed after earlier external findings, not a blinded discovery.

## What is established mathematically

For learned modality budgets `b_R` and `b_M`:

- Both optional inputs missing: exact fallback to the clipped clinical probability.
- Any optional values: total logit displacement is at most `b_R + b_M <= 1`.
- Replace one optional expert arbitrarily: logit change is at most twice its budget.
- Clinical logit differences greater than twice the total budget cannot have their
  order reversed by optional inputs.
- With fixed OOF features, the constrained gate objective is strictly convex.

These statements certify influence and stability, not correctness, discrimination,
calibration, or resilience to a corrupted clinical anchor.

## Empirical evidence

`RESULTS.md` is generated from the completed corrected run. It retains 12 models,
including fixed expert baselines, ordinary and mask-aware stacking, and four CARF
ablations. There are 25 outer folds, paired external
bootstrap intervals, and 30 realizations of each random optional-modality stress.

`python scripts/validate_revision3.py` verifies hashes, patient-fold separation,
metrics against saved predictions, the CARF stress outputs and bound, fallback,
bootstrap artifacts, and the corrected RPPA subset counts. Regression tests
exercise the architecture, missing expression values, and RPPA fit boundaries.

## Corrections that must remain disclosed

1. Partial expression missingness formerly erased a complete patient's rank row,
   causing constant external RNA predictions. Rank observed genes only and leave
   missing entries for training-fitted imputation.
2. Gene variance filtering formerly saw held-out discovery patients. Fit the
   filter within each expert's training partition.
3. RPPA test probabilities formerly came from fitting within the test partition.
   Predict them using training-partition experts.
4. Gate-only label permutations are not a complete-pipeline association test.
   Those earlier results do not support the new paper's significance claims.

## Claims that remain unsupported

Missing-modality training, mixture-of-experts, and reliability-aware fusion are
established research areas; the paper cites ModDrop, QA-MoE, and group-DRO work.
UCMR-Net also uses anchored residual fusion, and recent reliability diagnostics
question whether context scores change decisions usefully. The paper cites these
close precedents. The proposed combination is not a proven first-in-literature result. No comparison
against a tuned neural state of the art has been run. The external cohort was
seen in earlier revisions, so independent confirmation is still needed. An
architectural contribution does not justify claiming clinical utility or
statistically established improvement when the measured evidence does not.
