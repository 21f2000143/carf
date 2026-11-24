# E10: CARF on CMU-MOSEI (E3 protocol)

Data: MMSA `unaligned_50.pkl` (HF mirror tamb2203579/CMU-MOSEI); splits 16326/1871/4659. Label: sentiment >= 0 vs < 0 (Acc-2, zeros counted positive). Test positive rate 0.710.
Columns: audio -> col 0 (rna slot), text -> col 1 (anchor), visual -> col 2 (mri slot).

## Experts

| modality | model | C/alpha | val AUROC | OOF AUROC (train+valid) | test AUROC |
|---|---|---|---|---|---|
| audio | lr | 0.1 | 0.638 | 0.675 | 0.644 |
| text | lr | 0.000316 | 0.877 | 0.882 | 0.884 |
| visual | lr | 0.001 | 0.614 | 0.640 | 0.636 |

## Clean test

| model | AUROC | Acc@0.5 | Brier | ECE | CARF-model dAUROC [95% CI] |
|---|---|---|---|---|---|
| text | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| audio | 0.644 | 0.715 | 0.195 | 0.024 | +0.240 [+0.221, +0.259] |
| visual | 0.636 | 0.713 | 0.196 | 0.017 | +0.248 [+0.229, +0.267] |
| mean | 0.864 | 0.771 | 0.153 | 0.135 | +0.020 [+0.013, +0.027] |
| stack | 0.887 | 0.816 | 0.134 | 0.115 | -0.002 [-0.004, -0.001] |
| mask_stack | 0.886 | 0.817 | 0.134 | 0.114 | -0.002 [-0.004, -0.001] |
| offset_lr | 0.886 | 0.826 | 0.119 | 0.021 | -0.002 [-0.003, -0.000] |
| offset_lr_masked | 0.885 | 0.826 | 0.120 | 0.025 | -0.001 [-0.002, -0.001] |
| mlp_gate | 0.886 | 0.828 | 0.119 | 0.032 | -0.002 [-0.004, -0.001] |
| moddrop_mlp | 0.886 | 0.826 | 0.119 | 0.019 | -0.002 [-0.004, -0.001] |
| carf | 0.884 | 0.824 | 0.120 | 0.024 |  |
| carf_B0.5 | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| carf_B2 | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| carf_no_robust | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| carf_no_context | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| carf_unbounded | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |
| no_visual_stack | 0.884 | 0.815 | 0.135 | 0.115 | -0.000 [-0.002, +0.001] |
| carf_no_visual | 0.884 | 0.824 | 0.120 | 0.024 | +0.000 [+0.000, +0.000] |

## Stress (test AUROC, mean over repeats for random scenarios)

| scenario | text | mean | stack | mask_stack | offset_lr_masked | moddrop_mlp | carf | carf_B0.5 | carf_B2 | carf_no_robust |
|---|---|---|---|---|---|---|---|---|---|---|
| audio_high | 0.884 | 0.882 | 0.887 | 0.887 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_input_dropout50 | 0.884 | 0.849 | 0.886 | 0.885 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_input_noise_sd0.5 | 0.884 | 0.840 | 0.885 | 0.885 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_input_noise_sd1 | 0.884 | 0.804 | 0.884 | 0.883 | 0.883 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_low | 0.884 | 0.882 | 0.887 | 0.887 | 0.885 | 0.887 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_missing | 0.884 | 0.882 | 0.887 | 0.887 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_neutral | 0.884 | 0.882 | 0.887 | 0.887 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_noisy | 0.884 | 0.820 | 0.884 | 0.884 | 0.881 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| audio_permuted | 0.884 | 0.855 | 0.886 | 0.886 | 0.884 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| both_missing | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 |
| intact | 0.884 | 0.864 | 0.887 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_high | 0.884 | 0.868 | 0.885 | 0.885 | 0.885 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_input_dropout50 | 0.884 | 0.866 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_input_noise_sd0.5 | 0.884 | 0.862 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_input_noise_sd1 | 0.884 | 0.857 | 0.886 | 0.885 | 0.885 | 0.886 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_low | 0.884 | 0.868 | 0.885 | 0.885 | 0.885 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_missing | 0.884 | 0.868 | 0.885 | 0.885 | 0.885 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_neutral | 0.884 | 0.868 | 0.885 | 0.885 | 0.885 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_noisy | 0.884 | 0.816 | 0.879 | 0.878 | 0.882 | 0.880 | 0.884 | 0.884 | 0.884 | 0.884 |
| visual_permuted | 0.884 | 0.858 | 0.883 | 0.883 | 0.885 | 0.884 | 0.884 | 0.884 | 0.884 | 0.884 |

## Worst-case AUROC over scenarios and max |fused - anchor| logit

| model | clean AUROC | worst AUROC | worst scenario | max displacement |
|---|---|---|---|---|
| text | 0.884 | 0.884 | audio_high | 0.00 |
| audio | 0.644 | 0.500 | audio_high | 14.67 |
| visual | 0.636 | 0.499 | visual_permuted | 12.16 |
| mean | 0.864 | 0.804 | audio_input_noise_sd1 | 5.69 |
| stack | 0.887 | 0.879 | visual_noisy | 4.75 |
| mask_stack | 0.886 | 0.878 | visual_noisy | 4.80 |
| offset_lr | 0.886 | 0.868 | visual_noisy | 2.88 |
| offset_lr_masked | 0.885 | 0.881 | audio_noisy | 1.97 |
| mlp_gate | 0.886 | 0.882 | visual_noisy | 3.44 |
| moddrop_mlp | 0.886 | 0.880 | visual_noisy | 2.77 |
| carf | 0.884 | 0.884 | audio_high | 0.00 |
| carf_B0.5 | 0.884 | 0.884 | audio_high | 0.00 |
| carf_B2 | 0.884 | 0.884 | audio_high | 0.00 |
| carf_no_robust | 0.884 | 0.884 | audio_high | 0.00 |
| carf_no_context | 0.884 | 0.884 | audio_high | 0.00 |
| carf_unbounded | 0.884 | 0.884 | audio_high | 0.00 |

## Natural missingness (true masks)

Test: audio missing 0, visual missing 60, both 0 of 4659. Train: audio 0, visual 119; valid: audio 0, visual 4.
Regimes: zero_features = main fit, test features as delivered; mask_frozen = main fusions, test probability of a naturally missing modality set to NaN; mask_refit = fusions refit on OOF probabilities with true train+valid masks, true test masks.

| regime | group | n | text | mean | stack | mask_stack | offset_lr_masked | moddrop_mlp | carf | carf_B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| zero_features | all | 4659 | 0.884 | 0.864 | 0.887 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |
| zero_features | any_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.861 | 0.861 | 0.861 |
| zero_features | visual_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.861 | 0.861 | 0.861 |
| zero_features | complete | 4599 | 0.884 | 0.864 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |
| mask_frozen | all | 4659 | 0.884 | 0.864 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |
| mask_frozen | any_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.863 | 0.861 | 0.861 |
| mask_frozen | visual_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.863 | 0.861 | 0.861 |
| mask_frozen | complete | 4599 | 0.884 | 0.864 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |
| mask_refit | all | 4659 | 0.884 | 0.864 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |
| mask_refit | any_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.858 | 0.861 | 0.861 |
| mask_refit | visual_missing | 60 | 0.861 | 0.798 | 0.856 | 0.856 | 0.865 | 0.858 | 0.861 | 0.861 |
| mask_refit | complete | 4599 | 0.884 | 0.864 | 0.886 | 0.886 | 0.885 | 0.886 | 0.884 | 0.884 |

any_missing subgroup, zero_features, CARF minus model dAUROC [95% CI]: carf_B2 +0.000 [+0.000, +0.000], mask_stack +0.005 [-0.026, +0.040], mean +0.062 [-0.045, +0.200], moddrop_mlp +0.000 [-0.028, +0.025], offset_lr_masked -0.005 [-0.032, +0.012], stack +0.005 [-0.026, +0.040], text +0.000 [+0.000, +0.000]

any_missing subgroup, mask_refit, CARF minus model dAUROC [95% CI]: carf_B2 +0.000 [+0.000, +0.000], mask_stack +0.005 [-0.026, +0.040], mean +0.062 [-0.045, +0.200], moddrop_mlp +0.002 [-0.027, +0.032], offset_lr_masked -0.005 [-0.032, +0.012], stack +0.005 [-0.026, +0.040], text +0.000 [+0.000, +0.000]

## Notes

- Time-shuffling before pooling skipped (mean/std pooling is permutation-invariant).
- CARF gate (B=1), main: coef [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], budgets [audio, visual] [0.0, 0.0]; B=0.5 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; B=2 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
- CARF gate (B=1), mask_refit: coef [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
- Gate diagnostics (OOF, main): {"anchor_platt_slope_oof": 1.206013820145858, "corr_residual_vs_minus_anchor": {"audio": 0.898395853250284, "visual": 0.9691366286048296}, "sd_logit": {"audio": 0.7431457612843532, "text": 1.6839695922815938, "visual": 0.4153033175321515}, "grad_at_zero_no_robust_no_l2_no_context_full_mask": [0.012593013868275302, 0.01635354746627057], "coef_no_robust_no_l2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "coef_unbounded_no_context_no_robust_no_l2": [0.0, 0.0]}. A positive gradient at theta=0 with theta>=0 means theta=0 is the constrained optimum (KKT).
- offset_lr weights [intercept, audio, visual]: [-0.395, 0.185, 0.259].
