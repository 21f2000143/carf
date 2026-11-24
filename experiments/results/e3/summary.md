# E3: CARF on CMU-MOSI (second domain)

Data: MMSA `unaligned_50.pkl` (HF mirror tamb2203579/CMU-MOSI); splits 1284/229/686. Label: sentiment >= 0 vs < 0 (Acc-2, zeros counted positive). Test positive rate 0.448.
Columns: audio -> col 0 (rna slot), text -> col 1 (anchor), visual -> col 2 (mri slot).

## Experts

| modality | model | C/alpha | val AUROC | OOF AUROC (train+valid) | test AUROC |
|---|---|---|---|---|---|
| audio | lr | 0.0316 | 0.620 | 0.600 | 0.635 |
| text | lr | 0.00316 | 0.862 | 0.871 | 0.846 |
| visual | lr | 0.000316 | 0.569 | 0.617 | 0.569 |

## Clean test

| model | AUROC | Acc@0.5 | Brier | ECE | CARF-model dAUROC [95% CI] |
|---|---|---|---|---|---|
| text | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| audio | 0.635 | 0.510 | 0.255 | 0.147 | +0.211 [+0.162, +0.260] |
| visual | 0.569 | 0.469 | 0.260 | 0.128 | +0.277 [+0.225, +0.328] |
| mean | 0.852 | 0.726 | 0.204 | 0.195 | -0.006 [-0.016, +0.003] |
| stack | 0.850 | 0.783 | 0.157 | 0.050 | -0.004 [-0.008, -0.001] |
| mask_stack | 0.850 | 0.787 | 0.157 | 0.047 | -0.004 [-0.007, -0.001] |
| offset_lr | 0.853 | 0.768 | 0.159 | 0.068 | -0.007 [-0.012, -0.002] |
| offset_lr_masked | 0.851 | 0.765 | 0.162 | 0.083 | -0.005 [-0.009, -0.002] |
| mlp_gate | 0.852 | 0.776 | 0.161 | 0.065 | -0.006 [-0.011, -0.001] |
| moddrop_mlp | 0.854 | 0.776 | 0.158 | 0.059 | -0.008 [-0.014, -0.001] |
| carf | 0.846 | 0.774 | 0.162 | 0.066 |  |
| carf_B0.5 | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| carf_B2 | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| carf_no_robust | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| carf_no_context | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| carf_unbounded | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |
| no_visual_stack | 0.850 | 0.783 | 0.157 | 0.053 | -0.004 [-0.008, -0.001] |
| carf_no_visual | 0.846 | 0.774 | 0.162 | 0.066 | +0.000 [+0.000, +0.000] |

## Stress (test AUROC, mean over repeats for random scenarios)

| scenario | text | mean | stack | mask_stack | offset_lr_masked | moddrop_mlp | carf | carf_B0.5 | carf_B2 | carf_no_robust |
|---|---|---|---|---|---|---|---|---|---|---|
| audio_high | 0.846 | 0.849 | 0.846 | 0.846 | 0.847 | 0.848 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_input_dropout50 | 0.846 | 0.851 | 0.848 | 0.848 | 0.849 | 0.851 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_input_noise_sd0.5 | 0.846 | 0.851 | 0.850 | 0.849 | 0.851 | 0.853 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_input_noise_sd1 | 0.846 | 0.845 | 0.849 | 0.849 | 0.850 | 0.850 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_low | 0.846 | 0.849 | 0.846 | 0.846 | 0.847 | 0.848 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_missing | 0.846 | 0.849 | 0.846 | 0.846 | 0.847 | 0.848 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_neutral | 0.846 | 0.849 | 0.846 | 0.846 | 0.847 | 0.849 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_noisy | 0.846 | 0.794 | 0.841 | 0.844 | 0.826 | 0.825 | 0.846 | 0.846 | 0.846 | 0.846 |
| audio_permuted | 0.846 | 0.838 | 0.845 | 0.845 | 0.846 | 0.844 | 0.846 | 0.846 | 0.846 | 0.846 |
| both_missing | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 |
| intact | 0.846 | 0.852 | 0.850 | 0.850 | 0.851 | 0.854 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_high | 0.846 | 0.850 | 0.850 | 0.849 | 0.850 | 0.851 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_input_dropout50 | 0.846 | 0.852 | 0.850 | 0.850 | 0.851 | 0.853 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_input_noise_sd0.5 | 0.846 | 0.852 | 0.850 | 0.850 | 0.851 | 0.854 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_input_noise_sd1 | 0.846 | 0.852 | 0.850 | 0.850 | 0.851 | 0.853 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_low | 0.846 | 0.850 | 0.850 | 0.849 | 0.850 | 0.851 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_missing | 0.846 | 0.850 | 0.850 | 0.849 | 0.850 | 0.852 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_neutral | 0.846 | 0.850 | 0.850 | 0.849 | 0.850 | 0.852 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_noisy | 0.846 | 0.796 | 0.850 | 0.850 | 0.842 | 0.840 | 0.846 | 0.846 | 0.846 | 0.846 |
| visual_permuted | 0.846 | 0.848 | 0.850 | 0.849 | 0.850 | 0.851 | 0.846 | 0.846 | 0.846 | 0.846 |

## Worst-case AUROC over scenarios and max |fused - anchor| logit

| model | clean AUROC | worst AUROC | worst scenario | max displacement |
|---|---|---|---|---|
| text | 0.846 | 0.846 | audio_high | 0.00 |
| audio | 0.635 | 0.500 | audio_high | 13.54 |
| visual | 0.569 | 0.496 | visual_permuted | 12.47 |
| mean | 0.852 | 0.794 | audio_noisy | 6.50 |
| stack | 0.850 | 0.841 | audio_noisy | 5.22 |
| mask_stack | 0.850 | 0.844 | audio_noisy | 5.01 |
| offset_lr | 0.853 | 0.810 | audio_noisy | 4.38 |
| offset_lr_masked | 0.851 | 0.826 | audio_noisy | 2.90 |
| mlp_gate | 0.852 | 0.830 | audio_noisy | 5.29 |
| moddrop_mlp | 0.854 | 0.825 | audio_noisy | 5.69 |
| carf | 0.846 | 0.846 | audio_high | 0.00 |
| carf_B0.5 | 0.846 | 0.846 | audio_high | 0.00 |
| carf_B2 | 0.846 | 0.846 | visual_noisy | 0.00 |
| carf_no_robust | 0.846 | 0.846 | audio_high | 0.00 |
| carf_no_context | 0.846 | 0.846 | audio_high | 0.00 |
| carf_unbounded | 0.846 | 0.846 | audio_high | 0.00 |

## Notes

- Natural missingness (all-zero frames within valid length): {'test_counts': {'audio': 0, 'visual': 1}, 'trainvalid_counts': {'audio': 0, 'visual': 1}}. Too few for a subgroup analysis; these segments were kept with their (zero) pooled features.
- Time-shuffling before pooling was skipped: mean/std pooling is permutation-invariant, so it is an exact no-op on these features.
- CARF gate (B=1): coef [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], per-modality budgets [audio, visual] [0.0, 0.0].
- Gate diagnostics (OOF): {"anchor_platt_slope_oof": 1.0879185024446476, "corr_residual_vs_minus_anchor": {"audio": 0.9760566474912428, "visual": 0.9973684723174918}, "sd_logit": {"audio": 0.41484183455493867, "text": 1.8964028577272227, "visual": 0.13797431776550548}, "grad_at_zero_no_robust_no_l2_no_context_full_mask": [0.005717080642562362, 0.01417199021187473], "coef_no_robust_no_l2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "coef_unbounded_no_context_no_robust_no_l2": [0.006087261576512534, 0.0]}. A positive gradient at theta=0 with theta>=0 means theta=0 is the constrained optimum (KKT). The residual logit(p_j)-a is almost collinear with -a, so any positive weight mainly shrinks the (slightly underconfident, Platt slope>1) text anchor.
- offset_lr weights [intercept, audio, visual]: [-0.319, 0.575, 0.493].
