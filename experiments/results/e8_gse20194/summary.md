# E8: frozen external evaluation on GSE20194 (MAQC-II BR, MDACC)

Script: `experiments/e8_gse20194.py` (deterministic; a rerun reproduced identical metrics). Data: `experiments/data_e8/`.

## Cohort
- GSE20194 has 278 arrays but only **230 unique patients**. The cohort is MAQC_T (130) + MAQC_V (100), one array each. Excluded without labels: 29 technical replicates (R1/MDA_R) and 19 MAQC_Q "Not used" arrays.
- n = 230, pCR = 48 (20.9%). No patients were unlabeled.
- Clinical mapping:
  - age = `age` (0 missing).
  - hr = ER P or PR P (151 positive).
  - her2 = `her2 status` P (40 positive).
  - pcr = `pcr_vs_rd` == pCR.
- Expression: log2 MAS5 values. Probes map to symbols via GPL96, keeping single-symbol probes only (19,820) and averaging per symbol (12,548 symbols). Values are rank-normalized per patient over G.
- **G = 12,148 genes** (I-SPY2 19,134 ∩ GSE20194).

## Overlap
- **GSE25066: 187/230 patients (81%) are shared.**
  - 184 are ID-matched identical arrays (Spearman rho ≈ 1.0000; pCR agreement 100%).
  - Two more identical arrays (PERU14-16, PERU12-14) have different title formatting.
  - M367 has the same ID but a different array (rho 0.77).
  - Only 43 patients (44 counting M367) are unique to GSE20194.
  - The two are **not independent validation cohorts**.
- GSE25066 also contains 83 ISPY (I-SPY1) Affymetrix samples, which matters for the GSE25066 analysis.
- I-SPY2: no overlap is possible (different programme and era, no shared identifiers).

## Main table (MRI absent; Δ = CARF − model, paired bootstrap with 2000 resamples)
| model | AUROC (95% CI) | Brier | ECE | max abs(z−a) | Δ (95% CI) |
|---|---|---|---|---|---|
| clinical | 0.821 (0.748–0.886) | 0.177 | 0.222 | 0 | −0.005 (−0.013, 0.000) |
| rna | 0.716 (0.632–0.797) | 0.202 | 0.207 | 14.3 | +0.100 (0.018, 0.180) |
| stack | 0.816 | 0.147 | 0.111 | 1.89 | −0.000 (−0.018, 0.018) |
| mask_stack | 0.821 | 0.140 | 0.110 | 1.52 | −0.005 (−0.021, 0.009) |
| offset_lr | 0.795 | 0.160 | 0.139 | 5.69 | +0.021 (−0.025, 0.066) |
| offset_lr_masked | 0.795 | 0.161 | 0.142 | 5.75 | +0.021 (−0.025, 0.066) |
| robust_offset_lr | 0.811 | 0.133 | 0.062 | 2.56 | +0.004 (−0.033, 0.039) |
| moddrop_mlp | 0.819 | 0.154 | 0.115 | 2.43 | −0.004 (−0.020, 0.012) |
| **carf** (B=1, γ=0.25) | **0.816 (0.743–0.878)** | 0.151 | 0.152 | **0.47** | — |
| carf_platt | 0.816 | 0.129 | 0.058 | 1.66 | 0 |

## Stress (30 realizations; Δ = CARF − model, paired bootstrap, AUROC averaged over realizations)
| scenario | CARF | clinical | stack | mask_stack | offset_lr | robust_offset | moddrop |
|---|---|---|---|---|---|---|---|
| RNA output permuted | 0.819 | 0.821 | 0.810; +0.009 (0.001, 0.017) | 0.814; +0.005 (−0.001, 0.011) | 0.718; +0.101 (0.064, 0.136) | 0.794; +0.025 (−0.001, 0.050) | 0.813; +0.006 (−0.002, 0.012) |
| RNA 50% input dropout | 0.812 | 0.821 | 0.764; +0.047 (0.023, 0.070) | 0.783; +0.028 (0.010, 0.045) | 0.764; +0.048 (0.018, 0.077) | 0.805; +0.007 (−0.012, 0.025) | 0.776; +0.035 (0.014, 0.055) |

- CARF − clinical is −0.002 (−0.004, 0.000) under permutation and −0.009 (−0.021, 0.002) under dropout.
- CARF max abs(z−a) stays ≤ 0.48 throughout. The other fusions range from 1.5 to 5.9.

## Interpretation
- The clinical anchor dominates. The cross-platform RNA expert (Agilent → Affymetrix) is weak (0.72).
- No fusion beats clinical. Intact CARF is indistinguishable from stack, mask_stack and ModDrop, and slightly below clinical.
- CARF's value here is robustness: under RNA corruption it stays within 0.01 of clinical, with bounded displacement. Stack, mask_stack, ModDrop and offset LR lose 0.03–0.10 AUROC.
- Robust offset LR is the closest competitor.
- Descriptive only: on the 44 patients not in GSE25066 (17 pCR), AUROC is CARF 0.752, clinical 0.767, stack 0.758.
