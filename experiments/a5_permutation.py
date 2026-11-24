#!/usr/bin/env python3
"""A5: whole-pipeline label-permutation test.

Null: the pipeline learns nothing from I-SPY2 labels. For each permutation the
I-SPY2 pCR labels are shuffled and every stage (OOF experts, final experts,
fusions, offset LR) is refitted; external I-SPY1 AUROC uses the true labels.

Usage: python a5_permutation.py [n_perm] [n_jobs]   (n_perm=0 -> time one run)
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import offset_models, MIKE  # noqa: E402  (adds mike2026 to sys.path)
from scripts.run_revision2_analysis import common_data, experts, predict_expert, oof  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions, SEED  # noqa: E402

MODELS = ['clinical', 'rna', 'mri', 'stack', 'mask_stack', 'carf', 'carf_no_robust', 'offset_lr']
OUT = HERE / 'results' / 'a5'


def one(b, data):
    E, M, I, Ee, Me, Ie, y, ye = data
    t0 = time.time()
    yp = np.random.default_rng(SEED + 5000 + b).permutation(y)
    P = oof(E, M, I, yp, SEED + 100)
    fitted = experts(E, M, I, yp, SEED + 100)
    Q = np.column_stack([predict_expert(m, x, k) for m, x, k in
                         zip(fitted, [Ee, Me, Ie], ['rna', 'clinical', 'mri'])])
    pred = predict_fusions(fit_fusions(P, yp), Q)
    for name, m in offset_models(P, yp).items():
        pred[name] = m.predict_proba(Q)[:, 1]
    return {'perm': b, **{k: roc_auc_score(ye, pred[k]) for k in MODELS}, 'seconds': time.time() - t0}


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    OUT.mkdir(parents=True, exist_ok=True)
    E, M, I, Ee, Me, Ie = common_data()
    data = (E, M, I, Ee, Me, Ie, M.pcr.to_numpy(dtype=int), Me.pcr.to_numpy(dtype=int))
    if n_perm == 0:
        print(one(0, data), flush=True)
        return
    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, verbose=10)(delayed(one)(b, data) for b in range(n_perm))
    null = pd.DataFrame(rows).sort_values('perm')
    null.to_csv(OUT / 'null.csv', index=False)
    obs = pd.read_csv(HERE / 'results' / 'external.csv').set_index('model').auroc
    pv = pd.DataFrame([{'model': k, 'observed': obs[k], 'null_mean': null[k].mean(),
                        'null_sd': null[k].std(ddof=1), 'null_max': null[k].max(),
                        'n_ge': int((null[k] >= obs[k]).sum()), 'n_perm': len(null),
                        'p_value': (1 + (null[k] >= obs[k]).sum()) / (1 + len(null))}
                       for k in MODELS])
    pv.to_csv(OUT / 'pvalues.csv', index=False)
    print(pv.to_string(index=False))
    print(f'wall time {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
