"""Table 3 (App. A) + teaser MAE numbers:
leave-one-model-out accuracy prediction from OC-rate, CoT-SC-rate, and
naive-SC-rate. Provides the pooled MAE figures cited in the abstract and
Fig 1 caption.
"""
from collections import defaultdict
import numpy as np

from analysis.scoring import (score_official, is_correct_official,
                              pairwise_official)
from analysis.data_loader import load_nonthinking, NT_MODELS, NT_DATASETS


def per_cell_signals():
    """Per (model, dataset) -> dict with OC-rate, naive-SC-K=10 rate,
    CoT-SC-K=10 rate, accuracy."""
    counts = defaultdict(lambda: {'n': 0, 'oc': 0, 'correct': 0,
                                   'nv_sum': 0.0, 'nv_n': 0,
                                   'cot_sum': 0.0, 'cot_n': 0})
    for r in load_nonthinking():
        key = (r['model'], r['dataset']); ds = r['dataset']
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        op_f1 = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0
        counts[key]['n'] += 1
        if op_f1 >= 0.5: counts[key]['oc'] += 1
        if is_correct_official(direct, r['ground_truth'], ds, r['aliases']):
            counts[key]['correct'] += 1
        if len(nv) >= 10:
            counts[key]['nv_sum'] += pairwise_official(nv[:10], ds)
            counts[key]['nv_n']   += 1
        cot_ne = [s for s in cot if s.strip()]
        if len(cot_ne) >= 10:
            counts[key]['cot_sum'] += pairwise_official(cot_ne[:10], ds)
            counts[key]['cot_n']   += 1
    out = {}
    for key, c in counts.items():
        if not c['n']: continue
        out[key] = {
            'oc':  c['oc'] / c['n'],
            'acc': c['correct'] / c['n'],
            'nv_k10':  c['nv_sum'] / c['nv_n']   if c['nv_n']  else None,
            'cot_k10': c['cot_sum'] / c['cot_n'] if c['cot_n'] else None,
        }
    return out


def loo_one_signal(per_cell, signal_key, label):
    print(f"\n--- {label} ---")
    print(f"  {'dataset':<12} {'n':>3}  {'fit_a':>8} {'fit_b':>8} {'in_R2':>7}  {'LOO MAE':>8} {'LOO max':>8}")
    all_errs = []
    for ds in NT_DATASETS:
        pairs = []
        for s in NT_MODELS:
            cell = per_cell.get((s, ds))
            if not cell or cell.get(signal_key) is None: continue
            pairs.append((s, cell[signal_key], cell['acc']))
        if len(pairs) < 3:
            print(f"  {ds:<12} skip (n<3)")
            continue
        sigs = np.array([p[1] for p in pairs])
        accs = np.array([p[2] for p in pairs])
        n = len(sigs)
        a_full, b_full = np.polyfit(sigs, accs, 1)
        pred_full = a_full * sigs + b_full
        ss_res = ((accs - pred_full) ** 2).sum()
        ss_tot = ((accs - accs.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        loo_errs = []
        for i in range(n):
            sig_train = np.concatenate([sigs[:i], sigs[i+1:]])
            acc_train = np.concatenate([accs[:i], accs[i+1:]])
            a, b = np.polyfit(sig_train, acc_train, 1)
            loo_errs.append(a * sigs[i] + b - accs[i])
        loo_errs = np.array(loo_errs)
        mae = np.abs(loo_errs).mean()
        maxerr = np.abs(loo_errs).max()
        all_errs.extend(loo_errs.tolist())
        print(f"  {ds:<12} {n:>3}  {a_full:>+8.3f} {b_full:>+8.3f} {r2:>7.3f}  "
              f"{mae:>8.3f} {maxerr:>8.3f}")
    if all_errs:
        all_errs = np.array(all_errs)
        print(f"  pooled n={len(all_errs)}: MAE = {np.abs(all_errs).mean():.3f}, "
              f"max = {np.abs(all_errs).max():.3f}")


if __name__ == '__main__':
    per_cell = per_cell_signals()
    print('=' * 80)
    print(' Leave-one-model-out accuracy prediction (Table 3, App. A + teaser)')
    print('=' * 80)
    loo_one_signal(per_cell, 'oc',      'OC (paper headline)')
    loo_one_signal(per_cell, 'cot_k10', 'CoT-SC K=10 (new teaser right panel)')
    loo_one_signal(per_cell, 'nv_k10',  'naive-SC K=10 (old teaser right panel)')
