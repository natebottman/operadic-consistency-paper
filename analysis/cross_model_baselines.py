"""Cross-model Pearson r per baseline (defensive check for §3.1 / Fig 1).

For each (model, dataset) cell we compute:
- per-cell accuracy (mean correctness)
- per-cell baseline rate (mean of the baseline signal, or fraction
  exceeding a threshold for binarized variants)

Then per dataset, Pearson r between per-cell rates and per-cell accuracies
across the 12 non-thinking models. The story for §3.1 / Fig 1 is that no
standard sample-based or self-evaluation baseline correlates with accuracy
at the model level -- only OC does.

Run as: python -m analysis.cross_model_baselines
"""
from collections import defaultdict
import numpy as np
from scipy import stats

from analysis.scoring import (
    score_official, is_correct_official, pairwise_official, cluster_entropy,
)
from analysis.data_loader import (
    load_nonthinking, load_baseline, NT_MODELS, NT_DATASETS,
)


def _f(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_cells():
    """Build dict[(model, dataset)] -> list of per-question rows with every
    baseline value attached."""
    p_true   = load_baseline('p_true')
    dsc      = load_baseline('decomposed_sc')
    prm_sky  = load_baseline('prm_skywork')
    prm_rlh  = load_baseline('prm_rlhflow')

    cells = defaultdict(list)
    for r in load_nonthinking():
        ds = r['dataset']; stem = r['model']; qid = r['question_id']
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        if len(nv) < 10:
            continue
        op_f1   = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0.0
        correct = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        key3    = (stem, ds, qid)
        # CoT-SC K=3 / K=10: only defined when the cot_shorts vector has K
        # non-empty samples available (we still allow trailing empties to
        # short-circuit so that extract-failure rates show up).
        cot_nonempty = [s for s in cot if s.strip()]
        cot_k3  = pairwise_official(cot_nonempty[:3],  ds) if len(cot_nonempty) >= 3  else None
        cot_k10 = pairwise_official(cot_nonempty[:10], ds) if len(cot_nonempty) >= 10 else None
        cells[(stem, ds)].append({
            'op_f1':       op_f1,
            'sc_k3':       pairwise_official(nv[:3], ds),
            'sc_k10':      pairwise_official(nv[:10], ds),
            'neg_H_k10':   -cluster_entropy(nv[:10], dataset=ds),
            'cot_k3':      cot_k3,
            'cot_k10':     cot_k10,
            'p_true':      _f(p_true.get(key3, {}).get('p_true')),
            'dsc':         _f(dsc.get(key3, {}).get('dsc_proper')),
            'prm_sky_min': _f(prm_sky.get(key3, {}).get('prm_min')),
            'prm_rlh_min': _f(prm_rlh.get(key3, {}).get('prm_min')),
            'correct':     correct,
        })
    return cells


def per_cell_acc(rows):
    return float(np.mean([r['correct'] for r in rows])) if rows else None


def per_cell_rate(rows, key, threshold=None):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) < 5:
        return None
    if threshold is not None:
        return sum(1 for v in vals if v >= threshold) / len(vals)
    return float(np.mean(vals))


def cross_model_r(cells, signal_key, threshold=None):
    """Returns dict[dataset] -> (r, p, n_models)."""
    out = {}
    for ds in NT_DATASETS:
        accs, rates = [], []
        for m in NT_MODELS:
            rows = cells.get((m, ds), [])
            if not rows:
                continue
            rate = per_cell_rate(rows, signal_key, threshold)
            if rate is None:
                continue
            accs.append(per_cell_acc(rows))
            rates.append(rate)
        if len(accs) < 3:
            out[ds] = (None, None, len(accs))
            continue
        r, p = stats.pearsonr(rates, accs)
        out[ds] = (float(r), float(p), len(accs))
    return out


def report():
    cells = build_cells()

    signals = [
        ('OC (binarized at 0.5)',     'op_f1',       0.5),
        ('OC (continuous)',           'op_f1',       None),
        ('CoT-SC K=3 (continuous)',   'cot_k3',      None),
        ('CoT-SC K=10 (continuous)',  'cot_k10',     None),
        ('naive SC K=3 (continuous)', 'sc_k3',       None),
        ('naive SC K=10 (continuous)', 'sc_k10',     None),
        ('-H K=10 (continuous)',      'neg_H_k10',   None),
        ('P(True) (continuous)',      'p_true',      None),
        ('dSC K=10 (continuous)',     'dsc',         None),
        ('Skywork-PRM (min)',         'prm_sky_min', None),
        ('RLHFlow-PRM (min)',         'prm_rlh_min', None),
    ]

    print('=' * 92)
    print(' Cross-model Pearson r per baseline (per-cell rate vs per-cell accuracy, '
          '12 non-thinking models)')
    print('=' * 92)
    print(f"\n  {'signal':<26}", end='')
    for ds in NT_DATASETS:
        print(f"  {ds:<14}", end='')
    print()
    print('  ' + '-' * 90)
    for label, key, thr in signals:
        print(f"  {label:<26}", end='')
        out = cross_model_r(cells, key, thr)
        for ds in NT_DATASETS:
            r, p, n = out[ds]
            if r is None:
                print(f"  {f'(n={n} <3)':<14}", end='')
            else:
                sig = '*' if p < 0.05 else ' '
                print(f"  {r:+.2f} (p={p:.3f}){sig}", end='')
        print()

    # Note on per-cell n_models for P(True) (only 9 of 12 models have logprobs).
    print()
    print('  * marks p < 0.05.')
    print('  P(True) is missing on Gemini-2.5-Flash, Llama-4-Maverick, Mistral-Large-2')
    print('    (no top-k logprobs in API), so it uses 9 models per dataset.')


if __name__ == '__main__':
    report()
