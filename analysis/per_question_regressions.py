"""Tables 1, 4, and 5 of the paper plus the cross-model Pearson r reported in Fig 3.

Reproduces:
- Cross-model Pearson r between accuracy and OC-rate (Fig 3 captions)
- Three-way regression: correct ~ OC + SC + (-H)         (Table 1)
- Four-way regression with P(True), 9 models             (Table 4)
- Four-way regressions: + dSC, + NLI, + Skywork, + RLHFlow (Table 5)
- Univariate standalone pseudo-R^2                       (App. A)
"""
from collections import defaultdict
import numpy as np
from scipy import stats

from analysis.scoring import (
    score_official, is_correct_official, pairwise_official, cluster_entropy,
)
from analysis.common import logit_cluster, fp
from analysis.data_loader import (
    load_nonthinking, load_baseline, NT_MODELS, NT_DATASETS,
)


def build_per_question_rows():
    """Build per-question rows with (op_f1, sc, neg_H, correct) plus baseline columns."""
    p_true     = load_baseline('p_true')
    dsc        = load_baseline('decomposed_sc')
    nli        = load_baseline('nli')
    prm_sky    = load_baseline('prm_skywork')
    prm_rlh    = load_baseline('prm_rlhflow')

    by_dataset = defaultdict(list)
    per_model = {}

    for r in load_nonthinking():
        ds = r['dataset']; stem = r['model']; qid = r['question_id']
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        if len(nv) < 10:
            continue
        op_f1   = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0.0
        sc      = pairwise_official(nv[:10], ds)
        neg_H   = -cluster_entropy(nv[:10], dataset=ds)
        # CoT-SC K=10 pairwise agreement; only defined when extracted shorts
        # have >=10 non-empty samples available.
        cot_nonempty = [s for s in cot if s.strip()]
        cot_sc = pairwise_official(cot_nonempty[:10], ds) if len(cot_nonempty) >= 10 else None
        correct = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))

        key = (stem, ds, qid)
        row = {
            'qid':     qid,
            'op_f1':   op_f1,
            'sc':      sc,
            'cot_sc':  cot_sc,
            'neg_H':   neg_H,
            'correct': correct,
            'p_true':      _f(p_true.get(key, {}).get('p_true')),
            'dsc_proper':  _f(dsc.get(key, {}).get('dsc_proper')),
            'nli_score':   _f(nli.get(key, {}).get('nli_score')),
            'prm_sky_min': _f(prm_sky.get(key, {}).get('prm_min')),
            'prm_rlh_min': _f(prm_rlh.get(key, {}).get('prm_min')),
        }
        by_dataset[ds].append(row)

        # Per-model summary for cross-model r
        per_model.setdefault((stem, ds), {'n': 0, 'oc_count': 0, 'correct': 0})
        m = per_model[(stem, ds)]
        m['n'] += 1
        if op_f1 >= 0.5: m['oc_count'] += 1
        if correct:      m['correct']  += 1
    return by_dataset, per_model


def _f(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cross_model_r(per_model):
    print('=' * 80)
    print(' Cross-model Pearson r (12 models)')
    print('=' * 80)
    print(f"\n  {'dataset':<12}  {'r_OC':>8} {'p_OC':>10}")
    for ds in NT_DATASETS:
        accs, ocs = [], []
        for stem in NT_MODELS:
            m = per_model.get((stem, ds))
            if not m: continue
            accs.append(m['correct'] / m['n'])
            ocs.append(m['oc_count'] / m['n'])
        r, p = stats.pearsonr(accs, ocs)
        print(f"  {ds:<12}  {r:>+8.2f} {p:>10.4f}")


def three_way_main(by_dataset):
    """UPDATED Table 1 (left panel) per-question regression.

    correct ~ OC + CoT-SC_{K=10} + (-H_{K=10})
    Cluster-robust SE by question. Uses canonical Wang et al. CoT-SC; the
    naive direct-answer SC variant is included as an ablation in App. A.4
    only, not in the main-text regression.
    """
    print('\n' + '=' * 92)
    print(' Three-way per-question regression (Table 1, cluster-robust SE by question)')
    print(' correct ~ OC + CoT-SC_K=10 + (-H_K=10)')
    print('=' * 92)
    print(f"\n  {'dataset':<12} {'n':>5}  "
          f"{'OC':>20}  {'CoT-SC':>20}  {'-H':>20}  {'pR2':>6}")
    for ds in NT_DATASETS:
        rs = [r for r in by_dataset[ds] if r['cot_sc'] is not None]
        qids = [r['qid'] for r in rs]
        res = logit_cluster(rs, ['op_f1', 'cot_sc', 'neg_H'], cluster_groups=qids)
        if not res or 'error' in res:
            print(f'  {ds:<12} skip'); continue
        print(f"  {ds:<12} {res['n']:>5}  "
              f"{res['b']['op_f1']:>+5.2f} {fp(res['p']['op_f1']):>13}  "
              f"{res['b']['cot_sc']:>+5.2f} {fp(res['p']['cot_sc']):>13}  "
              f"{res['b']['neg_H']:>+5.2f} {fp(res['p']['neg_H']):>13}  "
              f"{res['pR2']:.3f}")


def four_way_with_p_true(by_dataset):
    """App. A.10 sensitivity: adds P(True) to the main regression.

    correct ~ OC + CoT-SC + (-H) + P(True)
    Restricted to the 9 models with top-k logprob access (P(True) is
    undefined on Gemini-2.5-Flash, Llama-4-Maverick, Mistral-Large-2)."""
    print('\n' + '=' * 92)
    print(' Four-way per-question regression (App. A.10 sensitivity, 9 models)')
    print(' correct ~ OC + CoT-SC_K=10 + (-H_K=10) + P(True)')
    print('=' * 92)
    print(f"\n  {'dataset':<12} {'n':>5}  "
          f"{'OC':>20}  {'CoT-SC':>20}  {'-H':>20}  {'P(True)':>20}  {'pR2':>6}")
    for ds in NT_DATASETS:
        rs = [r for r in by_dataset[ds]
              if r['cot_sc'] is not None and r['p_true'] is not None]
        qids = [r['qid'] for r in rs]
        res = logit_cluster(rs, ['op_f1', 'cot_sc', 'neg_H', 'p_true'],
                            cluster_groups=qids)
        if not res or 'error' in res:
            print(f'  {ds:<12} skip'); continue
        print(f"  {ds:<12} {res['n']:>5}  "
              f"{res['b']['op_f1']:>+5.2f} {fp(res['p']['op_f1']):>13}  "
              f"{res['b']['cot_sc']:>+5.2f} {fp(res['p']['cot_sc']):>13}  "
              f"{res['b']['neg_H']:>+5.2f} {fp(res['p']['neg_H']):>13}  "
              f"{res['b']['p_true']:>+5.2f} {fp(res['p']['p_true']):>13}  "
              f"{res['pR2']:.3f}")


def four_way(by_dataset, label, key):
    """4-way: correct ~ OC + CoT-SC + (-H) + {label}.

    Uses the updated 3-way base (with CoT-SC, not naive-SC) plus one
    decomposition-aware predictor at a time. This mirrors the paper's
    Table 5 structure but with the updated main regression base."""
    print(f"\n  4-way: correct ~ OC + CoT-SC + (-H) + {label}")
    print(f"  {'dataset':<12} {'n':>5}  {'OC':>20}  {label:>20}  {'pR2':>6}")
    for ds in NT_DATASETS:
        rs = [r for r in by_dataset[ds]
              if r[key] is not None and r['cot_sc'] is not None]
        qids = [r['qid'] for r in rs]
        res = logit_cluster(rs, ['op_f1', 'cot_sc', 'neg_H', key],
                            cluster_groups=qids)
        if not res or 'error' in res:
            print(f'  {ds:<12} skip'); continue
        print(f"  {ds:<12} {res['n']:>5}  "
              f"{res['b']['op_f1']:>+5.2f} {fp(res['p']['op_f1']):>13}  "
              f"{res['b'][key]:>+5.2f} {fp(res['p'][key]):>13}  "
              f"{res['pR2']:.3f}")


def univariate(by_dataset):
    print('\n' + '=' * 80)
    print(' Univariate standalone pseudo-R^2 (cluster-robust SE)')
    print('=' * 80)
    preds = [
        ('OC (op_f1)',         'op_f1'),
        ('CoT-SC (K=10)',      'cot_sc'),
        ('-H (K=10)',          'neg_H'),
        ('P(True)',            'p_true'),
        ('naive-SC (K=10)',    'sc'),  # appendix-only for cross-model table
        ('dSC proper',         'dsc_proper'),
        ('NLI score',          'nli_score'),
        ('Skywork-PRM (min)',  'prm_sky_min'),
        ('RLHFlow-PRM (min)',  'prm_rlh_min'),
    ]
    print(f"\n  {'predictor':<22}", end='')
    for ds in NT_DATASETS: print(f'  {ds:<10}', end='')
    print()
    for label, key in preds:
        print(f'  {label:<22}', end='')
        for ds in NT_DATASETS:
            rs = [r for r in by_dataset[ds] if r[key] is not None]
            qids = [r['qid'] for r in rs]
            res = logit_cluster(rs, [key], cluster_groups=qids)
            if not res or 'error' in res:
                print(f"  {'-':<10}", end='')
            else:
                print(f"  {res['pR2']:.3f}    ", end='')
        print()


if __name__ == '__main__':
    by_dataset, per_model = build_per_question_rows()
    cross_model_r(per_model)
    three_way_main(by_dataset)
    four_way_with_p_true(by_dataset)
    print('\n' + '=' * 80)
    print(' Five-way regressions (Table 5): each adds one decomposition-aware baseline')
    print(' to the new OC + CoT-SC + -H base.')
    print('=' * 80)
    four_way(by_dataset, 'dSC (proper)',       'dsc_proper')
    four_way(by_dataset, 'NLI score',          'nli_score')
    four_way(by_dataset, 'Skywork-PRM (min)',  'prm_sky_min')
    four_way(by_dataset, 'RLHFlow-PRM (min)',  'prm_rlh_min')
    univariate(by_dataset)
