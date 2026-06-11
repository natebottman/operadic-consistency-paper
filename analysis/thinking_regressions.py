"""Table 2 of the paper plus the §3.2 thinking-model numerical claims.

Reproduces:
- K=3 bivariate per-dataset:  correct ~ OC + SC_K=3
- K=10 three-way pooled:      correct ~ OC + SC_K=10 + (-H_K=10)
- Pooled K=3 bivariate:       correct ~ OC + SC_K=3 (across all 4 datasets)
- Univariate K=3 and K=10 standalone pseudo-R^2
"""
from collections import defaultdict
from analysis.scoring import (
    score_official, is_correct_official, pairwise_official, cluster_entropy,
)
from analysis.common import logit_cluster, fp
from analysis.data_loader import load_thinking, T_DATASETS

def build_rows():
    rows = []
    for r in load_thinking():
        ds = r['dataset']
        direct = r['direct_answer']
        decomp = r['decomp_answer']
        sc3, sc10 = r['sc_k3'], r['sc_k10']
        if not direct.strip():
            continue
        op_score = score_official(direct, decomp, ds) if decomp.strip() else 0.0
        correct  = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        rows.append({
            'qid':     r['question_id'],
            'op_f1':   op_score,
            'sc_k3':   pairwise_official(sc3, ds)  if len(sc3)  >= 2 else None,
            'sc_k10':  pairwise_official(sc10, ds) if len(sc10) >= 5 else None,
            'neg_H10': -cluster_entropy(sc10, dataset=ds) if len(sc10) >= 5 else None,
            'correct': correct,
            'dataset': ds,
        })
    return rows


def k3_bivariate_per_dataset(rows):
    print('=' * 80)
    print(' K=3 bivariate per dataset (Table 2 left): correct ~ OC + SC_K=3')
    print('=' * 80)
    print(f"\n  {'dataset':<12} {'n':>5}  {'OC':>20}  {'SC_K=3':>20}  {'pR2':>6}")
    by = defaultdict(list)
    for r in rows:
        if r['sc_k3'] is not None:
            by[r['dataset']].append(r)
    for ds in T_DATASETS:
        rs = by[ds]
        qids = [r['qid'] for r in rs]
        res = logit_cluster(rs, ['op_f1', 'sc_k3'], cluster_groups=qids)
        if not res or 'error' in res:
            print(f'  {ds:<12} skip'); continue
        print(f"  {ds:<12} {res['n']:>5}  "
              f"{res['b']['op_f1']:>+5.2f} {fp(res['p']['op_f1']):>13}  "
              f"{res['b']['sc_k3']:>+5.2f} {fp(res['p']['sc_k3']):>13}  "
              f"{res['pR2']:.3f}")


def k10_three_way_pooled(rows):
    print('\n' + '=' * 80)
    print(' K=10 three-way pooled across all 20 cells: correct ~ OC + SC_K=10 + (-H_K=10)')
    print('=' * 80)
    rs = [r for r in rows if r['sc_k10'] is not None and r['neg_H10'] is not None]
    qids = [r['qid'] for r in rs]
    res = logit_cluster(rs, ['op_f1', 'sc_k10', 'neg_H10'], cluster_groups=qids)
    if not res or 'error' in res:
        print('  skip'); return
    print(f"\n  n={res['n']}  "
          f"b_OC={res['b']['op_f1']:+.3f} ({fp(res['p']['op_f1'])})  "
          f"b_SC={res['b']['sc_k10']:+.3f} ({fp(res['p']['sc_k10'])})  "
          f"b_-H={res['b']['neg_H10']:+.3f} ({fp(res['p']['neg_H10'])})  "
          f"pR2={res['pR2']:.3f}")


def k3_pooled_bivariate(rows):
    print('\n' + '=' * 80)
    print(' Pooled K=3 bivariate: correct ~ OC + SC_K=3')
    print('=' * 80)
    rs = [r for r in rows if r['sc_k3'] is not None]
    qids = [r['qid'] for r in rs]
    res = logit_cluster(rs, ['op_f1', 'sc_k3'], cluster_groups=qids)
    if not res or 'error' in res:
        print('  skip'); return
    print(f"\n  n={res['n']}  "
          f"b_OC={res['b']['op_f1']:+.3f} ({fp(res['p']['op_f1'])})  "
          f"b_SC={res['b']['sc_k3']:+.3f} ({fp(res['p']['sc_k3'])})  "
          f"pR2={res['pR2']:.3f}")


def univariate(rows):
    print('\n' + '=' * 80)
    print(' Univariate standalone pseudo-R^2 (pooled, cluster-robust SE)')
    print('=' * 80)
    print()
    for budget, key in [('K=3 OC', ('op_f1', 'sc_k3')),
                         ('K=3 SC_K=3', ('sc_k3', 'sc_k3')),
                         ('K=10 OC', ('op_f1', 'sc_k10')),
                         ('K=10 SC_K=10', ('sc_k10', 'sc_k10')),
                         ('K=10 -H_K=10', ('neg_H10', 'sc_k10'))]:
        target_key, filter_key = key
        rs = [r for r in rows if r[filter_key] is not None and r[target_key] is not None]
        qids = [r['qid'] for r in rs]
        res = logit_cluster(rs, [target_key], cluster_groups=qids)
        if not res or 'error' in res:
            continue
        print(f"  {budget:<20} n={res['n']}  "
              f"b={res['b'][target_key]:+.3f} ({fp(res['p'][target_key])})  "
              f"pR2={res['pR2']:.3f}")


def signal_correlations(rows):
    """The S3.2 prose claims: SC/-H near-collinearity vs OC's weak coupling."""
    import numpy as np
    from scipy import stats
    print('\n' + '=' * 80)
    print(' Signal correlations (S3.2 prose)')
    print('=' * 80)
    k10 = [r for r in rows if r['sc_k10'] is not None and r['neg_H10'] is not None]
    sc10 = np.array([r['sc_k10'] for r in k10])
    nh10 = np.array([r['neg_H10'] for r in k10])
    print(f"\n  Pearson(SC_K=10, -H_K=10)  n={len(k10)}  "
          f"r={stats.pearsonr(sc10, nh10)[0]:+.2f}")
    k3 = [r for r in rows if r['sc_k3'] is not None]
    oc3 = np.array([r['op_f1'] for r in k3])
    sc3 = np.array([r['sc_k3'] for r in k3])
    print(f"  Pearson(OC, SC_K=3)        n={len(k3)}  "
          f"r={stats.pearsonr(oc3, sc3)[0]:+.2f}")


def joint_partition(rows=None):
    """The S3.2 binary (OC, SC unanimous at K=10) partition: per-dataset
    accuracy gap between jointly-consistent and jointly-inconsistent
    questions. Unanimity = all pairs agree at matched-axis score >= 0.5."""
    import itertools
    import numpy as np
    from collections import defaultdict
    print('\n' + '=' * 80)
    print(' Binary (OC >= 0.5, SC_K=10 unanimous) partition (S3.2 prose)')
    print('=' * 80)
    print(f"\n  {'dataset':<12} {'joint-consistent':>17} {'joint-inconsistent':>19} {'gap':>8}")
    cells = defaultdict(lambda: defaultdict(list))
    for r in load_thinking():
        ds = r['dataset']; direct = r['direct_answer']
        decomp = r['decomp_answer']; sc10 = r['sc_k10']
        if not direct.strip() or len(sc10) < 5:
            continue
        ocb = (score_official(direct, decomp, ds) if decomp.strip() else 0.0) >= 0.5
        unanimous = all(score_official(a, b, ds) >= 0.5
                        for a, b in itertools.combinations(sc10, 2))
        corr = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        cells[ds][(ocb, unanimous)].append(corr)
    for ds in T_DATASETS:
        both = cells[ds].get((True, True), [])
        neither = cells[ds].get((False, False), [])
        if not both or not neither:
            print(f'  {ds:<12} skip'); continue
        a, b = np.mean(both), np.mean(neither)
        print(f"  {ds:<12} {a:>9.3f} (n={len(both):>3}) {b:>9.3f} (n={len(neither):>3}) "
              f"{(a - b) * 100:>+6.0f} pp")


if __name__ == '__main__':
    rows = build_rows()
    k3_bivariate_per_dataset(rows)
    k10_three_way_pooled(rows)
    k3_pooled_bivariate(rows)
    univariate(rows)
    signal_correlations(rows)
    joint_partition()
