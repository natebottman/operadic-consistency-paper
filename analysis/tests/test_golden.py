"""Golden-output tests for the analysis pipeline.

These pin the headline paper numbers against the cached CSVs in data/.
If they fail, either:
  (a) someone changed scoring.py / common.py and the paper numbers shifted, or
  (b) the cached CSVs in data/per_question/ or data/baselines/ changed.

Re-run `python scripts/build_csvs.py` first if you've added or refreshed
raw data under data/raw/.
"""
import math
import pytest

from analysis.per_question_regressions  import build_per_question_rows, cross_model_r as _xmr
from analysis.thinking_regressions      import build_rows as build_thinking_rows
from analysis.loo_accuracy              import per_cell_signals
from analysis.common                    import logit_cluster
from analysis.data_loader               import NT_DATASETS

import numpy as np
from scipy import stats


# ── Headline cross-model Pearson r (Fig 1 captions) ─────────────────────────
@pytest.fixture(scope='module')
def per_model():
    _, pm = build_per_question_rows()
    return pm


@pytest.fixture(scope='module')
def by_dataset():
    bd, _ = build_per_question_rows()
    return bd


@pytest.mark.parametrize('ds, expected_r', [
    ('hotpotqa',   0.92),
    ('musique',    0.94),
    ('strategyqa', 0.94),
    ('drop',       0.86),
])
def test_cross_model_pearson_r(per_model, ds, expected_r):
    accs, ocs = [], []
    for (stem, dset), m in per_model.items():
        if dset != ds: continue
        accs.append(m['correct'] / m['n'])
        ocs.append(m['oc_count'] / m['n'])
    r, _ = stats.pearsonr(accs, ocs)
    assert abs(r - expected_r) < 0.01, f'r={r:+.3f} but paper={expected_r:+.2f}'


# ── Three-way per-question regression (Table 1) ─────────────────────────────
@pytest.mark.parametrize('ds, expected_b_oc', [
    ('hotpotqa',   2.12),
    ('musique',    2.40),
    ('strategyqa', 1.76),
    ('drop',       1.27),
])
def test_three_way_oc_coefficient(by_dataset, ds, expected_b_oc):
    rs = by_dataset[ds]
    qids = [r['qid'] for r in rs]
    res = logit_cluster(rs, ['op_f1', 'sc', 'neg_H'], cluster_groups=qids)
    assert res is not None, 'regression failed to fit'
    assert abs(res['b']['op_f1'] - expected_b_oc) < 0.01, \
        f"b_OC={res['b']['op_f1']:+.3f} but paper={expected_b_oc:+.2f}"


# ── Univariate OC pseudo-R^2 ────────────────────────────────────────────────
@pytest.mark.parametrize('ds, expected_pR2', [
    ('hotpotqa',   0.179),
    ('musique',    0.199),
    ('strategyqa', 0.180),
    ('drop',       0.089),
])
def test_oc_univariate_pseudo_r2(by_dataset, ds, expected_pR2):
    rs = by_dataset[ds]
    qids = [r['qid'] for r in rs]
    res = logit_cluster(rs, ['op_f1'], cluster_groups=qids)
    assert res is not None
    assert abs(res['pR2'] - expected_pR2) < 0.005, \
        f"pR2={res['pR2']:.3f} but paper={expected_pR2:.3f}"


# ── Thinking-model K=3 bivariate (Table 2 left) ─────────────────────────────
@pytest.fixture(scope='module')
def thinking_rows():
    return build_thinking_rows()


@pytest.mark.parametrize('ds, expected_b_oc, expected_pR2', [
    ('musique',    1.16, 0.086),
    ('strategyqa', 0.90, 0.114),
    ('gsm8k',      2.94, 0.417),
    ('drop',       0.87, 0.053),
])
def test_thinking_k3_bivariate(thinking_rows, ds, expected_b_oc, expected_pR2):
    rs = [r for r in thinking_rows if r['dataset'] == ds and r['sc_k3'] is not None]
    qids = [r['qid'] for r in rs]
    res = logit_cluster(rs, ['op_f1', 'sc_k3'], cluster_groups=qids)
    assert res is not None
    assert abs(res['b']['op_f1'] - expected_b_oc) < 0.02, \
        f"{ds}: b_OC={res['b']['op_f1']:+.3f} but paper={expected_b_oc:+.2f}"
    assert abs(res['pR2'] - expected_pR2) < 0.005, \
        f"{ds}: pR2={res['pR2']:.3f} but paper={expected_pR2:.3f}"


# ── S3.2 prose claims: signal correlations and joint partition ──────────────
def test_thinking_signal_correlations(thinking_rows):
    """Pins the two Pearson r's quoted in S3.2 prose: SC/-H near-collinearity
    (+0.97 at K=10) vs OC's weak coupling to sample diversity (+0.39 at K=3)."""
    from scipy import stats
    k10 = [r for r in thinking_rows
           if r['sc_k10'] is not None and r['neg_H10'] is not None]
    r_sc_nh = stats.pearsonr([r['sc_k10'] for r in k10],
                             [r['neg_H10'] for r in k10])[0]
    assert abs(r_sc_nh - 0.97) < 0.005, f'Pearson(SC,-H)={r_sc_nh:+.3f} vs paper +0.97'
    k3 = [r for r in thinking_rows if r['sc_k3'] is not None]
    r_oc_sc = stats.pearsonr([r['op_f1'] for r in k3],
                             [r['sc_k3'] for r in k3])[0]
    assert abs(r_oc_sc - 0.39) < 0.005, f'Pearson(OC,SC3)={r_oc_sc:+.3f} vs paper +0.39'


@pytest.mark.parametrize('ds, expected_gap_pp', [
    # S3.2 prose: "jointly-consistent questions are answered correctly
    # 43--62 pp higher than jointly-inconsistent ones."
    ('musique',    43),
    ('strategyqa', 45),
    ('gsm8k',      62),
    ('drop',       43),
])
def test_thinking_joint_partition_gap(ds, expected_gap_pp):
    import itertools
    from analysis.scoring import score_official, is_correct_official
    from analysis.data_loader import load_thinking
    both, neither = [], []
    for r in load_thinking():
        if r['dataset'] != ds:
            continue
        direct = r['direct_answer']; decomp = r['decomp_answer']; sc10 = r['sc_k10']
        if not direct.strip() or len(sc10) < 5:
            continue
        ocb = (score_official(direct, decomp, ds) if decomp.strip() else 0.0) >= 0.5
        unanimous = all(score_official(a, b, ds) >= 0.5
                        for a, b in itertools.combinations(sc10, 2))
        corr = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        if ocb and unanimous:
            both.append(corr)
        elif not ocb and not unanimous:
            neither.append(corr)
    gap_pp = (np.mean(both) - np.mean(neither)) * 100
    assert abs(gap_pp - expected_gap_pp) < 1.0, \
        f'{ds}: gap={gap_pp:+.1f} pp vs paper {expected_gap_pp:+d} pp'


# ── LOO accuracy prediction (Table 3) ───────────────────────────────────────
@pytest.mark.parametrize('ds, variant, expected_pR2', [
    # F1 (paper's primary)
    ('hotpotqa',   'F1',     0.241),
    ('strategyqa', 'F1',     0.283),
    # NLI alternative
    ('hotpotqa',   'NLI',    0.207),
    ('strategyqa', 'NLI',    0.218),
    # GPT-4o-mini strict CoT
    ('hotpotqa',   'mini_S', 0.213),
    ('strategyqa', 'mini_S', 0.223),
])
def test_alt_oc_scorers_3way_pR2(ds, variant, expected_pR2):
    """App. A alternative-OC-scorers table -- 3-way regression pseudo-R^2 per OC variant."""
    from analysis.oc_alternative_scorers import build_rows
    by_dataset, _ = build_rows()
    rows = by_dataset[ds]
    qids = [r['qid'] for r in rows]
    res = logit_cluster(rows, [f'op_{variant}', 'sc', 'neg_H'], cluster_groups=qids)
    assert res is not None
    assert abs(res['pR2'] - expected_pR2) < 0.005, \
        f'{ds}/{variant}: pR2={res["pR2"]:.3f} but paper={expected_pR2:.3f}'


# ── Selective prediction (AUARC + AUROC, §3.1 + §3.2) ───────────────────────
@pytest.fixture(scope='module')
def selective_nt():
    from analysis.selective_prediction import (
        build_nonthinking_cells, aggregate, best_non_oc_per_dataset,
        lomo_combined, METRICS,
        SIGNALS_BASE, SIGNALS_NONTHINKING_EXTRA, COMBINED_BASE,
    )
    from analysis.data_loader import NT_MODELS, NT_DATASETS
    cells = build_nonthinking_cells()
    candidates = SIGNALS_BASE + SIGNALS_NONTHINKING_EXTRA
    summary = aggregate(cells, NT_MODELS, NT_DATASETS, candidates, COMBINED_BASE)
    non_oc = [(l, k) for l, k in candidates if l != 'OC alone']
    best = best_non_oc_per_dataset(summary, NT_DATASETS, non_oc)
    for ds, (best_label, best_key) in best.items():
        for metric in METRICS:
            key = (best_label, ds, metric)
            if key in summary:
                summary[('best non-OC alone', ds, metric)] = summary[key]
        d = lomo_combined(cells, NT_MODELS, ds, ['op_f1', best_key])
        for metric in METRICS:
            dd = {m: v[metric] for m, v in d.items() if v.get(metric) is not None}
            if dd:
                summary[('OC + best-non-OC combined', ds, metric)] = dd
    return summary


@pytest.fixture(scope='module')
def selective_t():
    from analysis.selective_prediction import (
        build_thinking_cells, aggregate,
        SIGNALS_BASE, COMBINED_BASE,
    )
    from analysis.data_loader import T_MODELS, T_DATASETS
    cells = build_thinking_cells()
    return aggregate(cells, T_MODELS, T_DATASETS, SIGNALS_BASE, COMBINED_BASE)


@pytest.mark.parametrize('label, ds, metric, expected', [
    # Non-thinking: pin one cell per metric and signal-class.
    ('OC alone',                       'hotpotqa',   'auarc', 0.421),
    ('OC alone',                       'musique',    'auroc', 0.776),
    ('naive-SC (K=10) alone',          'strategyqa', 'auarc', 0.808),
    ('OC + naive-SC_K=10 combined',    'musique',    'auarc', 0.856),
    ('OC + naive-SC_K=10 combined',    'strategyqa', 'auroc', 0.809),
    ('OC + best-non-OC combined',      'drop',       'auroc', 0.772),
])
def test_selective_nt_point_estimates(selective_nt, label, ds, metric, expected):
    d = selective_nt.get((label, ds, metric))
    assert d, f'missing ({label}, {ds}, {metric})'
    got = float(np.mean(list(d.values())))
    assert abs(got - expected) < 0.005, f'{label}/{ds}/{metric}: {got:.3f} vs {expected:.3f}'


@pytest.mark.parametrize('ds, metric, expected_delta', [
    # Δ rows: OC + naive-SC_K=10 minus naive-SC_K=10 alone (paired mean).
    # The paper headlines deltas vs CoT-SC (canonical CoT self-consistency);
    # this test pins the analogous deltas vs the naive direct-answer SC
    # ablation so the same harness covers thinking models too (CoT-SC is
    # not computed for them).
    ('hotpotqa',   'auarc', 0.056),
    ('musique',    'auarc', 0.073),
    ('strategyqa', 'auarc', 0.054),
    ('drop',       'auarc', 0.037),
    ('hotpotqa',   'auroc', 0.054),
    ('musique',    'auroc', 0.128),
    ('strategyqa', 'auroc', 0.097),
    ('drop',       'auroc', 0.049),
])
def test_selective_nt_delta_over_sc_k10(selective_nt, ds, metric, expected_delta):
    a = selective_nt.get(('OC + naive-SC_K=10 combined', ds, metric))
    b = selective_nt.get(('naive-SC (K=10) alone',       ds, metric))
    common = sorted(set(a) & set(b))
    diffs = [a[m] - b[m] for m in common]
    got = float(np.mean(diffs))
    assert abs(got - expected_delta) < 0.005, f'{ds}/{metric}: Δ={got:+.3f} vs paper={expected_delta:+.3f}'


@pytest.mark.parametrize('ds, metric, expected_delta', [
    # Thinking K=10 vs naive-SC_K=10 (CoT-SC not computed for thinking).
    ('musique',    'auarc',  0.025),
    ('strategyqa', 'auarc',  0.013),
    ('gsm8k',      'auarc',  0.005),
    ('drop',       'auarc',  0.015),
    ('musique',    'auroc',  0.023),
    ('strategyqa', 'auroc',  0.012),
])
def test_selective_t_delta_over_sc_k10(selective_t, ds, metric, expected_delta):
    a = selective_t.get(('OC + naive-SC_K=10 combined', ds, metric))
    b = selective_t.get(('naive-SC (K=10) alone',       ds, metric))
    common = sorted(set(a) & set(b))
    diffs = [a[m] - b[m] for m in common]
    got = float(np.mean(diffs))
    assert abs(got - expected_delta) < 0.01, f'{ds}/{metric}: Δ={got:+.3f} vs paper={expected_delta:+.3f}'


# ── Cross-model Pearson r per baseline (App. A defensive table) ─────────────
@pytest.fixture(scope='module')
def cross_model_baseline_cells():
    from analysis.cross_model_baselines import build_cells
    return build_cells()


@pytest.mark.parametrize('signal, threshold, ds, expected_r', [
    # OC headline: r >= 0.85 on every dataset (binarized at 0.5).
    ('op_f1',       0.5,  'hotpotqa',   0.92),
    ('op_f1',       0.5,  'musique',    0.94),
    ('op_f1',       0.5,  'strategyqa', 0.94),
    ('op_f1',       0.5,  'drop',       0.86),
    # SC family: significant on strategyqa under matched-axis scoring,
    # weak elsewhere.
    ('sc_k10',      None, 'strategyqa', 0.70),
    # Decomposition-aware partial successes (must remain isolated).
    ('dsc',         None, 'hotpotqa',   0.67),
    ('prm_sky_min', None, 'drop',       0.82),
])
def test_cross_model_r(cross_model_baseline_cells, signal, threshold, ds, expected_r):
    from analysis.cross_model_baselines import cross_model_r
    out = cross_model_r(cross_model_baseline_cells, signal, threshold)
    r, p, n = out[ds]
    assert r is not None, f'{signal} on {ds}: not enough cells (n={n})'
    assert abs(r - expected_r) < 0.02, \
        f'{signal} on {ds}: r={r:+.3f} but paper={expected_r:+.2f}'


# ── LOO accuracy prediction (Table 3) ───────────────────────────────────────
@pytest.mark.parametrize('ds, expected_mae, expected_max', [
    ('hotpotqa',   0.028, 0.062),
    ('musique',    0.023, 0.069),
    ('strategyqa', 0.021, 0.052),
    ('drop',       0.047, 0.131),
])
def test_loo_mae(ds, expected_mae, expected_max):
    cells = per_cell_signals()
    pairs = [cells[(s, ds)] for s in {k[0] for k in cells} if (s, ds) in cells]
    ocs  = np.array([p['oc']  for p in pairs])
    accs = np.array([p['acc'] for p in pairs])
    n = len(ocs)
    loo_errs = []
    for i in range(n):
        oc_train  = np.concatenate([ocs[:i],  ocs[i+1:]])
        acc_train = np.concatenate([accs[:i], accs[i+1:]])
        a, b = np.polyfit(oc_train, acc_train, 1)
        loo_errs.append(a * ocs[i] + b - accs[i])
    loo_errs = np.array(loo_errs)
    mae    = np.abs(loo_errs).mean()
    maxerr = np.abs(loo_errs).max()
    assert abs(mae - expected_mae) < 0.001, f"{ds}: MAE={mae:.3f} vs paper={expected_mae:.3f}"
    assert abs(maxerr - expected_max) < 0.001, f"{ds}: max={maxerr:.3f} vs paper={expected_max:.3f}"
