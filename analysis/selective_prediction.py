"""Selective prediction (per-question abstention) -- AUARC and coverage tables.

Per-cell metric (one (model, dataset) cell at a time):

  AUARC: area under the accuracy-coverage curve. Coverage runs over
    1..n / n; accuracy at coverage k is the mean correctness of the
    top-k highest-confidence questions. AUARC = mean(correct) for a
    random ranker; perfect ranker is bounded above by 1.
  acc@50: accuracy on the top-50% by confidence. Complementary view of
    the same curve, often more interpretable than AUARC.

Cell AUARCs are aggregated as paired comparisons across the 12
non-thinking (or 5 thinking) models per dataset. Combined predictors
use leave-one-model-out (LOMO) calibration: train logistic regression
on pooled rows from all other models, predict P(correct) on the held-out
cell. Inputs to the combiner are logit-transformed and an OC×SC
interaction feature is added by default.

Δ rows (paired differences between two signals) are reported with
bootstrap 95% CIs over the per-model paired diffs.
"""
from collections import defaultdict
import numpy as np
import statsmodels.api as sm

from analysis.scoring import (
    score_official, is_correct_official, pairwise_official, cluster_entropy,
)
from analysis.data_loader import (
    load_nonthinking, load_thinking, load_baseline,
    NT_MODELS, NT_DATASETS, T_MODELS, T_DATASETS,
)


# ── core: AUARC, acc@coverage, logit transform ────────────────────────────

def _stable_order_desc(scores):
    s = np.asarray(scores, dtype=float)
    n = len(s)
    return np.lexsort((np.arange(n), -s))


def auarc(scores, correct):
    """Area under the accuracy-coverage curve."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(s)
    if n == 0:
        return float('nan')
    order = _stable_order_desc(s)
    y = y[order]
    cum = np.cumsum(y)
    acc_at_k = cum / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return float(np.trapezoid(acc_at_k, coverage))


def acc_at_coverage(scores, correct, coverage=0.5):
    """Mean accuracy on the top-(coverage) fraction by confidence."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(s)
    if n == 0:
        return float('nan')
    k = max(1, int(round(coverage * n)))
    order = _stable_order_desc(s)
    return float(y[order][:k].mean())


def auroc(scores, correct):
    """AUROC: probability that a random correct example outscores a random
    incorrect one. Uses the Mann-Whitney U / average-rank formula so ties
    are handled correctly. Returns NaN if all-correct or all-incorrect."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=int)
    n = len(s)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='stable')
    s_sorted = s[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = (i + j + 2) / 2.0  # average of ranks (i+1)..(j+1)
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def coverage_at_accuracy(scores, correct, target):
    """Maximum coverage c such that mean(correct on top-c by score) >= target.

    Returns 0.0 if no contiguous prefix of the score-sorted list reaches the
    target accuracy. Uses non-monotone safe scan: the largest k with
    cum_acc(k) >= target.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(s)
    if n == 0:
        return float('nan')
    order = _stable_order_desc(s)
    y = y[order]
    cum = np.cumsum(y)
    acc_at_k = cum / np.arange(1, n + 1)
    above = np.where(acc_at_k >= target)[0]
    if len(above) == 0:
        return 0.0
    return float((above[-1] + 1) / n)


def _logit(x, eps=1e-3):
    """Stable logit on [0, 1] inputs. Clips to [eps, 1 - eps] before transform."""
    x = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _design(rows, predictors, *, transform=True):
    """Build LR design matrix. Logit-transforms predictors before stacking."""
    cols = [np.ones(len(rows))]
    raw = [np.array([float(r[p]) for r in rows]) for p in predictors]
    if transform:
        raw = [_logit(c) for c in raw]
    cols.extend(raw)
    return np.column_stack(cols)


# ── data builders -----------------------------------------------------------

def _f(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_nonthinking_cells():
    """Returns dict[(model, dataset)] -> list of per-question rows.

    Carries OC, SC_K=3, SC_K=10, -H_K=3, -H_K=10, P(True), NLI, dSC.
    Extra signals beyond OC/SC are loaded for the "best non-OC" comparison.
    """
    p_true = load_baseline('p_true')
    dsc    = load_baseline('decomposed_sc')
    nli    = load_baseline('nli')
    cells  = defaultdict(list)
    for r in load_nonthinking():
        ds = r['dataset']; stem = r['model']; qid = r['question_id']
        direct = r['op_direct']; exp = r['op_expansion']
        nv = r['nv_answers']; cot = r['cot_shorts']
        if len(nv) < 10:
            continue
        op_f1   = score_official(direct, exp, ds) if direct.strip() and exp.strip() else 0.0
        correct = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        key3    = (stem, ds, qid)
        # Map -H to [0, 1] via 1/(1 + H). Caps at H=0 -> 1.0.
        nH3  = 1.0 / (1.0 + cluster_entropy(nv[:3],  dataset=ds))
        nH10 = 1.0 / (1.0 + cluster_entropy(nv[:10], dataset=ds))
        # CoT-SC pairwise agreement at K=3 / K=10 (only defined when we have
        # enough non-empty extracted shorts).
        cot_ne = [s for s in cot if s.strip()]
        cot_k3  = pairwise_official(cot_ne[:3],  ds) if len(cot_ne) >= 3  else None
        cot_k10 = pairwise_official(cot_ne[:10], ds) if len(cot_ne) >= 10 else None
        cells[(stem, ds)].append({
            'qid':     qid,
            'op_f1':   op_f1,
            'sc_k3':   pairwise_official(nv[:3], ds),
            'sc_k10':  pairwise_official(nv[:10], ds),
            'cot_k3':  cot_k3,
            'cot_k10': cot_k10,
            'neg_H_k3':  nH3,
            'neg_H_k10': nH10,
            'p_true':    _f(p_true.get(key3, {}).get('p_true')),
            'nli':       _f(nli.get(key3, {}).get('nli_score')),
            'dsc':       _f(dsc.get(key3, {}).get('dsc_proper')),
            'correct':   correct,
        })
    return cells


def build_thinking_cells():
    """Returns dict[(model, dataset)] -> list of per-question rows."""
    cells = defaultdict(list)
    for r in load_thinking():
        ds = r['dataset']; stem = r['model']
        direct = r['direct_answer']; decomp = r['decomp_answer']
        sc3, sc10 = r['sc_k3'], r['sc_k10']
        if not direct.strip():
            continue
        op_f1   = score_official(direct, decomp, ds) if decomp.strip() else 0.0
        correct = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))
        nH3  = 1.0 / (1.0 + cluster_entropy(sc3,  dataset=ds)) if len(sc3)  >= 2 else None
        nH10 = 1.0 / (1.0 + cluster_entropy(sc10, dataset=ds)) if len(sc10) >= 5 else None
        cells[(stem, ds)].append({
            'qid':     r['question_id'],
            'op_f1':   op_f1,
            'sc_k3':   pairwise_official(sc3, ds)  if len(sc3)  >= 2 else None,
            'sc_k10':  pairwise_official(sc10, ds) if len(sc10) >= 5 else None,
            'neg_H_k3':  nH3,
            'neg_H_k10': nH10,
            'correct': correct,
        })
    return cells


# ── per-signal AUARC aggregation -------------------------------------------

def cell_metric(rows, signal_key, metric, target=None):
    """metric: 'auarc' | 'auroc' | 'cov_r10' | 'cov_r20'.

    'cov_r10', 'cov_r20': coverage at risk <= 0.10 (= acc >= 0.90), 0.20.
    """
    rs = [r for r in rows if r.get(signal_key) is not None]
    if len(rs) < 5:
        return None
    s = [r[signal_key] for r in rs]
    y = [r['correct']   for r in rs]
    if metric == 'auarc':
        return auarc(s, y)
    if metric == 'auroc':
        return auroc(s, y)
    if metric == 'cov_r10':
        return coverage_at_accuracy(s, y, 0.90)
    if metric == 'cov_r20':
        return coverage_at_accuracy(s, y, 0.80)
    raise ValueError(metric)


def lomo_combined(cells, models, ds, predictors, *, only_held_out=None,
                  transform=True):
    """LOMO logistic combiner. Returns dict[held_out_model] -> per-cell metrics.

    Each value is a dict {'auarc': ..., 'acc50': ...} for the held-out cell.
    """
    out = {}
    iter_models = [only_held_out] if only_held_out is not None else list(models)
    for held_out in iter_models:
        train_rows = []
        for m in models:
            if m == held_out:
                continue
            train_rows.extend(
                r for r in cells.get((m, ds), [])
                if all(r.get(p) is not None for p in predictors)
            )
        test_rows = [
            r for r in cells.get((held_out, ds), [])
            if all(r.get(p) is not None for p in predictors)
        ]
        if not train_rows or len(test_rows) < 5:
            continue
        Xtr = _design(train_rows, predictors, transform=transform)
        ytr = np.array([r['correct'] for r in train_rows])
        Xte = _design(test_rows, predictors, transform=transform)
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            continue
        try:
            res = sm.Logit(ytr, Xtr).fit(disp=False, maxiter=500)
            proba = np.asarray(res.predict(Xte))
        except Exception:
            continue
        y = [r['correct'] for r in test_rows]
        out[held_out] = {
            'auarc':   auarc(proba, y),
            'auroc':   auroc(proba, y),
            'cov_r10': coverage_at_accuracy(proba, y, 0.90),
            'cov_r20': coverage_at_accuracy(proba, y, 0.80),
        }
    return out


def cell_no_abstention(rows):
    rs = [r for r in rows if r.get('correct') is not None]
    if not rs:
        return None
    return float(np.mean([r['correct'] for r in rs]))


METRICS = ('auarc', 'auroc', 'cov_r10', 'cov_r20')


def aggregate(cells, models, datasets, signals, combined=None):
    """Build summary[(label, dataset, metric)] -> dict[model] -> value."""
    out = {}
    base = {}
    for ds in datasets:
        for m in models:
            rows = cells.get((m, ds), [])
            v = cell_no_abstention(rows)
            if v is not None:
                base[(m, ds)] = v
    for label, key in signals:
        for ds in datasets:
            for metric in METRICS:
                d = {}
                for m in models:
                    rows = cells.get((m, ds), [])
                    v = cell_metric(rows, key, metric)
                    if v is not None:
                        d[m] = v
                if d:
                    out[(label, ds, metric)] = d
    for label, preds in (combined or []):
        for ds in datasets:
            d = lomo_combined(cells, models, ds, preds)
            for metric in METRICS:
                dd = {m: v[metric] for m, v in d.items() if v.get(metric) is not None}
                if dd:
                    out[(label, ds, metric)] = dd
    # No-abstention reference: only the AUARC = base accuracy is meaningful.
    for ds in datasets:
        d = {}
        for m in models:
            v = base.get((m, ds))
            if v is not None:
                d[m] = v
        if d:
            out[('no abstention', ds, 'auarc')] = d
    return out


def _se(vals):
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _bootstrap_paired_ci(diffs, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI for the mean of paired differences."""
    if len(diffs) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs, dtype=float)
    n = len(a)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = a[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── pretty printers ---------------------------------------------------------

def _print_table(title, datasets, summary, row_order, metric):
    print('=' * 80)
    print(f' {title}')
    print('=' * 80)
    width = 26
    print(f"\n  {'signal':<{width}}", end='')
    for ds in datasets:
        print(f"  {ds:<13}", end='')
    print()
    print(f"  {'-'*width}", end='')
    for _ in datasets:
        print(f"  {'-'*13}", end='')
    print()
    for label_row in row_order:
        print(f"  {label_row:<{width}}", end='')
        for ds in datasets:
            d = summary.get((label_row, ds, metric))
            if d is None or not d:
                print(f"  {'--':<13}", end='')
            else:
                vals = list(d.values())
                m  = float(np.mean(vals))
                se = _se(vals)
                print(f"  {m:.3f}±{se:.3f}", end='')
        print()


def _print_marginal_ci(label, datasets, summary, label_with, label_without, metric):
    """Δ row with bootstrap 95% CI."""
    print()
    print(f"  {label:<26}", end='')
    for ds in datasets:
        a = summary.get((label_with, ds, metric))
        b = summary.get((label_without, ds, metric))
        if a is None or b is None:
            print(f"  {'--':<13}", end='')
            continue
        common = sorted(set(a) & set(b))
        if not common:
            print(f"  {'--':<13}", end='')
            continue
        diffs = [a[m] - b[m] for m in common]
        d_mean = float(np.mean(diffs))
        lo, hi = _bootstrap_paired_ci(diffs)
        sign = '+' if d_mean >= 0 else ''
        print(f" {sign}{d_mean:.3f}[{lo:+.3f},{hi:+.3f}]", end='')
    print()


# ── pipelines --------------------------------------------------------------

SIGNALS_BASE = [
    ('OC alone',                'op_f1'),
    ('CoT-SC (K=3) alone',      'cot_k3'),
    ('CoT-SC (K=10) alone',     'cot_k10'),
    ('naive-SC (K=3) alone',    'sc_k3'),
    ('naive-SC (K=10) alone',   'sc_k10'),
]
SIGNALS_NONTHINKING_EXTRA = [
    ('-H (K=10) alone',   'neg_H_k10'),
    ('P(True) alone',     'p_true'),
    ('dSC alone',         'dsc'),
]
COMBINED_BASE = [
    ('OC + CoT-SC_K=3 combined',   ['op_f1', 'cot_k3']),
    ('OC + CoT-SC_K=10 combined',  ['op_f1', 'cot_k10']),
    ('OC + naive-SC_K=3 combined', ['op_f1', 'sc_k3']),
    ('OC + naive-SC_K=10 combined', ['op_f1', 'sc_k10']),
]
ROW_ORDER_BASE = [
    'no abstention',
    'OC alone',
    'CoT-SC (K=3) alone', 'CoT-SC (K=10) alone',
    'naive-SC (K=3) alone', 'naive-SC (K=10) alone',
    'OC + CoT-SC_K=3 combined', 'OC + CoT-SC_K=10 combined',
    'OC + naive-SC_K=3 combined', 'OC + naive-SC_K=10 combined',
]


def best_non_oc_per_dataset(summary, datasets, candidate_labels, metric='auarc'):
    """For each dataset, return the candidate signal with highest mean metric.
    Returns dict[dataset] -> (label, signal_key)."""
    best = {}
    for ds in datasets:
        ranked = []
        for label, key in candidate_labels:
            d = summary.get((label, ds, metric))
            if d is None or not d:
                continue
            ranked.append((float(np.mean(list(d.values()))), label, key))
        if ranked:
            ranked.sort(reverse=True)
            best[ds] = (ranked[0][1], ranked[0][2])
    return best


def _print_deltas(datasets, summary, metric, with_best_non_oc=None):
    print()
    # Updated headline: vs CoT-SC (canonical Wang et al. baseline)
    _print_marginal_ci(
        f'Δ over CoT-SC_K=3 (3-call) [{metric}]',
        datasets, summary,
        'OC + CoT-SC_K=3 combined', 'CoT-SC (K=3) alone', metric,
    )
    _print_marginal_ci(
        f'Δ over CoT-SC_K=10 (10-call) [{metric}]',
        datasets, summary,
        'OC + CoT-SC_K=10 combined', 'CoT-SC (K=10) alone', metric,
    )
    # Legacy: vs naive-SC (direct-answer baseline)
    _print_marginal_ci(
        f'Δ over naive-SC_K=3 (3-call) [{metric}]',
        datasets, summary,
        'OC + naive-SC_K=3 combined', 'naive-SC (K=3) alone', metric,
    )
    _print_marginal_ci(
        f'Δ over naive-SC_K=10 (10-call) [{metric}]',
        datasets, summary,
        'OC + naive-SC_K=10 combined', 'naive-SC (K=10) alone', metric,
    )
    if with_best_non_oc:
        _print_marginal_ci(
            f'Δ over best non-OC [{metric}]',
            datasets, summary,
            'OC + best-non-OC combined', 'best non-OC alone', metric,
        )


def run_nonthinking():
    cells = build_nonthinking_cells()

    # First pass: compute univariate metrics for all candidate non-OC signals
    # so we can identify the best one per dataset.
    candidates = SIGNALS_BASE + SIGNALS_NONTHINKING_EXTRA
    summary = aggregate(cells, NT_MODELS, NT_DATASETS,
                         candidates, COMBINED_BASE)
    non_oc_candidates = [(l, k) for l, k in candidates if l != 'OC alone']
    best = best_non_oc_per_dataset(summary, NT_DATASETS, non_oc_candidates)

    # Second pass: per dataset, run LOMO with [op_f1, best_non_oc_key].
    # Collect the per-dataset results into shared 'best non-OC alone' and
    # 'OC + best-non-OC combined' entries.
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

    row_order = ROW_ORDER_BASE + [
        'best non-OC alone',
        'OC + best-non-OC combined',
    ]
    print()
    _print_table('Non-thinking AUARC (mean ± SE over 12 models per dataset)',
                  NT_DATASETS, summary, row_order, 'auarc')
    _print_deltas(NT_DATASETS, summary, 'auarc', with_best_non_oc=True)

    print()
    _print_table('Non-thinking AUROC (mean ± SE over 12 models per dataset)',
                  NT_DATASETS, summary, row_order, 'auroc')
    _print_deltas(NT_DATASETS, summary, 'auroc', with_best_non_oc=True)

    print()
    print('  Best non-OC signal per dataset (used in "best non-OC" rows above):')
    for ds in NT_DATASETS:
        if ds in best:
            print(f'    {ds:<12}  {best[ds][0]}')


def run_thinking():
    cells = build_thinking_cells()
    summary = aggregate(cells, T_MODELS, T_DATASETS,
                         SIGNALS_BASE, COMBINED_BASE)
    print()
    _print_table('Thinking AUARC (mean ± SE over 5 models per dataset)',
                  T_DATASETS, summary, ROW_ORDER_BASE, 'auarc')
    _print_deltas(T_DATASETS, summary, 'auarc')

    print()
    _print_table('Thinking AUROC (mean ± SE over 5 models per dataset)',
                  T_DATASETS, summary, ROW_ORDER_BASE, 'auroc')
    _print_deltas(T_DATASETS, summary, 'auroc')



if __name__ == '__main__':
    run_nonthinking()
    run_thinking()
