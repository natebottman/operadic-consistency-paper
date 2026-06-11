"""Regression helpers and formatting used across analysis scripts."""
import math
import numpy as np
import statsmodels.api as sm


def fp(p):
    """Format a p-value compactly: 10^k for very small, decimal for borderline, n.s. else."""
    if p is None:
        return '--'
    if p < 1e-100:
        return '<10^{-100}'
    if p < 1e-3:
        return f'10^{{{int(math.floor(math.log10(p)))}}}'
    if p < 0.05:
        return f'{p:.3f}'
    return 'n.s.'


def logit_cluster(rows, predictors, ycol='correct', cluster_groups=None):
    """Fit logistic regression with optional cluster-robust SE.

    rows: list of dicts. Each dict must have `ycol` (0/1) and each name in `predictors`.
    cluster_groups: optional list of group ids (e.g. question_id) for cluster-robust SE.

    Returns dict with keys: b (coefficients), p (p-values), pR2 (McFadden), n.
    Returns None if not enough data; returns {'error': ...} if fit fails.
    """
    if not rows:
        return None
    X = np.column_stack(
        [[1.0] * len(rows)] + [[float(r[p]) for r in rows] for p in predictors]
    )
    y = np.array([r[ycol] for r in rows])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    try:
        if cluster_groups is not None:
            res = sm.Logit(y, X).fit(
                disp=False, maxiter=500,
                cov_type='cluster',
                cov_kwds={'groups': np.array(cluster_groups)},
            )
        else:
            res = sm.Logit(y, X).fit(disp=False, maxiter=500)
    except Exception as e:
        return {'error': str(e)[:80], 'n': len(rows)}
    return {
        'b':   {p: res.params[i + 1]  for i, p in enumerate(predictors)},
        'p':   {p: res.pvalues[i + 1] for i, p in enumerate(predictors)},
        'pR2': res.prsquared,
        'n':   len(rows),
    }
