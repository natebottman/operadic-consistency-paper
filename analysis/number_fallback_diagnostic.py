"""App. A: number-fallback diagnostic.

Reproduces:
- Per-dataset cross-model Pearson r between accuracy and OC-rate, both
  with and without the number-extraction fallback.
- Hit-rate of the number-extraction path (fraction of pairs where both
  prediction and the comparison string contain a parseable number).
"""
from collections import defaultdict
import numpy as np
from scipy import stats

from analysis.scoring import f1, normalize, first_number
from analysis.data_loader import load_nonthinking, NT_MODELS, NT_DATASETS


def score_with_optional_fallback(p, g, use_num):
    if not use_num:
        return f1(p, g)
    pn = first_number(p)
    gn = first_number(g)
    if pn is not None and gn is not None:
        return 1.0 if pn == gn else 0.0
    return f1(p, g)


def main():
    rows = list(load_nonthinking())

    # Per-dataset cross-model r under each protocol.
    print('=' * 80)
    print(' Cross-model Pearson r between accuracy and OC-rate (App. A)')
    print('=' * 80)
    print()
    for ds in NT_DATASETS:
        print(f'=== {ds} ===')
        for use_num in [False, True]:
            accs, ocs = [], []
            num_hit_acc = num_hit_oc = total = 0
            for stem in NT_MODELS:
                cell = [r for r in rows if r['model'] == stem and r['dataset'] == ds]
                if not cell:
                    continue
                n = a = o = 0
                for r in cell:
                    direct = r['op_direct']
                    exp    = r['op_expansion']
                    gold   = r['ground_truth']
                    aliases = r['aliases']
                    if not direct.strip():
                        continue
                    n += 1; total += 1
                    max_score = max(
                        score_with_optional_fallback(direct, c, use_num)
                        for c in [gold] + list(aliases or [])
                    )
                    if max_score >= 0.5:
                        a += 1
                    oc_score = score_with_optional_fallback(direct, exp, use_num) if exp.strip() else 0
                    if oc_score >= 0.5:
                        o += 1
                    if use_num:
                        if first_number(direct) is not None and first_number(gold) is not None:
                            num_hit_acc += 1
                        if first_number(direct) is not None and first_number(exp) is not None:
                            num_hit_oc += 1
                accs.append(a / n)
                ocs.append(o / n)
            r, p = stats.pearsonr(accs, ocs)
            label = 'F1 only      ' if not use_num else 'num-fallback '
            if use_num:
                print(f'  {label} r={r:+.3f} p={p:.4f}  '
                      f'(number-path hit {num_hit_acc/total:.1%} for acc, {num_hit_oc/total:.1%} for OC)')
            else:
                print(f'  {label} r={r:+.3f} p={p:.4f}')
        print()


if __name__ == '__main__':
    main()
