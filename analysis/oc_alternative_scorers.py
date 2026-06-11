"""App. A (alternative OC scorers): compare 6 OC variants on per-question regression.

For each (model, dataset, question_id), correctness is fixed at the per-dataset
official scorer (`is_correct_official`). OC is computed under each variant:
  - F1:      score_official(direct, exp, dataset)        # paper's primary
  - NLI:     bidirectional DeBERTa-v3-large entailment   (continuous)
  - mini-S:  GPT-4o-mini, strict CoT prompt              (binary)
  - mini-C:  GPT-4o-mini, categorical CoT (binary at category=1)
  - L70-S:   Llama-3.3-70B, strict CoT
  - L70-C:   Llama-3.3-70B, categorical CoT

Reports per dataset:
  - Cross-model Pearson r between accuracy-rate and OC-rate (binarized at 0.5)
  - Per-question 3-way regression pseudo-R^2 (correct ~ OC + SC + (-H))
"""
from collections import defaultdict
import numpy as np
from scipy import stats

from analysis.scoring import score_official, is_correct_official, pairwise_official, cluster_entropy
from analysis.common import logit_cluster
from analysis.data_loader import load_nonthinking, load_baseline, NT_MODELS, NT_DATASETS

VARIANTS = ['F1', 'NLI', 'mini_S', 'mini_C', 'L70_S', 'L70_C']
DISPLAY  = {'F1': 'F1', 'NLI': 'NLI', 'mini_S': 'mini-S', 'mini_C': 'mini-C',
            'L70_S': 'L70-S', 'L70_C': 'L70-C'}


def build_rows():
    nli   = load_baseline('nli')
    judge = load_baseline('llm_judge_oc')

    by_dataset = defaultdict(list)
    per_model = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'corr': 0,
                                                          **{f'oc_{v}': 0 for v in VARIANTS}}))

    for r in load_nonthinking():
        ds, stem, qid = r['dataset'], r['model'], r['question_id']
        direct, exp = r['op_direct'], r['op_expansion']
        nv = r['nv_answers']
        if not direct.strip() or not exp.strip(): continue
        if len(nv) < 10: continue
        key = (stem, ds, qid)

        # Need NLI score
        nli_row = nli.get(key)
        if not nli_row or nli_row.get('nli_score') in (None, ''): continue
        # Need all four LLM judge variants
        j = judge.get(key)
        if not j: continue
        ocs = {'F1': score_official(direct, exp, ds),
               'NLI': float(nli_row['nli_score'])}
        valid = True
        for v in ('mini_S', 'mini_C', 'L70_S', 'L70_C'):
            val = j.get(v)
            if val in (None, ''): valid = False; break
            ocs[v] = float(val)
        if not valid: continue

        sc      = pairwise_official(nv[:10], ds)
        neg_H   = -cluster_entropy(nv[:10], dataset=ds)
        correct = int(is_correct_official(direct, r['ground_truth'], ds, r['aliases']))

        row = {'qid': qid, 'sc': sc, 'neg_H': neg_H, 'correct': correct}
        for v in VARIANTS: row[f'op_{v}'] = ocs[v]
        by_dataset[ds].append(row)

        c = per_model[stem][ds]
        c['n'] += 1
        if correct: c['corr'] += 1
        for v in VARIANTS:
            if ocs[v] >= 0.5: c[f'oc_{v}'] += 1

    return by_dataset, per_model


def cross_model_r(per_model):
    print('=' * 100)
    print(' Cross-model Pearson r (12 models): accuracy-rate vs OC-rate (binarized at 0.5)')
    print('=' * 100)
    print(f"  {'dataset':<12} " + ' '.join(f'{DISPLAY[v]:>8}' for v in VARIANTS))
    for ds in NT_DATASETS:
        accs = []; ocs = {v: [] for v in VARIANTS}
        for stem in NT_MODELS:
            cell = per_model[stem].get(ds)
            if not cell or not cell['n']: continue
            accs.append(cell['corr'] / cell['n'])
            for v in VARIANTS:
                ocs[v].append(cell[f'oc_{v}'] / cell['n'])
        rs = {v: stats.pearsonr(accs, ocs[v])[0] for v in VARIANTS}
        print(f"  {ds:<12} " + ' '.join(f'{rs[v]:>+8.3f}' for v in VARIANTS))


def per_question_pR2(by_dataset):
    print()
    print('=' * 100)
    print(' Per-question 3-way regression pseudo-R^2: correct ~ OC + SC + (-H), cluster-robust SE')
    print('=' * 100)
    print(f"  {'dataset':<12} " + ' '.join(f'{DISPLAY[v]:>8}' for v in VARIANTS))
    for ds in NT_DATASETS:
        rs_ = by_dataset[ds]
        qids = [r['qid'] for r in rs_]
        out = {}
        for v in VARIANTS:
            res = logit_cluster(rs_, [f'op_{v}', 'sc', 'neg_H'], cluster_groups=qids)
            out[v] = res['pR2'] if res else float('nan')
        print(f"  {ds:<12} " + ' '.join(f'{out[v]:>8.3f}' for v in VARIANTS))


def per_question_b_oc(by_dataset):
    print()
    print('=' * 100)
    print(' Per-question 3-way regression beta_OC: correct ~ OC + SC + (-H), cluster-robust SE')
    print('=' * 100)
    print(f"  {'dataset':<12} " + ' '.join(f'{DISPLAY[v]:>8}' for v in VARIANTS))
    for ds in NT_DATASETS:
        rs_ = by_dataset[ds]
        qids = [r['qid'] for r in rs_]
        out = {}
        for v in VARIANTS:
            res = logit_cluster(rs_, [f'op_{v}', 'sc', 'neg_H'], cluster_groups=qids)
            out[v] = res['b'][f'op_{v}'] if res else float('nan')
        print(f"  {ds:<12} " + ' '.join(f'{out[v]:>+8.2f}' for v in VARIANTS))


if __name__ == '__main__':
    by_dataset, per_model = build_rows()
    cross_model_r(per_model)
    per_question_pR2(by_dataset)
    per_question_b_oc(by_dataset)
