"""Build the per-question and baseline CSVs in data/ from raw JSON outputs in data/raw/.

Expected layout under data/raw/ (configured by RAW_ROOT env var):

    data/raw/
    ├── nonthinking_oc/{dataset}/{model}.json              # nonthinking OC + SC
    ├── exp2_thinking_complementarity/{model}/          # thinking OC + SC
    │       cot_pilot_{dataset}.json
    │       sc_{dataset}.json
    │       k10/sc_{dataset}.json
    ├── p_true_nonthinking/{model}/{dataset}.json       # P(True) for Together models
    ├── p_true_openrouter/{model}/{dataset}.json        # P(True) for OpenRouter models
    ├── decomposed_sc_proper/{model}/{dataset}.json
    ├── nli_scores/{model}/{dataset}.json
    ├── prm_outputs/{model}__{dataset}.jsonl            # Skywork PRM
    └── rlhflow_outputs/{model}__{dataset}.jsonl        # RLHFlow PRM

Outputs: data/per_question/{nonthinking,thinking}.csv
         data/baselines/{p_true,decomposed_sc,nli,prm_skywork,prm_rlhflow,llm_judge_oc}.csv

Run from repo root: python scripts/build_csvs.py
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get('RAW_ROOT', REPO_ROOT / 'data' / 'raw'))
OUT = REPO_ROOT / 'data'

NT_MODELS = [
    'cogito_671b', 'deepseek_v3_1', 'gemma3n_4b', 'liquidai_24b',
    'llama33_70b', 'llama8b', 'qwen3_235b', 'qwen7b',
    'gpt_4o_mini', 'gemini_2_5_flash', 'llama_4_maverick', 'mistral_large_2',
]
T_MODELS = ['cogito-v2.1', 'deepseek-r1', 'glm-5.1', 'kimi-k2.6', 'minimax-m2.7']
NT_DATASETS = ['hotpotqa', 'musique', 'strategyqa', 'drop']
T_DATASETS = ['musique', 'strategyqa', 'gsm8k', 'drop']

# DROP aliases: ship as data/drop_aliases.json so the reproduction pipeline
# does not require the raw DROP dataset. Built once from the official DROP
# dev split.
DROP_ALIASES = json.load(open(OUT / 'drop_aliases.json'))


def get_aliases(dataset, qid):
    return DROP_ALIASES.get(qid, []) if dataset == 'drop' else []


def build_nonthinking():
    rows = []
    for stem in NT_MODELS:
        for ds in NT_DATASETS:
            p = RAW / 'nonthinking_oc' / ds / f'{stem}.json'
            if not p.exists():
                print(f'  skip (missing): {p.relative_to(RAW)}', file=sys.stderr)
                continue
            d = json.loads(p.read_text())
            for r in d['results']:
                if r.get('ground_truth') is None:
                    continue
                rows.append({
                    'model': stem,
                    'dataset': ds,
                    'question_id': r['question_id'],
                    'op_direct':    r.get('op_direct') or '',
                    'op_expansion': r.get('op_expansion') or '',
                    'nv_answers':   json.dumps(r.get('nv_answers') or []),
                    'ground_truth': r['ground_truth'],
                    'aliases':      json.dumps(get_aliases(ds, r['question_id'])),
                })
    out_path = OUT / 'per_question' / 'nonthinking.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_thinking():
    rows = []
    for stem in T_MODELS:
        for ds in T_DATASETS:
            cot_p = RAW / 'exp2_thinking_complementarity' / stem / f'cot_pilot_{ds}.json'
            sc3_p = RAW / 'exp2_thinking_complementarity' / stem / f'sc_{ds}.json'
            sc10_p = RAW / 'exp2_thinking_complementarity' / stem / 'k10' / f'sc_{ds}.json'
            if not cot_p.exists():
                print(f'  skip (missing cot): {cot_p.relative_to(RAW)}', file=sys.stderr)
                continue
            cot = json.loads(cot_p.read_text())['results']
            sc3 = json.loads(sc3_p.read_text())['results'] if sc3_p.exists() else []
            sc10 = json.loads(sc10_p.read_text())['results'] if sc10_p.exists() else []
            sc3_by = {r['question_id']: r for r in sc3}
            sc10_by = {r['question_id']: r for r in sc10}
            for r in cot:
                if r.get('error'):
                    continue
                if not (r.get('extraction') or {}).get('factorable'):
                    continue
                if r.get('ground_truth') is None:
                    continue
                qid = r['question_id']
                pt = r.get('pilot_think') or {}
                ext = r.get('extraction') or {}
                s3 = sc3_by.get(qid, {})
                s10 = sc10_by.get(qid, {})
                # Aliases on cot_pilot rows are populated directly by the
                # extraction pipeline, so prefer them over the per-dataset
                # alias map. Falls back to DROP_ALIASES for legacy DROP rows.
                aliases = r.get('ground_truth_aliases') or get_aliases(ds, qid)
                rows.append({
                    'model': stem,
                    'dataset': ds,
                    'question_id': qid,
                    'factorable':   True,
                    'q1':            ext.get('q1') or '',
                    'a1':            ext.get('a1') or '',
                    'q2':            ext.get('q2') or '',
                    'decomp_answer': pt.get('answer') or '',
                    'direct_answer': r.get('stated_answer') or '',
                    # Raw SC samples; pairwise F1 / entropy is computed
                    # downstream by analysis/scoring.py (single source of truth).
                    'sc_k3':  json.dumps(s3.get('short_answers') or []),
                    'sc_k10': json.dumps(s10.get('short_answers') or []),
                    'ground_truth': r['ground_truth'],
                    'aliases':      json.dumps(aliases),
                })
    out_path = OUT / 'per_question' / 'thinking.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_p_true():
    rows = []
    for source in ['p_true_nonthinking', 'p_true_openrouter']:
        root = RAW / source
        if not root.exists():
            continue
        for stem_dir in sorted(root.iterdir()):
            stem = stem_dir.name
            for ds in NT_DATASETS:
                p = stem_dir / f'{ds}.json'
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                for r in d['results']:
                    pyes = r.get('p_true', r.get('p_yes'))
                    if pyes is None:
                        continue
                    rows.append({
                        'model': stem, 'dataset': ds,
                        'question_id': r['question_id'],
                        'p_true': float(pyes),
                    })
    out_path = OUT / 'baselines' / 'p_true.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_decomposed_sc():
    rows = []
    root = RAW / 'decomposed_sc_proper'
    if not root.exists():
        print(f'  skip (missing dir): {root.relative_to(RAW)}', file=sys.stderr)
        return
    for stem_dir in sorted(root.iterdir()):
        stem = stem_dir.name
        for ds in NT_DATASETS:
            p = stem_dir / f'{ds}.json'
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for r in d['results']:
                if r.get('pairwise_k10') is None:
                    continue
                rows.append({
                    'model': stem, 'dataset': ds,
                    'question_id': r['question_id'],
                    'dsc_proper': float(r['pairwise_k10']),
                })
    out_path = OUT / 'baselines' / 'decomposed_sc.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_nli():
    rows = []
    root = RAW / 'nli_scores'
    if not root.exists():
        print(f'  skip (missing dir): {root.relative_to(RAW)}', file=sys.stderr)
        return
    for stem_dir in sorted(root.iterdir()):
        stem = stem_dir.name
        for ds in NT_DATASETS:
            p = stem_dir / f'{ds}.json'
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for r in d['results']:
                row = {
                    'model': stem, 'dataset': ds,
                    'question_id': r['question_id'],
                    'nli_score':      r.get('nli_score'),
                    'nli_consistent': int(r['nli_consistent']) if 'nli_consistent' in r else None,
                }
                if row['nli_score'] is None:
                    continue
                rows.append(row)
    out_path = OUT / 'baselines' / 'nli.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_prm(source_dir, out_name):
    rows = []
    root = RAW / source_dir
    if not root.exists():
        print(f'  skip (missing dir): {root.relative_to(RAW)}', file=sys.stderr)
        return
    for f in sorted(root.glob('*.jsonl')):
        # filename pattern: {model}__{dataset}.jsonl
        stem = f.stem
        if '__' not in stem:
            continue
        model, dataset = stem.rsplit('__', 1)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('prm_min') is None:
                continue
            rows.append({
                'model':       model,
                'dataset':     dataset,
                'question_id': r['question_id'],
                'prm_min':     float(r['prm_min']),
                'prm_mean':    float(r.get('prm_mean')) if r.get('prm_mean')  is not None else None,
                'prm_last':    float(r.get('prm_last')) if r.get('prm_last')  is not None else None,
            })
    out_path = OUT / 'baselines' / f'{out_name}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def build_llm_judge_oc():
    """Combine the four LLM-judge variants into one CSV.

    Each variant has its own .jsonl in data/raw/llm_judge_oc/. Each row has
    'oc_verdict' parsed by the inference script: 'yes'/'no' for strict
    prompts, '1'..'5' for categorical, or None / 'unparsed:...' otherwise.

    We binarize:
      strict variants:      1.0 if 'yes' else 0.0 if 'no' else NaN
      categorical variants: 1.0 if '1' else 0.0 if '2'..'5' else NaN
    """
    variants = {
        'mini_S': ('mini_strict.jsonl',           'strict'),
        'mini_C': ('mini_categorical.jsonl',      'categorical'),
        'L70_S':  ('llama3370b_strict.jsonl',     'strict'),
        'L70_C':  ('llama3370b_categorical.jsonl', 'categorical'),
    }
    root = RAW / 'llm_judge_oc'
    if not root.exists():
        print(f'  skip (missing dir): {root.relative_to(RAW)}', file=sys.stderr)
        return

    def parse(verdict, kind):
        if verdict is None: return None
        if kind == 'strict':
            return 1.0 if verdict == 'yes' else (0.0 if verdict == 'no' else None)
        if kind == 'categorical':
            return 1.0 if verdict == '1' else (0.0 if verdict in ('2','3','4','5') else None)
        return None

    # Collect per-key dict of variant -> score
    by_key = {}
    for col, (fname, kind) in variants.items():
        p = root / fname
        if not p.exists():
            print(f'  skip (missing): {p.relative_to(RAW)}', file=sys.stderr)
            continue
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            key = (r['model'], r['dataset'], r['question_id'])
            by_key.setdefault(key, {col: None for col in variants})[col] = parse(r.get('oc_verdict'), kind)

    rows = []
    for (stem, ds, qid), scores in by_key.items():
        rows.append({'model': stem, 'dataset': ds, 'question_id': qid, **scores})

    out_path = OUT / 'baselines' / 'llm_judge_oc.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    print(f'  wrote {len(rows):,} rows -> {out_path.relative_to(REPO_ROOT)}')


def write_csv(path, rows):
    if not rows:
        print(f'  WARNING: no rows to write to {path}', file=sys.stderr)
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    print(f'RAW_ROOT={RAW}')
    print(f'Building per-question CSVs...')
    print(' [1/2] nonthinking')
    build_nonthinking()
    print(' [2/2] thinking')
    build_thinking()
    print(f'Building baseline CSVs...')
    print(' [1/6] p_true')
    build_p_true()
    print(' [2/6] decomposed_sc')
    build_decomposed_sc()
    print(' [3/6] nli')
    build_nli()
    print(' [4/6] prm_skywork')
    build_prm('prm_outputs', 'prm_skywork')
    print(' [5/6] prm_rlhflow')
    build_prm('rlhflow_outputs', 'prm_rlhflow')
    print(' [6/6] llm_judge_oc')
    build_llm_judge_oc()
    print('Done.')
