"""Common loaders for the per-question and baseline CSVs."""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / 'data'

NT_MODELS = [
    'cogito_671b', 'deepseek_v3_1', 'gemma3n_4b', 'liquidai_24b',
    'llama33_70b', 'llama8b', 'qwen3_235b', 'qwen7b',
    'gpt_4o_mini', 'gemini_2_5_flash', 'llama_4_maverick', 'mistral_large_2',
]
NT_DATASETS = ['hotpotqa', 'musique', 'strategyqa', 'drop']

T_MODELS = ['cogito-v2.1', 'deepseek-r1', 'glm-5.1', 'kimi-k2.6', 'minimax-m2.7']
T_DATASETS = ['musique', 'strategyqa', 'gsm8k', 'drop']


def load_nonthinking():
    """Yield dicts with op_direct, op_expansion, nv_answers (parsed list),
    cot_shorts (parsed list of CoT-SC short answers; possibly empty),
    ground_truth, aliases.

    cot_shorts is the K=10 list of LLM-extracted short answers from canonical
    chain-of-thought self-consistency (Wang et al. 2023; details in
    experiments/cot_sc/). Empty list when CoT-SC was not run on this row.
    """
    p = DATA / 'per_question' / 'nonthinking.csv'
    with open(p) as f:
        for r in csv.DictReader(f):
            r['nv_answers'] = json.loads(r['nv_answers'])
            r['aliases']    = json.loads(r['aliases'])
            r['cot_shorts'] = json.loads(r.get('cot_shorts', '[]'))
            yield r


def load_thinking():
    """Yield thinking-model rows with parsed sc_k3, sc_k10, aliases."""
    p = DATA / 'per_question' / 'thinking.csv'
    with open(p) as f:
        for r in csv.DictReader(f):
            r['sc_k3']   = json.loads(r['sc_k3'])
            r['sc_k10']  = json.loads(r['sc_k10'])
            r['aliases'] = json.loads(r['aliases'])
            yield r


def load_baseline(name):
    """Load a baseline CSV (p_true, decomposed_sc, nli, prm_skywork, prm_rlhflow).

    Returns dict keyed by (model, dataset, question_id) -> remaining columns.
    """
    p = DATA / 'baselines' / f'{name}.csv'
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            key = (r['model'], r['dataset'], r['question_id'])
            extras = {k: v for k, v in r.items() if k not in ('model', 'dataset', 'question_id')}
            out[key] = extras
    return out
