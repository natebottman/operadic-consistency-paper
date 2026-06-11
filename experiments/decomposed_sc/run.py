"""Proper joint-sampled decomposed-SC.

For each (model, question) pair: sample K=10 independent (a1, a2) pairs at T=0.7.
That is, for each k in 1..K:
  - sample a1_k ← model(Q1) at T=0.7
  - sample a2_k ← model(Q2[A1 := a1_k]) at T=0.7
Then compute pairwise F1 over the 10 a2 values.

This addresses the LLM-feedback critique that the previous version
(holding a1 at greedy) underestimated decomposed-path variance.

Inputs: existing nonthinking_oc per-model JSONs (for question metadata + step1/step2 templates).
Outputs: data/raw/decomposed_sc_proper/{stem}/{dataset}.json
"""

import os
import json, os, re, sys, time, argparse, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ── API keys ──────────────────────────────────────────────────────────────────
TOGETHER_API_KEY  = os.environ.get("TOGETHER_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

TOGETHER_MODELS = {
    "cogito_671b":   "deepcogito/cogito-v2-1-671b",
    "deepseek_v3_1": "deepseek-ai/DeepSeek-V3.1",
    "gemma3n_4b":    "google/gemma-3n-E4B-it",
    "liquidai_24b":  "LiquidAI/LFM2-24B-A2B",
    "llama33_70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama8b":       "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "qwen3_235b":    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "qwen7b":        "Qwen/Qwen2.5-7B-Instruct-Turbo",
}
OPENROUTER_MODELS = {
    "gpt_4o_mini":      "openai/gpt-4o-mini",
    "gemini_2_5_flash": "google/gemini-2.5-flash",
    "llama_4_maverick": "meta-llama/llama-4-maverick",
    "gemini_2_5_pro":   "google/gemini-2.5-pro",
    "mistral_large_2":  "mistralai/mistral-large-2411",
}
ALL_MODELS = {**TOGETHER_MODELS, **OPENROUTER_MODELS}

EXP_DIR = os.environ.get('EXP_ROOT', os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')) + '/nonthinking_oc'
OUT_DIR = os.path.join(
    os.environ.get("EXP_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")),
    "decomposed_sc_proper",
)
DATASETS = ["hotpotqa", "musique", "strategyqa", "drop"]

K = 10
TEMP = 0.7
MAX_TOKENS = 64
MAX_CONCURRENT = 64

SYSTEM = (
    "You are a concise factual question-answering assistant. "
    "Answer with a short phrase or a few words. "
    "Do not explain your reasoning."
)

# Reuse dataset loaders from the non-thinking OC pipeline (same dataset prep).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nonthinking_oc'))
from run_together import (load_drop_passages, load_musique_passages, load_questions)

# ── Token-F1 ─────────────────────────────────────────────────────────────────
def normalize(t):
    t = str(t).lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def token_f1(p, g):
    pp, gg = normalize(p).split(), normalize(g).split()
    if not pp or not gg: return float(pp == gg)
    common = set(pp) & set(gg)
    if not common: return 0.0
    prec = sum(pp.count(t) for t in common) / len(pp)
    rec  = sum(gg.count(t) for t in common) / len(gg)
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

def pairwise_f1_mean(answers):
    if len(answers) < 2: return 0.0
    s, c = 0.0, 0
    for i in range(len(answers)):
        for j in range(i+1, len(answers)):
            s += token_f1(answers[i], answers[j]); c += 1
    return s/c if c else 0.0

def pairwise_agreement(answers):
    if len(answers) < 2: return 0.0
    s, c = 0.0, 0
    for i in range(len(answers)):
        for j in range(i+1, len(answers)):
            s += float(token_f1(answers[i], answers[j]) >= 0.5); c += 1
    return s/c if c else 0.0

def unanimous(answers):
    if len(answers) < 2: return False
    rep = answers[0]
    return all(token_f1(rep, a) >= 0.5 for a in answers)

# ── API clients ──────────────────────────────────────────────────────────────
_local = threading.local()
_sem = None

def _get_client(stem):
    if stem in TOGETHER_MODELS:
        if not hasattr(_local, "tog"):
            import together
            if not TOGETHER_API_KEY: raise RuntimeError("Set TOGETHER_API_KEY")
            _local.tog = together.Together(api_key=TOGETHER_API_KEY)
        return _local.tog
    else:
        if not hasattr(_local, "orc"):
            from openai import OpenAI
            if not OPENROUTER_API_KEY: raise RuntimeError("Set OPENROUTER_API_KEY")
            _local.orc = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        return _local.orc

def query(stem, model_id, prompt, retries=4):
    for attempt in range(retries):
        try:
            with _sem:
                client = _get_client(stem)
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role":"system","content":SYSTEM},
                        {"role":"user","content":prompt},
                    ],
                    max_tokens=MAX_TOKENS, temperature=TEMP,
                )
                raw = (r.choices[0].message.content or "").strip()
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw.split("\n")[0].strip()
        except Exception as e:
            err = str(e)
            if "404" in err: raise RuntimeError(f"Model unavailable: {err[:160]}")
            time.sleep(2 ** attempt)
    return ""

def make_prompt(question, passage=None):
    if passage:
        return f"Passage: {passage}\n\nQuestion: {question}"
    return question

# ── Per-question worker (joint sampling) ─────────────────────────────────────
def process_question(stem, model_id, q, passage):
    """Sample K independent (a1, a2) pairs at T=0.7."""
    step1_prompt = make_prompt(q["step1"], passage)
    a1_samples = []
    a2_samples = []
    for k in range(K):
        a1_k = query(stem, model_id, step1_prompt)
        if not a1_k.strip():
            a1_samples.append(""); a2_samples.append("")
            continue
        step2_filled = q["step2"].replace("[A1]", a1_k).replace("[a1]", a1_k)
        a2_k = query(stem, model_id, make_prompt(step2_filled, passage))
        a1_samples.append(a1_k)
        a2_samples.append(a2_k)
    return {
        "question_id": q["question_id"],
        "a1_samples": a1_samples,
        "a2_samples": a2_samples,
        "pairwise_k10": pairwise_f1_mean(a2_samples),
        "binary_pairwise_k10": pairwise_agreement(a2_samples),
        "unanimous_k10": unanimous(a2_samples),
    }

# ── Per-(model,dataset) runner ───────────────────────────────────────────────
def run_cell(stem, dataset, q_list):
    model_id = ALL_MODELS[stem]
    out_dir = os.path.join(OUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}.json")
    ckpt_path = out_path + ".ckpt"

    if os.path.exists(out_path):
        print(f"  [{stem}/{dataset}] already done, skipping")
        return None

    # We need step1, step2 templates per question. Get from existing exp data.
    src_path = os.path.join(EXP_DIR, dataset, f"{stem}.json")
    if not os.path.exists(src_path):
        print(f"  [{stem}/{dataset}] SKIP: no source data"); return None

    # Index questions from q_list (which already has passages)
    q_by_id = {q["question_id"]: q for q in q_list}

    # Resume
    results, done = [], set()
    if os.path.exists(ckpt_path):
        try:
            results = json.load(open(ckpt_path))
            done = {r["question_id"] for r in results}
            print(f"  [{stem}/{dataset}] resume: {len(results)} done")
        except Exception:
            results = []

    pending = [qid for qid in q_by_id if qid not in done]
    print(f"  [{stem}/{dataset}] {len(pending)} pending (of {len(q_by_id)})")

    lock = threading.Lock()
    def _ckpt():
        with open(ckpt_path, "w") as f: json.dump(results, f)

    def worker(qid):
        q = q_by_id[qid]
        try:
            rec = process_question(stem, model_id, q, q.get("passage"))
            with lock:
                results.append(rec)
                if len(results) % 50 == 0:
                    pf1 = sum(r["pairwise_k10"] for r in results)/len(results)
                    print(f"    [{stem}/{dataset}] {len(results)}/{len(pending)+len(done)}  pairwise_k10={pf1:.3f}")
                    _ckpt()
        except RuntimeError as e:
            with lock:
                if "unavailable" in str(e).lower():
                    print(f"  [{stem}/{dataset}] UNAVAILABLE: {e}"); raise

    if pending:
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(pending))) as pool:
            futs = [pool.submit(worker, qid) for qid in pending]
            for f in as_completed(futs):
                exc = f.exception()
                if exc and "unavailable" in str(exc).lower():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return {"error":"unavailable","stem":stem,"dataset":dataset}

    n = len(results)
    summary = {
        "model_stem": stem, "model_id": model_id, "dataset": dataset, "n": n,
        "decomposed_sc_proper_pairwise_k10": sum(r["pairwise_k10"] for r in results)/n if n else 0,
        "decomposed_sc_proper_binary_k10": sum(r["binary_pairwise_k10"] for r in results)/n if n else 0,
        "decomposed_sc_proper_unanimous_k10": sum(r["unanimous_k10"] for r in results)/n if n else 0,
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    print(f"  [{stem}/{dataset}] DONE n={n}  pairwise_k10={summary['decomposed_sc_proper_pairwise_k10']:.3f}")
    return summary

def main():
    global _sem
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS.keys()))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT)
    args = p.parse_args()
    _sem = threading.Semaphore(args.max_concurrent)

    print(f"decomposed_sc_proper (joint sampling) | {len(args.models)} models x {len(args.datasets)} datasets | K={K} T={TEMP}")
    print(f"  Output: {OUT_DIR}")

    drop_p = load_drop_passages() if "drop" in args.datasets else {}
    mus_p = load_musique_passages() if "musique" in args.datasets else {}

    qs = {}
    for ds in args.datasets:
        print(f"Loading {ds}...")
        qs[ds] = load_questions(ds,
            drop_passages=drop_p if ds=="drop" else None,
            musique_passages=mus_p if ds=="musique" else None,
        )

    tasks = [(stem, ds) for stem in args.models if stem in ALL_MODELS for ds in args.datasets]
    print(f"Running {len(tasks)} tasks")

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futs = {pool.submit(run_cell, s, ds, qs[ds]): (s, ds) for s, ds in tasks}
        for f in as_completed(futs):
            stem, ds = futs[f]
            try: f.result()
            except Exception as e:
                print(f"  [ERROR] {stem}/{ds}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
