"""Full canonical CoT self-consistency run.

Runs CoT-SC across all 12 non-thinking models × 4 datasets × full question
sets × K=10. Per-model provider routing matches the original OC data
(Together for 8 models, OpenRouter for 4) so CoT-SC and OC live on the same
provider/variant per model.

Output: data/raw/exp_cot_sc/{dataset}/{model_stem}.json
Checkpointed per cell via .json.ckpt; resume by re-running.

Usage:
    set -a && source env/.env && set +a
    python experiments/cot_sc/run_full.py
    # or restrict scope:
    python experiments/cot_sc/run_full.py --models qwen7b llama8b --datasets hotpotqa
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import together
from openai import OpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from analysis.scoring import pairwise_official  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────

TOGETHER_KEY   = os.environ.get("TOGETHER_API_KEY")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
if not TOGETHER_KEY or not OPENROUTER_KEY:
    sys.exit("ERROR: set TOGETHER_API_KEY and OPENROUTER_API_KEY in env")

REF_BASE = os.environ.get("LEGACY_OC_ASSETS")
if not REF_BASE:
    sys.exit("ERROR: set LEGACY_OC_ASSETS to the path containing the "
             "nonthinking OC raw outputs (the dir holding results/{dataset}/{model}.json files)")
RESULTS_DIR = os.path.join(REF_BASE, "results")
OUT_ROOT    = os.path.join(REPO_ROOT, "data", "raw", "exp_cot_sc")

# Per-model provider routing -- matches the original OC data per the
# "option B" methodological choice.
MODELS = {
    # Together-served (8)
    "cogito_671b":     ("together",   "deepcogito/cogito-v2-1-671b"),
    "deepseek_v3_1":   ("together",   "deepseek-ai/DeepSeek-V3.1"),
    "gemma3n_4b":      ("together",   "google/gemma-3n-E4B-it"),
    "liquidai_24b":    ("together",   "LiquidAI/LFM2-24B-A2B"),
    "llama33_70b":     ("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "llama8b":         ("together",   "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    "qwen3_235b":      ("together",   "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    "qwen7b":          ("together",   "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    # OpenRouter-served (4)
    "gemini_2_5_flash": ("openrouter", "google/gemini-2.5-flash"),
    "gpt_4o_mini":      ("openrouter", "openai/gpt-4o-mini"),
    "llama_4_maverick": ("openrouter", "meta-llama/llama-4-maverick"),
    "mistral_large_2":  ("openrouter", "mistralai/mistral-large-2411"),
}

DATASETS = ["hotpotqa", "musique", "strategyqa", "drop"]
DATASET_REF = {
    "hotpotqa":   os.path.join(RESULTS_DIR, "8b.json"),
    "musique":    os.path.join(RESULTS_DIR, "musique",    "deepseek_v3_1.json"),
    "strategyqa": os.path.join(RESULTS_DIR, "strategyqa", "deepseek_v3_1.json"),
    "drop":       os.path.join(RESULTS_DIR, "drop",       "deepseek_v3_1.json"),
}

K_SAMPLES      = 10
COT_MAX_TOKENS = 512
COT_TEMP       = 0.7
CONCURRENCY    = 8

EXTRACT_MODEL_TOGETHER = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EXTRACT_MODEL_OR       = "meta-llama/llama-3.3-70b-instruct"

SYSTEM_COT = (
    "Answer the question accurately. "
    "Think step by step, then state the final answer."
)
EXTRACT_PROMPT = """\
Question: {question}

Verbose answer:
{verbose}

Extract ONLY the direct answer to the question in 5 words or fewer.
Output the answer only -- no explanation, no punctuation except what's part of the answer."""


# ── Clients ──────────────────────────────────────────────────────────────────

_together = together.Together(api_key=TOGETHER_KEY)
_or       = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")


def _chat(provider, model, messages, max_tokens, temperature, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            if provider == "together":
                r = _together.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
            else:
                r = _or.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
            return (r.choices[0].message.content or "").strip(), None
        except Exception as e:
            last_err = str(e)[:200]
            # Only abort the whole cell on TRUE model-unavailability errors,
            # not per-question content/format rejections. Together signals
            # unavailability via "non-serverless model"; 404 is always a hard
            # routing failure. Per-question 400s (e.g., OpenRouter's
            # "Provider returned error" from a single bad prompt) should fail
            # the sample and let the cell continue.
            if re.search(r"Error code: 404\b", last_err) \
                    or "non-serverless model" in last_err.lower() \
                    or "not found" in last_err.lower():
                raise RuntimeError(f"Model unavailable: {last_err[:120]}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return "", last_err


def sample_chain(provider, model_id, prompt):
    text, err = _chat(provider, model_id,
                      [{"role": "system", "content": SYSTEM_COT},
                       {"role": "user",   "content": prompt}],
                      max_tokens=COT_MAX_TOKENS, temperature=COT_TEMP)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return {"chain": text, "error": err, "chain_tokens": len(text.split())}


def extract_short(provider, question, verbose):
    if not verbose.strip():
        return ""
    model = EXTRACT_MODEL_TOGETHER if provider == "together" else EXTRACT_MODEL_OR
    prompt = EXTRACT_PROMPT.format(question=question, verbose=verbose[:1500])
    text, _ = _chat(provider, model,
                    [{"role": "user", "content": prompt}],
                    max_tokens=32, temperature=0.0, retries=3)
    return text.strip()


# ── Dataset loading ──────────────────────────────────────────────────────────

def _extract_uuid(qid):
    m = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        str(qid), re.I,
    )
    return m.group(1) if m else None


def load_musique_passages():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="dgslibisey/MuSiQue",
                               filename="musique_ans_v1.0_dev.jsonl",
                               repo_type="dataset")
        result = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                paras = [p for p in row["paragraphs"] if p["is_supporting"]]
                ctx = "\n\n".join(f'[{p["title"]}] {p["paragraph_text"]}'
                                  for p in paras)
                result[row["id"]] = ctx
        return result
    except Exception as e:
        print(f"  [warn] MuSiQue passages unavailable: {e}", flush=True)
        return {}


def load_drop_passages():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="ucinlp/drop",
                               filename="data/validation-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path, columns=["query_id", "passage"])
        return dict(zip(df["query_id"], df["passage"]))
    except Exception as e:
        print(f"  [warn] DROP passages unavailable: {e}", flush=True)
        return {}


def load_questions(dataset, musique_passages=None, drop_passages=None):
    import random as _random
    with open(DATASET_REF[dataset]) as f:
        data = json.load(f)

    qs = []
    for r in data["results"]:
        if "step1" not in r or "step2" not in r:
            continue
        qid = r.get("question_id")
        q = {"question_id": qid, "question": r["question"],
             "ground_truth": r.get("ground_truth"),
             "passage": None}
        if dataset == "musique" and musique_passages and qid in musique_passages:
            q["passage"] = musique_passages[qid]
        elif dataset == "drop" and drop_passages:
            uuid = _extract_uuid(qid)
            if uuid and uuid in drop_passages:
                q["passage"] = drop_passages[uuid]
        qs.append(q)

    # Mirror existing run_together.py: MuSiQue gets random-sampled to 500
    # for parity with the cohort in the per-question CSV.
    if dataset == "musique" and len(qs) > 500:
        qs = _random.Random(42).sample(qs, 500)
    return qs


def make_prompt(question, passage):
    if passage:
        return f"Passage: {passage}\n\nQuestion: {question}"
    return question


# ── Per-cell runner ──────────────────────────────────────────────────────────

def run_cell(model_stem, provider, model_id, dataset, questions):
    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_stem}.json")
    ckpt     = out_path + ".ckpt"

    if os.path.exists(out_path):
        print(f"  [{model_stem}/{dataset}] already done, skipping", flush=True)
        return None

    results, done_ids = [], set()
    if os.path.exists(ckpt):
        try:
            results = json.load(open(ckpt))
            done_ids = {r["question_id"] for r in results}
            print(f"  [{model_stem}/{dataset}] resuming from ckpt "
                  f"({len(results)} done)", flush=True)
        except Exception:
            results = []
    todo = [q for q in questions if q["question_id"] not in done_ids]

    lock = threading.Lock()
    aborted_msg = [None]

    def _persist():
        with open(ckpt, "w") as f:
            json.dump(results, f)

    def process(q):
        if aborted_msg[0]:
            return
        prompt = make_prompt(q["question"], q.get("passage"))
        chains = []
        for _ in range(K_SAMPLES):
            try:
                chains.append(sample_chain(provider, model_id, prompt))
            except RuntimeError as e:
                with lock:
                    aborted_msg[0] = str(e)
                return

        # Extract short answers (parallel within this question)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            shorts = list(pool.map(
                lambda c: extract_short(provider, q["question"], c["chain"]),
                chains))

        chain_errors = [c["error"] for c in chains if c["error"]]
        nonempty = [s for s in shorts if s.strip()]
        if len(nonempty) >= 2:
            agree_k10 = pairwise_official(nonempty, dataset)
            agree_k3  = pairwise_official([s for s in shorts[:3] if s.strip()], dataset) \
                        if sum(1 for s in shorts[:3] if s.strip()) >= 2 else 0.0
        else:
            agree_k10 = agree_k3 = 0.0

        rec = {
            "question_id":         q["question_id"],
            "question":            q["question"],
            "cot_chains":          [c["chain"] for c in chains],
            "cot_chain_tokens":    [c["chain_tokens"] for c in chains],
            "cot_shorts":          shorts,
            "cot_pairwise_k3":     agree_k3,
            "cot_pairwise_k10":    agree_k10,
            "n_extract_fail":      K_SAMPLES - len(nonempty),
            "n_chain_fail":        len(chain_errors),
            "chain_errors":        chain_errors,
            "ground_truth":        q.get("ground_truth"),
        }
        with lock:
            results.append(rec)
            n = len(results)
            if n % 25 == 0:
                _persist()
                k10 = sum(r["cot_pairwise_k10"] for r in results) / n
                print(f"  [{model_stem}/{dataset}] {n}/{len(questions)} "
                      f"cot_k10={k10:.3f}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futs = [pool.submit(process, q) for q in todo]
            for fut in as_completed(futs):
                if fut.exception():
                    msg = str(fut.exception())[:160]
                    print(f"  [{model_stem}/{dataset}] task error: {msg}",
                          flush=True)
                if aborted_msg[0]:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    if aborted_msg[0]:
        print(f"  [{model_stem}/{dataset}] ABORTED: {aborted_msg[0]}",
              flush=True)
        return None

    n = len(results)
    if n == 0:
        return None

    summary = {
        "model_stem":        model_stem,
        "model_id":          model_id,
        "provider":          provider,
        "dataset":           dataset,
        "n":                 n,
        "k_max":             K_SAMPLES,
        "cot_pairwise_k3":   sum(r["cot_pairwise_k3"]  for r in results) / n,
        "cot_pairwise_k10":  sum(r["cot_pairwise_k10"] for r in results) / n,
        "n_chain_fail":      sum(r["n_chain_fail"]    for r in results),
        "n_extract_fail":    sum(r["n_extract_fail"]  for r in results),
        "mean_chain_tokens": sum(sum(r["cot_chain_tokens"]) for r in results)
                             / max(1, n * K_SAMPLES),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"  [{model_stem}/{dataset}] DONE n={n}  "
          f"cot_k10={summary['cot_pairwise_k10']:.3f}  "
          f"chain_fail={summary['n_chain_fail']}  "
          f"extract_fail={summary['n_extract_fail']}  "
          f"mean_tokens={summary['mean_chain_tokens']:.0f}  "
          f"-> {out_path}", flush=True)
    return summary


# ── Driver ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"\n=== CoT-SC FULL RUN ===", flush=True)
    print(f"models:    {args.models}", flush=True)
    print(f"datasets:  {args.datasets}", flush=True)
    print(f"K_max:     {K_SAMPLES}", flush=True)
    print(f"CoT:       {COT_MAX_TOKENS}-tok budget, T={COT_TEMP}", flush=True)
    print(f"output:    {OUT_ROOT}/<dataset>/<model>.json", flush=True)
    t0 = time.time()

    print("\nLoading passages...", flush=True)
    musique_p = load_musique_passages() if "musique" in args.datasets else {}
    drop_p    = load_drop_passages()    if "drop"    in args.datasets else {}
    print(f"  musique: {len(musique_p)} passages, drop: {len(drop_p)} passages",
          flush=True)

    for ds in args.datasets:
        qs = load_questions(ds, musique_p, drop_p)
        print(f"\n--- dataset: {ds} ({len(qs)} questions) ---", flush=True)
        for stem in args.models:
            provider, model_id = MODELS[stem]
            t1 = time.time()
            run_cell(stem, provider, model_id, ds, qs)
            elapsed_total = (time.time() - t0) / 60
            print(f"      ({time.time()-t1:.0f}s)  "
                  f"[total elapsed: {elapsed_total:.0f}min]", flush=True)

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
