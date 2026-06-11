"""Non-thinking OC + SC inference for the eight Together-served models.

Produces, for each (model, question) pair: greedy direct answer, greedy
step-1 answer, greedy step-2 (decomposed) answer, plus K=10 naive samples
at T=0.7. Both K=3 and K=10 SC stats are derived from the same K=10 draw.

Models: cogito_671b, deepseek_v3_1, gemma3n_4b, liquidai_24b,
        llama33_70b, llama8b, qwen3_235b, qwen7b.

Output: data/raw/nonthinking_oc/{dataset}/{model_stem}.json (override via EXP_ROOT).
"""

import os
import json, logging, os, re, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading
import together

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("TOGETHER_API_KEY")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY in env (see env/.env.example).")
# BASE points at a local cache of dataset reference files (gold + alias maps).
BASE    = os.environ.get('LEGACY_OC_ASSETS', os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'legacy_oc_assets'))
RESULTS = os.path.join(BASE, "results")
# Outputs land under data/raw/nonthinking_oc/{dataset}/{model}.json.
OUT_DIR = os.environ.get('EXP_ROOT', os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')) + '/nonthinking_oc'

K_MAX  = 10   # naive samples drawn per question
K3     = 3    # sub-sample for K=3 stats (first K3 of K_MAX)
TEMP_N = 0.7
TEMP_O = 0.0

MAX_CONCURRENT    = 20
DEFAULT_MAX_TOKENS = 64
MODEL_MAX_TOKENS  = {}

SYSTEM = (
    "You are a concise factual question-answering assistant. "
    "Answer with a short phrase or a few words. "
    "Do not explain your reasoning."
)

# ── Models ────────────────────────────────────────────────────────────────────

MODELS = {
    "cogito_671b":   "deepcogito/cogito-v2-1-671b",
    "deepseek_v3_1": "deepseek-ai/DeepSeek-V3.1",
    "gemma3n_4b":    "google/gemma-3n-E4B-it",
    "liquidai_24b":  "LiquidAI/LFM2-24B-A2B",
    "llama33_70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama8b":       "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "qwen3_235b":    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "qwen7b":        "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "gpt_oss_20b":   "openai/gpt-oss-20b",
    "gpt_oss_120b":  "openai/gpt-oss-120b",
}
# gpt-oss models include hidden reasoning tokens; need a higher token budget so the
# user-visible answer span isn't truncated.
MODEL_MAX_TOKENS_OVERRIDE = {
    "gpt_oss_20b":  256,
    "gpt_oss_120b": 256,
}

# ── Datasets ──────────────────────────────────────────────────────────────────

DATASET_REF = {
    "hotpotqa":   os.path.join(RESULTS, "8b.json"),
    "musique":    os.path.join(RESULTS, "musique",    "deepseek_v3_1.json"),
    "strategyqa": os.path.join(RESULTS, "strategyqa", "deepseek_v3_1.json"),
    "drop":       os.path.join(RESULTS, "drop",       "deepseek_v3_1.json"),
}

# ── Token F1 ─────────────────────────────────────────────────────────────────

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = sum(p.count(t) for t in common) / len(p)
    rec  = sum(g.count(t) for t in common) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def answers_agree(a, b, thresh=0.5):
    if not a.strip() or not b.strip():
        return False
    return token_f1(a, b) >= thresh

def pairwise_agreement(answers):
    pairs = [(i, j) for i in range(len(answers)) for j in range(i+1, len(answers))]
    if not pairs:
        return 0.0
    return sum(answers_agree(answers[i], answers[j]) for i, j in pairs) / len(pairs)

def majority_vote(answers):
    k = len(answers)
    scores = [
        sum(answers_agree(answers[i], answers[j]) for j in range(k) if j != i)
        for i in range(k)
    ]
    return answers[scores.index(max(scores))]

def is_correct(pred, gold, aliases=None):
    if not pred.strip() or gold is None:
        return False
    candidates = [gold] + (aliases or [])
    return max(token_f1(pred, c) for c in candidates) >= 0.5

# ── API ───────────────────────────────────────────────────────────────────────

_sem = None

def _get_client():
    if not hasattr(_get_client, '_local'):
        _get_client._local = threading.local()
    if not hasattr(_get_client._local, 'client'):
        _get_client._local.client = together.Together(api_key=API_KEY)
    return _get_client._local.client

def query(model_id, prompt, temperature, max_tokens=DEFAULT_MAX_TOKENS, retries=4):
    for attempt in range(retries):
        try:
            with _sem:
                client = _get_client()
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = (r.choices[0].message.content or "").strip()
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                return raw.split("\n")[0].strip()
        except Exception as e:
            err = str(e)
            if "400" in err or "404" in err:
                raise RuntimeError(f"Model unavailable: {err[:120]}")
            delay = 2 ** attempt
            print(f"  [warn] attempt {attempt+1} failed ({err[:60]}) -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"All {retries} attempts failed")

# ── Dataset loading ───────────────────────────────────────────────────────────

def _extract_uuid(qid):
    m = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        qid, re.I,
    )
    return m.group(1) if m else None

def load_musique_passages():
    try:
        from huggingface_hub import hf_hub_download
        print("  Loading MuSiQue passages...")
        path = hf_hub_download(
            repo_id="dgslibisey/MuSiQue",
            filename="musique_ans_v1.0_dev.jsonl",
            repo_type="dataset",
        )
        result = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                paras = [p for p in row["paragraphs"] if p["is_supporting"]]
                ctx = "\n\n".join(f'[{p["title"]}] {p["paragraph_text"]}' for p in paras)
                result[row["id"]] = ctx
        print(f"  Loaded {len(result)} MuSiQue passages")
        return result
    except Exception as e:
        print(f"  [warn] Could not load MuSiQue passages: {e}")
        return {}

def load_drop_passages():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        print("  Loading DROP passages...")
        path = hf_hub_download(
            repo_id="ucinlp/drop",
            filename="data/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path, columns=["query_id", "passage"])
        result = dict(zip(df["query_id"], df["passage"]))
        print(f"  Loaded {len(result)} DROP passages")
        return result
    except Exception as e:
        print(f"  [warn] Could not load DROP passages: {e}")
        return {}

def load_hotpotqa_gold():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="hotpot_qa",
            filename="fullwiki/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path, columns=["id", "answer"])
        return dict(zip(df["id"], df["answer"]))
    except Exception as e:
        print(f"  [warn] Could not load HotpotQA gold: {e}")
        return {}

def load_questions(dataset_name, drop_passages=None, musique_passages=None):
    ref_path = DATASET_REF[dataset_name]
    with open(ref_path) as f:
        data = json.load(f)

    skipped = 0
    questions = []
    for r in data["results"]:
        if "step1" not in r or "step2" not in r:
            skipped += 1
            continue
        q = {
            "question_id":          r.get("question_id"),
            "question":             r["question"],
            "step1":                r["step1"],
            "step2":                r["step2"],
            "ground_truth":         r.get("ground_truth"),
            "ground_truth_aliases": r.get("ground_truth_aliases", []) or [],
            "passage":              None,
        }
        if dataset_name == "drop" and drop_passages:
            uuid = _extract_uuid(r.get("question_id", ""))
            if uuid and uuid in drop_passages:
                q["passage"] = drop_passages[uuid]
        if dataset_name == "musique" and musique_passages:
            qid = r.get("question_id", "")
            if qid in musique_passages:
                q["passage"] = musique_passages[qid]
        questions.append(q)

    if skipped:
        print(f"  [{dataset_name}] skipped {skipped} error records")

    if dataset_name == "musique" and len(questions) > 500:
        import random as _random
        total = len(questions)
        questions = _random.Random(42).sample(questions, 500)
        print(f"  [musique] sampled 500/{total} questions")

    if dataset_name == "drop":
        n_with = sum(1 for q in questions if q["passage"])
        print(f"  [drop] {n_with}/{len(questions)} questions have passages")
    if dataset_name == "musique":
        n_with = sum(1 for q in questions if q["passage"])
        print(f"  [musique] {n_with}/{len(questions)} questions have passages")

    return questions

def make_prompt(question, passage=None):
    if passage:
        return f"Passage: {passage}\n\nQuestion: {question}"
    return question

# ── Per-question evaluation ───────────────────────────────────────────────────

def evaluate_question(model_id, model_stem, q):
    mt      = MODEL_MAX_TOKENS_OVERRIDE.get(model_stem, MODEL_MAX_TOKENS.get(model_stem, DEFAULT_MAX_TOKENS))
    passage = q.get("passage")
    gold    = q["ground_truth"]
    aliases = q["ground_truth_aliases"]

    direct_prompt = make_prompt(q["question"], passage)
    step1_prompt  = make_prompt(q["step1"],    passage)

    # Operadic (greedy)
    direct    = query(model_id, direct_prompt, TEMP_O, mt)
    a1        = query(model_id, step1_prompt,  TEMP_O, mt)
    step2_q   = q["step2"].replace("[A1]", a1).replace("[a1]", a1)
    expansion = query(model_id, make_prompt(step2_q, passage), TEMP_O, mt)

    op_f1         = token_f1(direct, expansion) if direct.strip() and expansion.strip() else 0.0
    op_consistent = op_f1 >= 0.5
    op_correct    = is_correct(direct, gold, aliases) if gold is not None else None

    # Naive: draw K_MAX samples, derive K3 stats from first K3
    naive_answers = [query(model_id, direct_prompt, TEMP_N, mt) for _ in range(K_MAX)]

    nv_pairwise_k3  = pairwise_agreement(naive_answers[:K3])
    nv_pairwise_k10 = pairwise_agreement(naive_answers)

    def mv_correct(answers):
        if gold is None or not any(a.strip() for a in answers):
            return None
        return is_correct(majority_vote(answers), gold, aliases)

    return {
        "question_id":       q["question_id"],
        "question":          q["question"],
        "op_direct":         direct,
        "op_step1_answer":   a1,
        "op_expansion":      expansion,
        "op_f1":             op_f1,
        "op_consistent":     op_consistent,
        "op_correct":        op_correct,
        "nv_answers":        naive_answers,
        "nv_pairwise_k3":    nv_pairwise_k3,
        "nv_pairwise_k10":   nv_pairwise_k10,
        "nv_mv_correct_k3":  mv_correct(naive_answers[:K3]),
        "nv_mv_correct_k10": mv_correct(naive_answers),
        "ground_truth":      gold,
    }

# ── Per-model-dataset runner ──────────────────────────────────────────────────

def run_model_dataset(model_stem, model_id, dataset_name, questions, skip_done=False):
    ds_out_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(ds_out_dir, exist_ok=True)
    out_path  = os.path.join(ds_out_dir, f"{model_stem}.json")
    ckpt_path = out_path + ".ckpt"

    if skip_done and os.path.exists(out_path):
        print(f"  [{model_stem}/{dataset_name}] already done, skipping")
        return None

    print(f"  [{model_stem}/{dataset_name}] {len(questions)} questions -- {model_id}")

    # Resume from checkpoint
    results, done_ids = [], set()
    if os.path.exists(ckpt_path):
        try:
            results   = json.load(open(ckpt_path))
            done_ids  = {r["question_id"] for r in results}
            print(f"  [{model_stem}/{dataset_name}] resuming from checkpoint ({len(results)} done)")
        except Exception:
            results = []
    questions = [q for q in questions if q["question_id"] not in done_ids]

    lock = threading.Lock()

    def _write_ckpt():
        with open(ckpt_path, "w") as f:
            json.dump(results, f)

    def process(q):
        try:
            rec = evaluate_question(model_id, model_stem, q)
            with lock:
                results.append(rec)
                n = len(results)
                if n % 50 == 0:
                    op_c  = sum(r["op_consistent"]    for r in results) / n
                    nv_c  = sum(r["nv_pairwise_k10"]  for r in results) / n
                    oa    = [r["op_correct"] for r in results if r["op_correct"] is not None]
                    acc_s = f"  op_acc={sum(oa)/len(oa):.3f}" if oa else ""
                    print(f"    [{model_stem}/{dataset_name}] {n}/{len(questions)+len(done_ids)}"
                          f"  op={op_c:.3f} nv_k10={nv_c:.3f}{acc_s}")
                    _write_ckpt()
        except RuntimeError as e:
            with lock:
                if "unavailable" in str(e).lower():
                    print(f"  [{model_stem}/{dataset_name}] MODEL UNAVAILABLE: {e}")
                raise

    aborted = False
    if questions:
        with ThreadPoolExecutor(max_workers=min(20, len(questions))) as pool:
            futs = {pool.submit(process, q): q for q in questions}
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc and "unavailable" in str(exc).lower():
                    pool.shutdown(wait=False, cancel_futures=True)
                    aborted = True
                    break

    if aborted or not results:
        return {"error": "model_unavailable" if aborted else "no_results",
                "model_stem": model_stem, "dataset": dataset_name}

    n    = len(results)
    oa   = [r["op_correct"]        for r in results if r["op_correct"]        is not None]
    na3  = [r["nv_mv_correct_k3"]  for r in results if r["nv_mv_correct_k3"]  is not None]
    na10 = [r["nv_mv_correct_k10"] for r in results if r["nv_mv_correct_k10"] is not None]

    summary = {
        "model_stem":          model_stem,
        "model_id":            model_id,
        "dataset":             dataset_name,
        "n":                   n,
        "op_consistency":      sum(r["op_consistent"]    for r in results) / n,
        "nv_consistency_k3":   sum(r["nv_pairwise_k3"]  for r in results) / n,
        "nv_consistency_k10":  sum(r["nv_pairwise_k10"] for r in results) / n,
        "op_accuracy":         sum(oa)   / len(oa)   if oa   else None,
        "nv_mv_accuracy_k3":   sum(na3)  / len(na3)  if na3  else None,
        "nv_mv_accuracy_k10":  sum(na10) / len(na10) if na10 else None,
        "n_with_gold":         len(oa),
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"  [{model_stem}/{dataset_name}] DONE"
          f"  op={summary['op_consistency']:.3f}"
          f"  nv_k3={summary['nv_consistency_k3']:.3f}"
          f"  nv_k10={summary['nv_consistency_k10']:.3f}"
          f"  op_acc={_fmt(summary['op_accuracy'])}"
          f"  -> {out_path}")
    return summary

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir, timestamp):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    fmt    = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

def log(msg=""):
    logging.info(msg)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _sem
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",         nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets",       nargs="+", default=list(DATASET_REF.keys()))
    parser.add_argument("--skip_done",      action="store_true")
    parser.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    _sem = threading.Semaphore(args.max_concurrent)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join(OUT_DIR, "logs")
    setup_logging(log_dir, timestamp)

    config = {
        "timestamp":          timestamp,
        "models":             {s: MODELS[s] for s in args.models if s in MODELS},
        "datasets":           args.datasets,
        "K_max":              K_MAX,
        "K3":                 K3,
        "T_operadic":         TEMP_O,
        "T_naive":            TEMP_N,
        "max_concurrent":     args.max_concurrent,
        "skip_done":          args.skip_done,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "system_prompt":      SYSTEM,
    }
    config_path = os.path.join(log_dir, f"config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log(f"nonthinking_oc | {len(args.models)} models x {len(args.datasets)} datasets"
        f" | K_max={K_MAX} K3={K3} T_op={TEMP_O} T_naive={TEMP_N}")
    log(f"  Output: {OUT_DIR}")
    log()

    drop_passages    = load_drop_passages()    if "drop"    in args.datasets else {}
    musique_passages = load_musique_passages() if "musique" in args.datasets else {}

    question_sets = {}
    for ds in args.datasets:
        log(f"Loading {ds}...")
        question_sets[ds] = load_questions(
            ds,
            drop_passages    = drop_passages    if ds == "drop"    else None,
            musique_passages = musique_passages if ds == "musique" else None,
        )
        log(f"  {len(question_sets[ds])} questions")
        if ds == "hotpotqa":
            log("  Loading HotpotQA gold from HF...")
            gold_map = load_hotpotqa_gold()
            patched  = 0
            for q in question_sets[ds]:
                lookup = q["question_id"].replace("HOTPOT_dev_", "").replace("HOTPOT_train_", "")
                if lookup in gold_map:
                    q["ground_truth"] = gold_map[lookup]
                    patched += 1
            log(f"  Patched {patched}/{len(question_sets[ds])} HotpotQA gold answers")
    log()

    tasks = [
        (stem, MODELS[stem], ds, question_sets[ds])
        for stem in args.models if stem in MODELS
        for ds in args.datasets
    ]
    log(f"Running {len(tasks)} tasks ({K_MAX} naive samples each, storing K=3 and K=10 stats)")
    log()

    all_summaries = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        fut_map = {
            pool.submit(run_model_dataset, s, m, ds, qs, args.skip_done): (s, ds)
            for s, m, ds, qs in tasks
        }
        for fut in as_completed(fut_map):
            stem, ds = fut_map[fut]
            try:
                r = fut.result()
                if r is not None:
                    all_summaries.append(r)
            except Exception as e:
                log(f"[ERROR] {stem}/{ds}: {e}")

    log()
    log("=== Summary ===")
    log(f"{'Model':<20} {'Dataset':<12} {'op_cons':>8} {'nv_k3':>7} {'nv_k10':>7} {'op_acc':>7}")
    log("-" * 75)
    for s in sorted(all_summaries, key=lambda x: (x.get("dataset",""), x.get("model_stem",""))):
        if "error" in s:
            log(f"{s.get('model_stem','?'):<20} {s.get('dataset','?'):<12}  ERROR: {s['error']}")
            continue
        def _f(v): return f"{v:.3f}" if v is not None else "n/a"
        log(f"{s['model_stem']:<20} {s['dataset']:<12}"
            f"  {s['op_consistency']:>7.3f}"
            f"  {s['nv_consistency_k3']:>6.3f}"
            f"  {s['nv_consistency_k10']:>6.3f}"
            f"  {_f(s['op_accuracy']):>6}")

    summary_path = os.path.join(OUT_DIR, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"config": config, "results": all_summaries}, f, indent=2)
    log(f"\nSaved summary -> {summary_path}")
    log("All done.")


if __name__ == "__main__":
    main()
