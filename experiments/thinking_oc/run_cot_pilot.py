"""
run_cot_pilot.py (patched v2)

Patched copy of operadic_consistency/scripts/run_cot_pilot.py with:
  - API key from TOGETHER_API_KEY env var
  - Optional --model-think and --model-plain CLI flags (so we can run other thinking models)
  - Multi-dataset support (musique, strategyqa, gsm8k, drop) -- borrowed from run_cot_pilot_multi.py
  - Thread-pool concurrency over examples
  - RPM logging via _rpm.py + per-call pre-throttle

Usage:
    TOGETHER_API_KEY=... python run_cot_pilot.py \
        --dataset musique --n 2 --output data/raw/cot_pilot_smoke.json \
        --concurrency 4
"""
import os
import argparse, json, os, re, sys, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import together
from datasets import load_dataset

# Make loaders importable from the original repo
_OC_SCRIPTS = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _OC_SCRIPTS)

# RPM logger (sibling module)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rpm  # type: ignore

# ── config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("TOGETHER_API_KEY")
if not API_KEY:
    sys.exit("ERROR: TOGETHER_API_KEY env var not set")

DEFAULT_MODEL_THINK = "Qwen/Qwen3-235B-A22B-Thinking-2507"
DEFAULT_MODEL_PLAIN = "Qwen/Qwen3-Next-80B-A3B-Instruct"

client = together.Together(api_key=API_KEY)

_PLAIN_SYSTEM = (
    "You are a concise factual question-answering assistant. "
    "Answer with a short phrase or a few words. "
    "Do not explain your reasoning. Do not use full sentences unless necessary."
)

# Set by main(); used by call_thinking / call_plain
MODEL_THINK = DEFAULT_MODEL_THINK
MODEL_PLAIN = DEFAULT_MODEL_PLAIN


# ── Example dataclass (multi-dataset) ─────────────────────────────────────────

@dataclass
class Example:
    question_id: str
    question: str
    display_question: str
    ground_truth: str
    ground_truth_aliases: List[str] = field(default_factory=list)
    gt_sub_q1: Optional[str] = None
    gt_sub_q2: Optional[str] = None
    context: Optional[str] = None


# ── Dataset loaders (mirrored from run_cot_pilot_multi.py) ───────────────────

def _load_musique(n, offset=0):
    # No context injection -- MuSiQue sub-questions are self-contained factual queries
    from loaders.musique_loader import load_musique_2hop
    raw = load_musique_2hop("validation")[offset:offset + n]
    return [
        Example(
            question_id=ex.question_id,
            question=ex.original_question,
            display_question=ex.original_question,
            ground_truth=ex.ground_truth,
            ground_truth_aliases=list(ex.ground_truth_aliases),
            gt_sub_q1=ex.toq.nodes[1].text,
            gt_sub_q2=ex.toq.nodes[2].text,
            context=None,
        )
        for ex in raw
    ]


def _load_strategyqa(n, offset=0):
    # No context injection -- StrategyQA sub-questions are self-contained factual queries
    ds = load_dataset("ChilleD/StrategyQA", split="train")
    out = []
    for i, ex in enumerate(ds):
        if i < offset: continue
        if i >= offset + n:
            break
        out.append(Example(
            question_id=f"strategyqa_{i}",
            question=ex["question"],
            display_question=ex["question"],
            ground_truth="yes" if ex.get("answer") else "no",
            context=None,
        ))
    return out


def _load_gsm8k(n, offset=0):
    # Inject the full problem statement (= the question) as context
    ds = load_dataset("gsm8k", "main", split="train")
    out = []
    for i, ex in enumerate(ds):
        if i < offset: continue
        if i >= offset + n:
            break
        m = re.search(r"####\s*([-\d,\.]+)", ex["answer"])
        gt = m.group(1).replace(",", "") if m else ex["answer"].strip().split("####")[-1].strip()
        out.append(Example(
            question_id=f"gsm8k_{i}",
            question=ex["question"],
            display_question=ex["question"],
            ground_truth=gt,
            context=ex["question"],  # problem statement is the context for sub-questions
        ))
    return out


def _load_drop(n, offset=0):
    # Inject ONLY the passage (not the original question) as context
    ds = load_dataset("ucinlp/drop", split="validation")
    out = []
    for i, ex in enumerate(ds):
        if i < offset: continue
        if i >= offset + n:
            break
        ans = ex["answers_spans"]["spans"]
        gt = ans[0] if ans else ""
        passage = ex.get("passage", "").strip()
        q = ex["question"].strip()
        display = f"Passage:\n{passage}\n\nQuestion: {q}"
        out.append(Example(
            question_id=f"drop_{i}",
            question=display,
            display_question=q,
            ground_truth=gt,
            ground_truth_aliases=ans[1:],
            context=passage,  # ONLY passage, not the original question
        ))
    return out


LOADERS = {
    "musique":    _load_musique,
    "strategyqa": _load_strategyqa,
    "gsm8k":      _load_gsm8k,
    "drop":       _load_drop,
}


# ── LLM call helpers (with RPM logging + pre-throttle) ───────────────────────

def _create(*, model, messages, max_tokens, temperature, retries: int = 6):
    """Throttled API call with exponential backoff on 429/503/transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            _rpm.wait_for_headroom()
            _rpm.log_request()
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            # backoff: 2,4,8,16,32,64s with jitter
            if attempt < retries - 1:
                base = 2 ** (attempt + 1)
                jitter = base * 0.5 * (0.5 + 0.5 * (attempt % 2))
                time.sleep(base + jitter)
            else:
                raise
    raise last_err


def call_thinking(question: str, max_tokens: int = 8192) -> tuple:
    r = _create(
        model=MODEL_THINK,
        messages=[
            {"role": "system", "content": "Answer the question accurately."},
            {"role": "user",   "content": question},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning", "") or ""
    return reasoning, (msg.content or "").strip()


def call_plain(question: str, max_tokens: int = 256) -> str:
    r = _create(
        model=MODEL_PLAIN,
        messages=[
            {"role": "system", "content": _PLAIN_SYSTEM},
            {"role": "user",   "content": question},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (r.choices[0].message.content or "").strip()


# ── factorization extractor ───────────────────────────────────────────────────

_EXTRACT_SYSTEM = "You are a precise analytical assistant. Follow output format instructions exactly."

_EXTRACT_PROMPT = """\
A reasoning model was asked the question below, produced a chain-of-thought, and gave a final answer.

Question: {question}

Reasoning trace:
{reasoning}

Stated final answer: {final_answer}

Your task: identify the single most important intermediate step -- a sub-question Q1 that the model answered during reasoning, and a follow-up question Q2 such that knowing [A1] = answer to Q1 directly yields the final answer.

CRITICAL RULES for FOLLOW_UP:
- The literal string [A1] MUST appear in FOLLOW_UP as a placeholder for the intermediate answer
- Do NOT put the actual sub-answer value inside FOLLOW_UP, only [A1]
- FOLLOW_UP should be simpler than the original question

EXAMPLE (do not copy -- for format illustration only):
  Question: What is the capital of the country where the Eiffel Tower is located?
  SUB_QUESTION: In which country is the Eiffel Tower located?
  SUB_ANSWER: France
  FOLLOW_UP: What is the capital of [A1]?
  UNFACTORABLE: no

Now output for the actual question above, EXACTLY in this format (no extra text, no markdown):
SUB_QUESTION: <standalone question whose answer is the key intermediate step>
SUB_ANSWER: <the intermediate answer as stated in the reasoning>
FOLLOW_UP: <question using [A1] as placeholder that directly yields the final answer>
UNFACTORABLE: <yes or no -- yes only if no clean 2-step structure exists>\
"""


def extract_factorization(question: str, reasoning: str, final_answer: str) -> dict:
    words = reasoning.split()
    if len(words) > 2500:
        reasoning_trunc = " ".join(words[:2500]) + " …[truncated]"
    else:
        reasoning_trunc = reasoning

    prompt = _EXTRACT_PROMPT.format(
        question=question,
        reasoning=reasoning_trunc,
        final_answer=final_answer,
    )

    r = _create(
        model=MODEL_PLAIN,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    raw = r.choices[0].message.content or ""

    result = {
        "raw": raw.strip(),
        "sub_q1": None,
        "sub_a1_stated": None,
        "sub_q2_template": None,
        "factorable": False,
    }

    unfact = re.search(r"UNFACTORABLE:\s*(yes|no)", raw, re.IGNORECASE)
    if unfact and unfact.group(1).lower() == "yes":
        return result

    sub_q  = re.search(r"SUB_QUESTION:\s*(.+?)(?=\nSUB_ANSWER:|\Z)", raw, re.DOTALL)
    sub_a  = re.search(r"SUB_ANSWER:\s*(.+?)(?=\nFOLLOW_UP:|\Z)",     raw, re.DOTALL)
    follow = re.search(r"FOLLOW_UP:\s*(.+?)(?=\nUNFACTORABLE:|\Z)",   raw, re.DOTALL)

    if sub_q and sub_a and follow:
        sub_q1 = sub_q.group(1).strip()
        sub_a1 = sub_a.group(1).strip()
        sub_q2 = follow.group(1).strip()
        if "[A1]" not in sub_q2 and sub_a1:
            cand = sub_q2.replace(sub_a1, "[A1]")
            if "[A1]" in cand:
                sub_q2 = cand
        if "[A1]" in sub_q2 and sub_q1 and sub_a1:
            result.update({
                "sub_q1": sub_q1,
                "sub_a1_stated": sub_a1,
                "sub_q2_template": sub_q2,
                "factorable": True,
            })
    return result


# ── short-answer extraction ───────────────────────────────────────────────────

_SHORT_ANSWER_PROMPT = """\
Question: {question}

Verbose answer:
{verbose}

Extract ONLY the direct answer to the question in 5 words or fewer.
Output the answer only -- no explanation, no punctuation except what's part of the answer.\
"""


def extract_short_answer(question: str, verbose: str) -> str:
    prompt = _SHORT_ANSWER_PROMPT.format(question=question, verbose=verbose[:600])
    return call_plain(prompt, max_tokens=32)


# ── metric helpers ────────────────────────────────────────────────────────────

def _normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(pred, gt, aliases=()):
    candidates = [gt] + list(aliases)
    pred_tok = _normalize(pred)
    best = 0.0
    for c in candidates:
        gt_tok = _normalize(c)
        common = Counter(pred_tok) & Counter(gt_tok)
        n = sum(common.values())
        if not pred_tok or not gt_tok or n == 0:
            continue
        p = n / len(pred_tok)
        r = n / len(gt_tok)
        best = max(best, 2 * p * r / (p + r))
    return best


# ── per-example pipeline ──────────────────────────────────────────────────────

def process_example(idx: int, ex: Example) -> dict:
    rec = {
        "question_id": ex.question_id,
        "question":    ex.display_question,
        "ground_truth": ex.ground_truth,
        "ground_truth_aliases": list(ex.ground_truth_aliases),
        "gt_sub_q1":   ex.gt_sub_q1,
        "gt_sub_q2":   ex.gt_sub_q2,
    }
    try:
        t0 = time.time()
        reasoning, verbose_answer = call_thinking(ex.question)
        rec["verbose_answer"]  = verbose_answer
        rec["reasoning_words"] = len(reasoning.split())

        stated_answer = extract_short_answer(ex.display_question, verbose_answer)
        rec["stated_answer"] = stated_answer

        rec["is_correct"] = token_f1(stated_answer, ex.ground_truth, ex.ground_truth_aliases) >= 0.5

        ext = extract_factorization(ex.display_question, reasoning, stated_answer)
        rec["extraction"] = ext

        if ext["factorable"]:
            sub_q1, sub_q2_tmpl = ext["sub_q1"], ext["sub_q2_template"]
            if ex.gt_sub_q1:
                rec["sub_q1_match"] = token_f1(sub_q1, ex.gt_sub_q1) >= 0.5
            else:
                rec["sub_q1_match"] = None

            # Per-dataset context injection (matches deep-dive's run_cot_pilot_multi.py):
            #   musique, strategyqa: no context (sub-questions are self-contained)
            #   gsm8k: full problem statement (so sub-Qs have given quantities)
            #   drop:  ONLY the passage (so sub-Qs aren't redundant with the original question)
            def with_ctx(sub_q: str) -> str:
                if ex.context:
                    return f"Context:\n{ex.context}\n\nQuestion: {sub_q}"
                return sub_q

            # Pilot A: no-thinking sub-steps
            A1_plain = call_plain(with_ctx(sub_q1))
            q2_plain = sub_q2_tmpl.replace("[A1]", A1_plain)
            ans_A    = call_plain(with_ctx(q2_plain))
            rec["pilot_nothink"] = {
                "A1":          A1_plain,
                "answer":      ans_A,
                "consistent":  token_f1(stated_answer, ans_A) >= 0.5,
                "sub_a1_match": token_f1(A1_plain, ext["sub_a1_stated"]) >= 0.5,
            }

            # Pilot B: thinking sub-steps
            _, A1_think_verbose = call_thinking(with_ctx(sub_q1), max_tokens=4096)
            A1_think = extract_short_answer(sub_q1, A1_think_verbose)
            q2_think = sub_q2_tmpl.replace("[A1]", A1_think)
            _, ans_B_verbose = call_thinking(with_ctx(q2_think), max_tokens=4096)
            ans_B = extract_short_answer(q2_think, ans_B_verbose)
            rec["pilot_think"] = {
                "A1":          A1_think,
                "answer":      ans_B,
                "consistent":  token_f1(stated_answer, ans_B) >= 0.5,
                "sub_a1_match": token_f1(A1_think, ext["sub_a1_stated"]) >= 0.5,
            }
        else:
            rec["sub_q1_match"]  = None
            rec["pilot_nothink"] = None
            rec["pilot_think"]   = None

        rec["_elapsed"] = round(time.time() - t0, 2)
    except Exception as e:
        rec["error"]     = str(e)
        rec["traceback"] = traceback.format_exc()
    return rec


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="musique", choices=list(LOADERS))
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--offset", type=int, default=0,
                   help="Skip the first --offset examples; lets us produce alternate seeds.")
    p.add_argument("--output", default="data/raw/cot_pilot_smoke.json")
    p.add_argument("--model-think", default=DEFAULT_MODEL_THINK)
    p.add_argument("--model-plain", default=DEFAULT_MODEL_PLAIN)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    global MODEL_THINK, MODEL_PLAIN
    MODEL_THINK = args.model_think
    MODEL_PLAIN = args.model_plain

    print(f"[cot_pilot] dataset={args.dataset} n={args.n} model_think={MODEL_THINK} concurrency={args.concurrency}", flush=True)

    examples = LOADERS[args.dataset](args.n, offset=args.offset)
    print(f"[cot_pilot] loaded {len(examples)} examples (offset={args.offset})", flush=True)

    results = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex_pool:
        futs = {ex_pool.submit(process_example, i, ex): i for i, ex in enumerate(examples)}
        for fut in as_completed(futs):
            i = futs[fut]
            rec = fut.result()
            results[i] = rec
            done = sum(1 for r in results if r is not None)
            cur_rpm = _rpm.current_rpm()
            print(f"  [{done}/{len(examples)}] {rec['question_id']} elapsed={rec.get('_elapsed','?')}s "
                  f"correct={rec.get('is_correct')} factorable={(rec.get('extraction') or {}).get('factorable')} "
                  f"rpm={cur_rpm}", flush=True)

    summary = _summarize(results)
    out = {"summary": summary, "results": results, "config": {
        "dataset": args.dataset, "n": args.n,
        "model_think": MODEL_THINK, "model_plain": MODEL_PLAIN,
        "concurrency": args.concurrency,
    }}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[cot_pilot] saved to {args.output}")
    print(f"[cot_pilot] peak rpm = {_rpm.peak_rpm()}")


def _summarize(results):
    n = sum(1 for r in results if r and not r.get("error"))
    factorable = [r for r in results if r and (r.get("extraction") or {}).get("factorable")]
    n_fac = len(factorable)
    n_correct = sum(1 for r in results if r and r.get("is_correct"))
    n_cons_A = sum(1 for r in factorable if (r.get("pilot_nothink") or {}).get("consistent"))
    n_cons_B = sum(1 for r in factorable if (r.get("pilot_think")   or {}).get("consistent"))
    return {
        "n_total": n,
        "n_factorable": n_fac,
        "n_correct": n_correct,
        "factorable_rate": round(n_fac / n, 3) if n else 0,
        "accuracy":        round(n_correct / n, 3) if n else 0,
        "consistency_nothink": round(n_cons_A / n_fac, 3) if n_fac else 0,
        "consistency_think":   round(n_cons_B / n_fac, 3) if n_fac else 0,
    }


if __name__ == "__main__":
    main()
