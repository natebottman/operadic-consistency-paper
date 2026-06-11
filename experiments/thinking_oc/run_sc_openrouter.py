"""
run_thinking_sc.py

Self-consistency baseline for thinking models -- companion to run_cot_pilot.py.

Given an existing cot_pilot output (or any JSON with a 'results' list whose items
have 'question_id' and 'question'), re-query the same thinking model K times at
positive temperature, extract a short factoid answer from each sample using a
cheap "plain" extractor model, then compute pairwise token-F1 self-consistency
on the SHORT answers (so we measure answer-level agreement, not surrounding-text
similarity).

Usage:
    TOGETHER_API_KEY=... python run_thinking_sc.py \
        --input  data/raw/cot_pilot_smoke.json \
        --output data/raw/sc_smoke.json \
        --model  deepseek-ai/DeepSeek-R1 \
        --plain-model meta-llama/Llama-3.3-70B-Instruct-Turbo \
        --k 3 --temperature 0.7 --concurrency 4
"""
import argparse, json, os, re, sys, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import together

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rpm  # type: ignore

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("ERROR: OPENROUTER_API_KEY env var not set")
PLAIN_API_KEY = os.environ.get("TOGETHER_API_KEY")  # plain extractor still uses Together

client = OpenAI(api_key=API_KEY, base_url='https://openrouter.ai/api/v1')
plain_client = together.Together(api_key=PLAIN_API_KEY)  # extractor on Together


# ── F1 + agreement ───────────────────────────────────────────────────────────

def normalize(text: str):
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(a: str, b: str) -> float:
    pa, pb = normalize(a), normalize(b)
    if not pa or not pb:
        return float(pa == pb)
    common = Counter(pa) & Counter(pb)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pa)
    r = n / len(pb)
    return 2 * p * r / (p + r)


def agree(a: str, b: str, thresh: float = 0.5) -> bool:
    return token_f1(a, b) >= thresh


def pairwise_agreement(answers, thresh: float = 0.5) -> float:
    n = len(answers)
    if n < 2:
        return 1.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return sum(agree(answers[i], answers[j], thresh) for i, j in pairs) / len(pairs)


def all_agree(answers, thresh: float = 0.5) -> bool:
    if len(answers) <= 1:
        return True
    return all(agree(a, answers[0], thresh) for a in answers[1:])


# ── thinking-model query ─────────────────────────────────────────────────────

def query_thinking(model: str, question: str, max_tokens: int = 8192,
                   temperature: float = 0.7, retries: int = 6) -> dict:
    """Returns {verbose_answer, reasoning_words, error}."""
    last_err = None
    for attempt in range(retries):
        try:
            _rpm.wait_for_headroom()
            _rpm.log_request()
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer the question accurately."},
                    {"role": "user",   "content": question},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = r.choices[0].message
            reasoning = getattr(msg, "reasoning", "") or ""
            content = (msg.content or "").strip()
            return {
                "verbose_answer":  content,
                "reasoning_words": len(reasoning.split()),
                "error":           None,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                base = 2 ** (attempt + 1)
                jitter = base * 0.5 * (0.5 + 0.5 * (attempt % 2))
                time.sleep(base + jitter)
    return {"verbose_answer": "", "reasoning_words": 0, "error": last_err}


_SHORT_ANSWER_PROMPT = """\
Question: {question}

Verbose answer:
{verbose}

Extract ONLY the direct answer to the question in 5 words or fewer.
Output the answer only -- no explanation, no punctuation except what's part of the answer.\
"""


def extract_short_answer(plain_model: str, question: str, verbose: str,
                         retries: int = 6) -> str:
    """Use the plain extractor model to distill a verbose response → short answer."""
    if not verbose:
        return ""
    prompt = _SHORT_ANSWER_PROMPT.format(question=question, verbose=verbose[:600])
    for attempt in range(retries):
        try:
            _rpm.wait_for_headroom()
            _rpm.log_request()
            r = plain_client.chat.completions.create(
                model=plain_model,
                messages=[
                    {"role": "system", "content":
                        "You are a concise factual question-answering assistant. "
                        "Answer with a short phrase or a few words. "
                        "Do not explain your reasoning. Do not use full sentences unless necessary."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32,
                temperature=0.0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                base = 2 ** (attempt + 1)
                jitter = base * 0.5 * (0.5 + 0.5 * (attempt % 2))
                time.sleep(base + jitter)
    return ""


# ── per-question worker ──────────────────────────────────────────────────────

def process_question(q_idx: int, ex: dict, model: str, plain_model: str,
                     k: int, temperature: float) -> dict:
    qid = ex.get("question_id") or f"q_{q_idx}"
    question = ex.get("question")
    samples = []
    short_answers = []
    for _ in range(k):
        s = query_thinking(model, question, temperature=temperature)
        short = extract_short_answer(plain_model, question, s.get("verbose_answer", ""))
        s["short_answer"] = short
        samples.append(s)
        short_answers.append(short)
    return {
        "question_id":  qid,
        "question":     question,
        "k":            k,
        "samples":      samples,
        "short_answers": short_answers,
        "pairwise_f1":  pairwise_agreement(short_answers),
        "unanimous":    all_agree(short_answers),
        "ground_truth": ex.get("ground_truth"),
        "is_correct_majority": (
            agree(_majority(short_answers), ex.get("ground_truth", ""))
            if ex.get("ground_truth") else None
        ),
    }


def _majority(answers):
    """Pick the answer that agrees with the most others (ties → first)."""
    if not answers:
        return ""
    best, best_score = answers[0], -1
    for a in answers:
        score = sum(1 for b in answers if agree(a, b))
        if score > best_score:
            best, best_score = a, score
    return best


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="JSON with a 'results' list of questions")
    p.add_argument("--output", required=True)
    p.add_argument("--model",  required=True, help="Thinking model to sample K times")
    p.add_argument("--plain-model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                   help="Plain extractor model for short-answer distillation")
    p.add_argument("--k",      type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, take only the first --limit questions from input.")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    questions = data["results"] if isinstance(data, dict) else data
    if args.limit is not None:
        questions = questions[:args.limit]

    print(f"[sc] thinking_model={args.model}  plain_model={args.plain_model}", flush=True)
    print(f"[sc] n={len(questions)}  k={args.k}  T={args.temperature}  concurrency={args.concurrency}", flush=True)

    results = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(process_question, i, q, args.model, args.plain_model,
                            args.k, args.temperature): i
                for i, q in enumerate(questions)}
        for fut in as_completed(futs):
            i = futs[fut]
            rec = fut.result()
            results[i] = rec
            done = sum(1 for r in results if r is not None)
            cur_rpm = _rpm.current_rpm()
            print(f"  [{done}/{len(questions)}] {rec['question_id']} pairwise={rec['pairwise_f1']:.2f} "
                  f"unan={rec['unanimous']} rpm={cur_rpm}", flush=True)

    summary = {
        "model":            args.model,
        "n":                len(results),
        "k":                args.k,
        "temperature":      args.temperature,
        "mean_pairwise_f1": round(sum(r["pairwise_f1"] for r in results) / len(results), 4),
        "frac_unanimous":   round(sum(1 for r in results if r["unanimous"]) / len(results), 4),
    }
    n_with_gt = sum(1 for r in results if r["is_correct_majority"] is not None)
    if n_with_gt:
        summary["mv_accuracy"] = round(
            sum(1 for r in results if r["is_correct_majority"]) / n_with_gt, 4
        )

    out = {"summary": summary, "results": results, "config": vars(args)}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[sc] saved to {args.output}")
    print(f"[sc] summary: {summary}")
    print(f"[sc] peak rpm = {_rpm.peak_rpm()}")


if __name__ == "__main__":
    main()
