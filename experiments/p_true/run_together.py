"""P(True) self-evaluation baseline (Kadavath et al. 2022).

For each (model, question, greedy_answer) triple, ask the model:
  Q: <question>
  Proposed answer: <greedy>
  Is the proposed answer correct? Reply with only "Yes" or "No".

Then read logprobs of "Yes" / "No" tokens from the first generated token,
normalize, and report P(yes).

Input file formats supported:
  (1) Non-thinking nonthinking_oc: {"results": [{"question_id", "question", "op_direct", ...}, ...]}
       → use record.op_direct as the proposed answer.
  (2) Thinking cot_pilot: {"results": [{"question_id", "question", "stated_answer" or "direct_answer", ...}, ...]}
       → use record.stated_answer (or pilot direct) as the proposed answer.

Output: {"model", "input", "results": [{"question_id", "p_yes", "raw_first_token", ...}, ...]}
"""
import os, sys, json, math, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import together
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rpm

API_KEY = os.environ.get("TOGETHER_API_KEY")
if not API_KEY:
    sys.exit("ERROR: TOGETHER_API_KEY env var not set")
client = together.Together(api_key=API_KEY)

PROMPT_NOPASSAGE = """Question: {q}

Proposed answer: {a}

Is the proposed answer correct? Reply with only "Yes" or "No"."""

PROMPT_WITHPASSAGE = """Passage: {p}

Question: {q}

Proposed answer: {a}

Is the proposed answer correct? Reply with only "Yes" or "No"."""

SYSTEM = "You are a careful evaluator. Reply with exactly one word: Yes or No."


def call_p_true(model, question, answer, passage=None, max_tokens=4, retries=6):
    if passage:
        user = PROMPT_WITHPASSAGE.format(p=passage, q=question, a=answer)
    else:
        user = PROMPT_NOPASSAGE.format(q=question, a=answer)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user},
    ]
    last_err = None
    for attempt in range(retries):
        try:
            _rpm.wait_for_headroom()
            _rpm.log_request()
            return client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tokens, temperature=0.0,
                logprobs=True,
                extra_body={"top_logprobs": 5},
            )
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    raise last_err


def extract_p_yes(resp):
    """Compute P(yes), preferring full top_logprobs (P(yes)/(P(yes)+P(no)))
    and falling back to the chosen-token logprob heuristic when only top-1 is exposed.
    Returns (p_yes, first_token, info_dict)."""
    try:
        lp = resp.choices[0].logprobs
        first_token = None
        first_lp = None
        cand = {}  # full top-k Yes/No mass when available

        # ---- (1) Try Together's "tokens / top_logprobs" schema ------------
        if hasattr(lp, "tokens") and lp.tokens:
            first_token = lp.tokens[0]
            tlp = getattr(lp, "token_logprobs", None) or []
            if tlp: first_lp = tlp[0]
            tops_list = getattr(lp, "top_logprobs", None) or []
            if tops_list:
                tops = tops_list[0]
                if isinstance(tops, dict):
                    for tok, lpv in tops.items():
                        key = tok.strip().lower().rstrip(".:!,")
                        if key in ("yes", "no"):
                            cand[key] = max(cand.get(key, -1e9), lpv)

        # ---- (2) OpenAI-style "content[i].top_logprobs" schema ------------
        elif hasattr(lp, "content") and lp.content:
            tok = lp.content[0]
            _g = (lambda o, k: o[k] if isinstance(o, dict) else getattr(o, k, None))
            first_token = _g(tok, "token")
            first_lp = _g(tok, "logprob")
            tops = _g(tok, "top_logprobs") or []
            for t in tops:
                tk = _g(t, "token"); tlp = _g(t, "logprob")
                if tk is None: continue
                key = tk.strip().lower().rstrip(".:!,")
                if key in ("yes", "no"):
                    cand[key] = max(cand.get(key, -1e9), tlp)

        # If we have top-k yes/no, use the precise normalization.
        if cand.get("yes") is not None or cand.get("no") is not None:
            py = math.exp(cand.get("yes", -50.0))
            pn = math.exp(cand.get("no", -50.0))
            z = py + pn
            if z > 0:
                return py / z, first_token, {"mode": "topk", **cand}

        # ---- (3) Fallback: first-token heuristic --------------------------
        # P(emitted) = exp(first_lp). If "yes", P(yes)=that; if "no", P(yes)=1-that.
        if first_token is not None and first_lp is not None:
            key = first_token.strip().lower().rstrip(".:!,")
            p_emit = math.exp(min(first_lp, 0.0))  # clamp num issues
            if key in ("yes", "yeah", "yep", "true", "y"):
                return p_emit, first_token, {"mode": "fallback_yes", "lp": first_lp}
            elif key in ("no", "nope", "false", "n"):
                return 1.0 - p_emit, first_token, {"mode": "fallback_no", "lp": first_lp}
            else:
                return None, first_token, {"mode": "fallback_other", "lp": first_lp}

        return None, first_token, {"mode": "no_logprobs"}
    except Exception as e:
        return None, None, {"_err": str(e)}


def process(qid, q, a, model, passage=None):
    if not a:
        return {"question_id": qid, "p_yes": None, "skip_reason": "empty_answer"}
    try:
        r = call_p_true(model, q, a, passage=passage)
    except Exception as e:
        return {"question_id": qid, "p_yes": None, "error": str(e)}
    p_yes, raw, info = extract_p_yes(r)
    return {
        "question_id":   qid,
        "p_yes":         p_yes,
        "raw_first_tok": raw,
        "info":          info,
    }


def load_records(path):
    """Returns list of (qid, question, proposed_answer) triples."""
    d = json.load(open(path))
    rows = []
    for r in d.get("results", []):
        qid = r.get("question_id") or f"q_{len(rows)}"
        q = r.get("question") or ""
        # try a few schemas
        a = (r.get("op_direct")
             or r.get("stated_answer")
             or (r.get("pilot_direct") or {}).get("answer")
             or r.get("direct_answer")
             or r.get("greedy_answer")
             or "")
        rows.append((qid, q, a))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model",  required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--passages-pkl", default=None,
                    help="Path to a pickle file: {dataset_key: {qid: passage_text}}")
    ap.add_argument("--passages-key", default=None,
                    help="Key into passages-pkl (e.g. 'musique', 'drop')")
    args = ap.parse_args()

    passages = {}
    if args.passages_pkl and args.passages_key:
        import pickle
        all_p = pickle.load(open(args.passages_pkl, "rb"))
        passages = all_p.get(args.passages_key, {})
        print(f"[p_true] loaded {len(passages)} passages for key='{args.passages_key}'", flush=True)

    rows = load_records(args.input)
    if args.n: rows = rows[:args.n]
    print(f"[p_true] model={args.model}  n={len(rows)}  conc={args.concurrency}", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # For DROP, the question_id has a HF query_id format; for MuSiQue it's the dataset id.
        # Try direct lookup, then a few transforms.
        def get_passage(qid):
            if not passages: return None
            if qid in passages: return passages[qid]
            # DROP qids look like 'DROP_dev_history_1043_<uuid>'; passage key is the trailing uuid.
            tail = qid.split("_")[-1]
            if tail in passages: return passages[tail]
            return None
        futs = {pool.submit(process, qid, q, a, args.model, get_passage(qid)): (qid, a)
                for qid, q, a in rows}
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            out.append(res)
            if i % 20 == 0 or i == len(futs) - 1:
                got = sum(1 for r in out if r.get("p_yes") is not None)
                print(f"  [{i+1}/{len(rows)}] got_p_yes={got}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model, "input": args.input,
            "n": len(out), "n_with_p_yes": sum(1 for r in out if r.get("p_yes") is not None),
            "results": out,
        }, f, indent=2)
    print(f"[p_true] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
