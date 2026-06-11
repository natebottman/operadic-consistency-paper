"""NLI scoring of (op_direct, op_expansion) pairs across all (model, dataset) cells.

For each existing per-question record in nonthinking_oc, run an NLI classifier
(DeBERTa-v3-large fine-tuned on MNLI) bidirectionally on the (direct, expansion)
pair and store:
  - p_entail_d2e: probability that direct entails expansion
  - p_entail_e2d: probability that expansion entails direct
  - nli_consistent: bidirectional entailment indicator (both p_entail >= 0.5)
  - nli_score: min(p_entail_d2e, p_entail_e2d) -- continuous "mutual entailment"

Outputs: data/raw/nli_scores/{stem}/{dataset}.json
"""
import os
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EXP_DIR = os.environ.get('EXP_ROOT', os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')) + '/nonthinking_oc'
OUT_DIR = os.path.join(
    os.environ.get("EXP_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")),
    "nli_scores",
)
DATASETS = ["hotpotqa", "musique", "strategyqa", "drop"]
MODELS = ["cogito_671b","deepseek_v3_1","gemma3n_4b","liquidai_24b","llama33_70b",
          "llama8b","qwen3_235b","qwen7b","gpt_4o_mini","gemini_2_5_flash","llama_4_maverick",
          "gemini_2_5_pro","llama_31_405b","mistral_large_2"]

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"  # strong public NLI checkpoint
BATCH = 32

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_nli():
    print(f"Loading {NLI_MODEL}...")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    dev = get_device()
    mdl.to(dev).eval()
    # Determine label index for ENTAILMENT
    id2label = mdl.config.id2label
    ent_idx = next(i for i, l in id2label.items() if "entail" in str(l).lower())
    print(f"  device={dev}  entail_idx={ent_idx}  labels={id2label}")
    return tok, mdl, dev, ent_idx

@torch.no_grad()
def nli_batch(tok, mdl, dev, ent_idx, prems, hyps):
    """Returns p(entailment) for each (prem, hyp) pair."""
    enc = tok(prems, hyps, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(dev) for k, v in enc.items()}
    logits = mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, ent_idx]
    return probs.cpu().tolist()

def score_cell(stem, ds, tok, mdl, dev, ent_idx):
    src = f"{EXP_DIR}/{ds}/{stem}.json"
    if not os.path.exists(src):
        print(f"  [{stem}/{ds}] SKIP (no source)")
        return
    out_dir = f"{OUT_DIR}/{stem}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{ds}.json"
    if os.path.exists(out_path):
        print(f"  [{stem}/{ds}] already done, skipping")
        return

    data = json.load(open(src))
    rows = []
    for r in data["results"]:
        d = (r.get("op_direct") or "").strip()
        e = (r.get("op_expansion") or "").strip()
        if not d or not e:
            rows.append({"question_id": r["question_id"],
                         "p_entail_d2e": 0.0, "p_entail_e2d": 0.0,
                         "nli_consistent": False, "nli_score": 0.0})
            continue
        rows.append({"question_id": r["question_id"], "_d": d, "_e": e})

    # Run NLI in batches
    todo = [r for r in rows if "_d" in r]
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i+BATCH]
        # direct -> expansion
        p_de = nli_batch(tok, mdl, dev, ent_idx,
                         [r["_d"] for r in batch], [r["_e"] for r in batch])
        # expansion -> direct
        p_ed = nli_batch(tok, mdl, dev, ent_idx,
                         [r["_e"] for r in batch], [r["_d"] for r in batch])
        for r, a, b in zip(batch, p_de, p_ed):
            r["p_entail_d2e"] = a
            r["p_entail_e2d"] = b
            r["nli_consistent"] = (a >= 0.5) and (b >= 0.5)
            r["nli_score"] = min(a, b)
            del r["_d"]; del r["_e"]

    n = len(rows)
    n_cons = sum(r["nli_consistent"] for r in rows)
    avg_score = sum(r["nli_score"] for r in rows) / n if n else 0
    print(f"  [{stem}/{ds}] n={n}  nli_consistent_rate={n_cons/n:.3f}  mean_score={avg_score:.3f}")
    with open(out_path, "w") as f:
        json.dump({"summary": {"model_stem": stem, "dataset": ds, "n": n,
                               "nli_consistent_rate": n_cons/n if n else 0,
                               "mean_nli_score": avg_score},
                   "results": rows}, f, indent=2)

def main():
    tok, mdl, dev, ent_idx = load_nli()
    t0 = time.time()
    for stem in MODELS:
        for ds in DATASETS:
            score_cell(stem, ds, tok, mdl, dev, ent_idx)
    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
