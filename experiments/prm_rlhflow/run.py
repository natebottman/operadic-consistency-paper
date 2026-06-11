"""Run RLHFlow PRM (Llama-3.1-8B-PRM-Mistral-Data) over all (model, dataset) cells.

For each step (separated by newlines in the response), score the step by
running the conversation up through that step + '\n\n' and reading the next-token
softmax probability of '+' relative to '-'.

Inputs:  /root/prm_inputs/{stem}__{ds}.jsonl
Outputs: /root/rlhflow_outputs/{stem}__{ds}.jsonl
"""
import json, os, sys, time, glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/rlhflow-prm"
INPUT_DIR  = "/root/prm_inputs"
OUTPUT_DIR = "/root/rlhflow_outputs"

print("Loading tokenizer + model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
mdl = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda").eval()
print("Model loaded.")

PLUS_ID  = tok.encode("+", add_special_tokens=False)[0]
MINUS_ID = tok.encode("-", add_special_tokens=False)[0]
print(f"plus={PLUS_ID} minus={MINUS_ID}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

@torch.no_grad()
def score_question(problem, response):
    steps = [s for s in response.split("\n") if s.strip()]
    if not steps:
        return [], None, None, None
    rewards = []
    for k in range(1, len(steps)+1):
        partial = "\n\n".join(steps[:k]) + "\n\n"
        msgs = [{"role": "user", "content": problem},
                {"role": "assistant", "content": partial}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        logits = mdl(ids).logits[0, -1]
        p_plus = torch.softmax(logits, dim=-1)[PLUS_ID].item()
        p_minus = torch.softmax(logits, dim=-1)[MINUS_ID].item()
        score = p_plus / (p_plus + p_minus + 1e-12)
        rewards.append(float(score))
    return rewards, float(min(rewards)), float(sum(rewards)/len(rewards)), float(rewards[-1])

files = sorted(glob.glob(f"{INPUT_DIR}/*.jsonl"))
print(f"Found {len(files)} input files")

t_start = time.time()
total_rows = 0
for fp in files:
    cell = os.path.basename(fp).replace(".jsonl", "")
    out_fp = f"{OUTPUT_DIR}/{cell}.jsonl"
    if os.path.exists(out_fp):
        print(f"  {cell}: already done, skipping")
        continue
    rows = [json.loads(line) for line in open(fp)]
    out_rows = []
    t0 = time.time()
    for r in rows:
        try:
            srs, mn, mean, last = score_question(r["problem"], r["response"])
            r2 = dict(r)
            r2["step_rewards"] = srs
            r2["prm_min"] = mn
            r2["prm_mean"] = mean
            r2["prm_last"] = last
            out_rows.append(r2)
        except Exception as e:
            print(f"  [{cell}] err: {str(e)[:120]}")
            r2 = dict(r); r2["step_rewards"] = None
            r2["prm_min"] = r2["prm_mean"] = r2["prm_last"] = None
            out_rows.append(r2)
    with open(out_fp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    total_rows += len(out_rows)
    elapsed = time.time() - t0
    print(f"  {cell}: {len(out_rows)} rows in {elapsed:.1f}s ({len(out_rows)/elapsed:.1f}/s)")

print(f"\nTotal: {total_rows} rows in {time.time()-t_start:.1f}s")
