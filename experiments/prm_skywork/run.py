"""Run Skywork-o1-Open-PRM-Qwen-2.5-7B over all prepared (model, dataset) cells.

Inputs:  /root/prm_inputs/{stem}__{ds}.jsonl     (one row per question)
Outputs: /root/prm_outputs/{stem}__{ds}.jsonl    (same rows + step_rewards + aggregates)
"""
import json, os, sys, time, glob
import torch
from transformers import AutoTokenizer
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards

MODEL_PATH = "/root/skywork-prm"
INPUT_DIR  = "/root/prm_inputs"
OUTPUT_DIR = "/root/prm_outputs"
BATCH = 4  # reasonable on a 24GB 4090

print("Loading tokenizer + model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
mdl = PRM_MODEL.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16).eval()
print("Model loaded.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

@torch.no_grad()
def score_batch(rows):
    """Each row dict has 'problem' and 'response'. Returns list of step_rewards lists."""
    processed = [prepare_input(r["problem"], r["response"], tokenizer=tok, step_token="\n") for r in rows]
    input_ids, _, reward_flags = zip(*processed)
    input_ids, attn_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, tok.pad_token_id)
    input_ids = input_ids.to("cuda"); attn_mask = attn_mask.to("cuda")
    _, _, rewards = mdl(input_ids=input_ids, attention_mask=attn_mask, return_probs=True)
    step_rewards = derive_step_rewards(rewards, reward_flags)
    return step_rewards

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
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i+BATCH]
        try:
            srs = score_batch(batch)
        except Exception as e:
            print(f"  [{cell}] batch {i} error: {str(e)[:120]}")
            srs = [None] * len(batch)
        for r, sr in zip(batch, srs):
            r2 = dict(r)
            if sr is None:
                r2["step_rewards"] = None
                r2["prm_min"] = None; r2["prm_mean"] = None; r2["prm_last"] = None
            else:
                r2["step_rewards"] = [float(x) for x in sr]
                r2["prm_min"] = float(min(sr)) if sr else None
                r2["prm_mean"] = float(sum(sr)/len(sr)) if sr else None
                r2["prm_last"] = float(sr[-1]) if sr else None
            out_rows.append(r2)
    with open(out_fp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    total_rows += len(out_rows)
    elapsed = time.time() - t0
    print(f"  {cell}: {len(out_rows)} rows in {elapsed:.1f}s ({len(out_rows)/elapsed:.1f}/s)")

print(f"\nTotal: {total_rows} rows in {time.time()-t_start:.1f}s")
