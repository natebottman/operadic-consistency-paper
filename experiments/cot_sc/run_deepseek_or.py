"""Backfill: rerun DeepSeek-V3.1 CoT-SC on OpenRouter.

Together's serverless tier no longer serves deepseek-ai/DeepSeek-V3.1
(HTTP 400 'Unable to access non-serverless model'). OpenRouter still
serves it as deepseek/deepseek-chat-v3.1, so we route DeepSeek through
OR for the SC reproduction. Methodology note: DeepSeek's original OC data
was generated on Together; the SC reproduction uses OR because of the
serverless-availability change. All other models keep their original
provider (option B).

Output: data/raw/exp_cot_sc/{dataset}/deepseek_v3_1.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_full import (run_cell, load_questions, load_musique_passages,  # noqa
                       load_drop_passages, DATASETS)

# Override: route DeepSeek through OR
MODEL_STEM = "deepseek_v3_1"
PROVIDER   = "openrouter"
MODEL_ID   = "deepseek/deepseek-chat-v3.1"


def main():
    print(f"=== DeepSeek-V3.1 backfill on {PROVIDER} ({MODEL_ID}) ===", flush=True)
    print("Loading passages...", flush=True)
    musique_p = load_musique_passages()
    drop_p    = load_drop_passages()
    print(f"  musique: {len(musique_p)} / drop: {len(drop_p)}", flush=True)

    for ds in DATASETS:
        qs = load_questions(ds, musique_p, drop_p)
        print(f"\n--- {ds} ({len(qs)} questions) ---", flush=True)
        run_cell(MODEL_STEM, PROVIDER, MODEL_ID, ds, qs)


if __name__ == "__main__":
    main()
