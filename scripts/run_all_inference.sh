#!/usr/bin/env bash
# Full inference pipeline: regenerates everything in data/raw/ from API calls.
# Cost: several hundred USD on Together + OpenRouter; several days wall-clock.
#
# Not required for reproducing paper numbers -- the cached CSVs in
# data/per_question/ and data/baselines/ are sufficient. Use
# scripts/reproduce_paper.sh for that. Run this only if you want to re-derive
# the raw signals from scratch.
#
# Required env vars (see env/.env.example):
#   TOGETHER_API_KEY, OPENROUTER_API_KEY
#
# Optional:
#   EXP_ROOT (default: ./data/raw) -- where raw outputs land
#   LEGACY_OC_ASSETS                -- where dataset reference files live
#                                      (gold answers + alias maps); see
#                                      individual run scripts for the
#                                      expected layout

set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present
if [ -f env/.env ]; then
  set -a; . env/.env; set +a
fi

: "${TOGETHER_API_KEY:?Set TOGETHER_API_KEY in env/.env}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in env/.env}"

echo "==================================================================="
echo " Stage 1: Non-thinking OC + naive-SC (12 models x 4 datasets x ~500 questions)"
echo "==================================================================="
python experiments/nonthinking_oc/run_together.py
python experiments/nonthinking_oc/run_openrouter.py
python experiments/nonthinking_oc/run_three_new_models.py

echo "==================================================================="
echo " Stage 2: Canonical CoT-SC (12 models x 4 datasets, K=10 CoT samples + LLM extractor)"
echo "==================================================================="
python experiments/cot_sc/run_full.py
# DeepSeek-V3.1 backfill on OpenRouter (Together removed serverless mid-run)
python experiments/cot_sc/run_deepseek_or.py

echo "==================================================================="
echo " Stage 3: Thinking-model OC + SC (5 models x 4 datasets x 100 questions)"
echo "==================================================================="
python experiments/thinking_oc/run_cot_pilot.py
python experiments/thinking_oc/run_sc_together.py
python experiments/thinking_oc/run_sc_openrouter.py

echo "==================================================================="
echo " Stage 4: P(True) -- 8 Together models + GPT-4o-mini"
echo "==================================================================="
python experiments/p_true/run_together.py
python experiments/p_true/run_openrouter.py

echo "==================================================================="
echo " Stage 5: Decomposition-aware baselines (depend on Stage 1 outputs)"
echo "==================================================================="
echo " 5a: joint-sampled K=10 decomposed SC..."
python experiments/decomposed_sc/run.py

echo " 5b: NLI bidirectional entailment (DeBERTa-v3-large; needs ~4 GB GPU)..."
python experiments/nli/run.py

echo " 5c: Skywork-o1-Open-PRM-Qwen-2.5-7B (needs ~24 GB GPU)..."
python experiments/prm_skywork/run.py

echo " 5d: RLHFlow Llama-3.1-8B-PRM-Mistral-Data (needs ~24 GB GPU)..."
python experiments/prm_rlhflow/run.py

echo "==================================================================="
echo " Stage 6: Materialize CSVs from raw outputs, merge canonical CoT-SC"
echo "==================================================================="
python scripts/build_csvs.py
python scripts/merge_cot_sc_into_csv.py

echo
echo "Done. Now run: bash scripts/reproduce_paper.sh"
