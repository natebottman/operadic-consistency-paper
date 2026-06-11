#!/usr/bin/env bash
# Regenerate every paper number from the cached CSVs in data/per_question/
# and data/baselines/. No LLM inference is performed.
#
# Outputs land in results/ and should match the paper exactly (modulo
# floating-point noise at the 4th decimal).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

echo "[1/7] Per-question regressions (Tables 1, 4, 5)..."
python -m analysis.per_question_regressions > results/per_question_regressions.txt

echo "[2/7] Thinking-model regressions (Table 2, §3.2)..."
python -m analysis.thinking_regressions > results/thinking_regressions.txt

echo "[3/7] LOO accuracy prediction (Table 3, App A)..."
python -m analysis.loo_accuracy > results/loo.txt

echo "[4/7] Number-fallback diagnostic (App A) and teaser figure..."
python -m analysis.number_fallback_diagnostic > results/number_fallback_diagnostic.txt
python -m analysis.plot_teaser

echo "[5/7] Alternative OC scorers (App A)..."
python -m analysis.oc_alternative_scorers > results/oc_alternative_scorers.txt

echo "[6/7] Selective prediction (AUARC + AUROC, §3.1 + §3.2)..."
python -m analysis.selective_prediction > results/selective_prediction.txt

echo "[7/7] Cross-model Pearson r per baseline (App A)..."
python -m analysis.cross_model_baselines > results/cross_model_baselines.txt

echo
echo "Done. Outputs in results/. Compare against the paper:"
ls -la results/
