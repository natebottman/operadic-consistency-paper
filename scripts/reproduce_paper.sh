#!/usr/bin/env bash
# Regenerate every paper number from the cached CSVs in data/per_question/
# and data/baselines/. No LLM inference is performed.
#
# Outputs land in results/ and should match the paper exactly (modulo
# floating-point noise at the 4th decimal).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

echo "[1/8] Per-question regressions (Tables 1, 4, 5)..."
python -m analysis.per_question_regressions > results/per_question_regressions.txt

echo "[2/8] Thinking-model regressions (Table 2, §3.2)..."
python -m analysis.thinking_regressions > results/thinking_regressions.txt

echo "[3/8] LOO accuracy prediction (Table 3, App B.9)..."
python -m analysis.loo_accuracy > results/loo.txt

echo "[4/8] Number-fallback diagnostic (App B.14) and teaser figure (Figure 1)..."
python -m analysis.number_fallback_diagnostic > results/number_fallback_diagnostic.txt
python -m analysis.plot_teaser

echo "[5/8] Alternative OC scorers (App B.14)..."
python -m analysis.oc_alternative_scorers > results/oc_alternative_scorers.txt

echo "[6/8] Selective prediction (AUARC + AUROC, §3.1 + §3.2)..."
python -m analysis.selective_prediction > results/selective_prediction.txt

echo "[7/8] Cross-model Pearson r per baseline (App B.13)..."
python -m analysis.cross_model_baselines > results/cross_model_baselines.txt

echo "[8/8] Signal-vs-accuracy scatter (Figure 3)..."
python -m analysis.plot_fig1

echo
echo "Done. Outputs in results/. Compare against the paper:"
ls -la results/
