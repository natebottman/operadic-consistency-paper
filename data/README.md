# Data

Pre-computed per-question signals derived from the raw API outputs in
`data/raw/` (which is `.gitignored` -- too large to ship). All paper
numbers can be reproduced from the CSVs here without re-running any LLM
inference.

## Files

```
data/
├── per_question/
│   ├── nonthinking.csv          # 24,624 rows (12 models × 4 datasets × ~500 questions)
│   └── thinking.csv             # 1,878 rows (5 models × 4 datasets × ~94 questions after filtering)
├── baselines/
│   ├── p_true.csv               # P(True) score per (model, dataset, question_id)
│   ├── decomposed_sc.csv        # joint-sampled K=10 decomposed-SC pairwise F1
│   ├── nli.csv                  # DeBERTa-v3-large bidirectional entailment score
│   ├── prm_skywork.csv          # Skywork-o1-Open-PRM-Qwen-2.5-7B (min/mean/last over steps)
│   ├── prm_rlhflow.csv          # RLHFlow Llama-3.1-8B-PRM-Mistral-Data (same)
│   └── llm_judge_oc.csv         # 4 LLM-judge variants of OC scoring (App B)
└── raw/                         # raw API outputs -- .gitignored, see below
```

## Schema

### `per_question/nonthinking.csv`

| column          | type            | description                                  |
|-----------------|-----------------|----------------------------------------------|
| `model`         | str             | model identifier (see top-level README)      |
| `dataset`       | str             | hotpotqa / musique / strategyqa / drop       |
| `question_id`   | str             | dataset-native ID                            |
| `op_direct`     | str             | greedy direct answer (T=0)                   |
| `op_expansion`  | str             | greedy decomposed answer (T=0)               |
| `nv_answers`    | str (JSON list) | K=10 naive direct-answer SC samples (T=0.7)  |
| `cot_shorts`    | str (JSON list) | K=10 LLM-extracted short answers from canonical CoT-SC chains (T=0.7, zero-shot CoT; produced by `experiments/cot_sc/run_full.py`) |
| `ground_truth`  | str             | gold answer                                  |
| `aliases`       | str (JSON list) | annotator aliases (DROP only; else `[]`)     |

### `per_question/thinking.csv`

| column          | type            | description                                  |
|-----------------|-----------------|----------------------------------------------|
| `model`         | str             | one of the 5 thinking models                 |
| `dataset`       | str             | musique / strategyqa / gsm8k / drop          |
| `question_id`   | str             | dataset-native ID                            |
| `factorable`    | bool            | whether the trace decomposed cleanly         |
| `q1`            | str             | extracted first sub-question                 |
| `a1`            | str             | model's answer to q1                         |
| `q2`            | str             | extracted second sub-question template       |
| `decomp_answer` | str             | model's answer to q2 with [A1] := a1         |
| `direct_answer` | str             | greedy direct answer                         |
| `sc_k3`         | str (JSON list) | 3 SC samples at T=0.7                        |
| `sc_k10`        | str (JSON list) | 10 SC samples at T=0.7                       |
| `ground_truth`  | str             | gold answer                                  |
| `aliases`       | str (JSON list) | annotator aliases (DROP only; else `[]`)    |

### `baselines/*.csv`

All six baseline files share the schema:

| column          | type    | description                                          |
|-----------------|---------|------------------------------------------------------|
| `model`         | str     | model identifier                                     |
| `dataset`       | str     | hotpotqa / musique / strategyqa / drop               |
| `question_id`   | str     | dataset-native ID                                    |
| `<signal>`      | float   | the baseline's continuous score                      |

For PRM files (`prm_skywork.csv`, `prm_rlhflow.csv`), three score columns are
present: `prm_min`, `prm_mean`, `prm_last` (aggregations over the per-step
scores along the operadic trace).

`nli.csv` has two score columns: `nli_score` (continuous bidirectional
entailment probability) and `nli_consistent` (the same score binarized
at 0.5).

Row counts per baseline file can slightly exceed the regression n's
reported in the paper: the regressions additionally drop rows where any
other predictor in the same model is missing (e.g. fewer than 10
non-empty CoT-SC samples).

`llm_judge_oc.csv` has four score columns, one per (judge, prompt) variant.
Each column is the binarized OC verdict from that judge: 1.0 if the judge
ruled the (direct, decomposed) pair consistent, 0.0 if not, NaN if the
verdict was unparseable. The four variants are:

| column   | judge model        | prompt        |
|----------|--------------------|---------------|
| `mini_S` | GPT-4o-mini        | strict CoT    |
| `mini_C` | GPT-4o-mini        | categorical CoT (binarized at 'identical' only) |
| `L70_S`  | Llama-3.3-70B      | strict CoT    |
| `L70_C`  | Llama-3.3-70B      | categorical CoT |

## Building the CSVs from raw outputs

If you have the raw `data/raw/` JSON outputs, you can rebuild the CSVs:

```bash
python scripts/build_csvs.py
```

This reads from `data/raw/` and writes the CSVs in this directory.
Idempotent.

## Why no `data/raw/`?

Raw API outputs total several hundred megabytes and aren't necessary to
reproduce any paper number. They are `.gitignored`. If you want them, run
the inference pipeline (`scripts/run_all_inference.sh`) -- it'll regenerate
`data/raw/` from API calls (cost: several hundred USD).
