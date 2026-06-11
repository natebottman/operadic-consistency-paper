# experiments/

Inference scripts that produce the raw signal data (`data/raw/`). Each
subdirectory targets one signal in the paper. None of the analysis numbers
in `results/` requires re-running these scripts -- the cached CSVs in
`data/per_question/` and `data/baselines/` are sufficient. These scripts
are provided for full transparency / re-derivation.

Required environment: copy `env/.env.example` to `env/.env` and fill in
`TOGETHER_API_KEY`, `OPENROUTER_API_KEY`. NLI and PRM scripts also need a
GPU (~24 GB) for inference-only model loading.

## Subdirectories

| Subdir              | Paper section | What it does                                                |
|---------------------|---------------|-------------------------------------------------------------|
| `nonthinking_oc/`   | §3.1          | Greedy direct + decomposed answers + K=10 naive-SC samples  |
| `cot_sc/`           | §3.1          | Canonical K=10 chain-of-thought self-consistency (Wang et al.) |
| `thinking_oc/`      | §3.2          | Thinking-trace generation, CoQ extraction, K=3/K=10 SC      |
| `p_true/`           | §3.1, App. B  | Kadavath self-evaluation P(True) via top-k logprobs         |
| `decomposed_sc/`    | §3.1, App. B  | Joint-sampled K=10 decomposed SC (the "proper" variant)     |
| `nli/`              | §3.1, App. B  | DeBERTa-v3-large bidirectional entailment                   |
| `prm_skywork/`      | §3.1, App. B  | Skywork-o1-Open-PRM-Qwen-2.5-7B inference                   |
| `prm_rlhflow/`      | App. B        | RLHFlow Llama-3.1-8B-PRM-Mistral-Data inference (robustness) |

## Running

Each `run.py` (or `run_together.py` / `run_openrouter.py`) takes
command-line arguments controlling the model, dataset, and output
location. Outputs land under `data/raw/...` with a layout that the
top-level `scripts/build_csvs.py` knows how to read. See the docstring at
the top of each script for the exact CLI.

A typical full-pipeline invocation is documented in
`scripts/run_all_inference.sh` (cost: several hundred USD on Together +
OpenRouter; see top-level README for breakdown).
