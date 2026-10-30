# Operadic consistency

Companion code for *Operadic consistency: a label-free signal for compositional
reasoning failures in LLMs* (NeurIPS 2026; [arXiv:2606.13649](https://arxiv.org/abs/2606.13649)).
Implements operadic consistency as a per-question
selection signal, runs all baselines reported in the paper, and reproduces
every table and figure. The operad-theoretic background is developed in a
companion paper, *Operads for compositional reasoning in LLMs*
([arXiv:2606.13634](https://arxiv.org/abs/2606.13634)).

## What's here

| Directory      | Contents                                                        |
|----------------|-----------------------------------------------------------------|
| `experiments/` | Scripts that call LLM APIs to produce raw signal data           |
| `analysis/`    | Statistical analysis and plotting (no LLM calls)                |
| `data/`        | Pre-computed per-question signals -- enough to re-run all stats |
| `results/`     | Auto-generated tables and figures matching the paper            |
| `scripts/`     | Reproduction helpers                                            |
| `env/`         | Python requirements + `.env` template                           |

## Quick reproduction (no API keys, no GPU, ~30 seconds)

This regenerates every number in the paper from the cached signal data
under `data/`. No LLM inference is performed.

```bash
git clone https://github.com/natebottman/operadic-consistency-paper.git
cd operadic-consistency-paper
pip install -r env/requirements.txt
bash scripts/reproduce_paper.sh
```

Output lands in `results/`. Compare against the paper:

| `results/` file                       | Paper section                                          |
|---------------------------------------|--------------------------------------------------------|
| `teaser.pdf`                          | Figure 1                                               |
| `fig1.pdf`                            | Figure 3 (signal vs. accuracy, per dataset)            |
| `per_question_regressions.txt`        | Tables 1, 4, 5 (3-way, P(True) sensitivity, dec-aware) |
| `thinking_regressions.txt`            | Table 2 left (thinking-model regression, §3.2)         |
| `selective_prediction.txt`            | Tables 1, 2 right (AUARC/AUROC selective prediction)   |
| `cross_model_baselines.txt`           | Table 7 (cross-model Pearson r per baseline, App. B.13) |
| `loo.txt`                             | Table 3 (LOO accuracy prediction, App. B.9)            |
| `number_fallback_diagnostic.txt`      | App. B.14 (DROP scorer sensitivity, hit rates)         |
| `oc_alternative_scorers.txt`          | App. B.14 (alternative OC scorers: LLM-judge sensitivity) |

## Full reproduction from scratch (API keys required, several hundred USD, several days)

If you want to re-run inference end-to-end:

1. Copy `env/.env.example` to `env/.env` and fill in:
   ```
   TOGETHER_API_KEY=...
   OPENROUTER_API_KEY=...
   EXP_ROOT=$PWD/experiments
   DATA_ROOT=$PWD/data
   ```
2. Run `bash scripts/run_all_inference.sh`. This calls each `experiments/*/run.py`
   in sequence. Total cost is several hundred USD on Together + OpenRouter at
   the prices we paid. Breakdown:
   - Non-thinking OC (12 models × 4 datasets × ~500 questions × 24 calls)
   - Canonical CoT-SC (12 models × 4 datasets × ~500 questions × 20 calls)
   - Thinking OC (5 models × 4 datasets × 100 questions × 13–23 calls)
   - P(True), decomposed-SC, NLI, PRMs (much cheaper; PRMs and NLI are GPU-only after raw data exists)
3. Run `python scripts/build_csvs.py` to materialize `data/per_question/*.csv`
   from the raw outputs, then `python scripts/merge_cot_sc_into_csv.py` to
   join the canonical CoT-SC short answers into `nonthinking.csv`.
4. Run `bash scripts/reproduce_paper.sh` to re-derive the tables and figures.

GPU requirements: NLI and both PRMs run inference-only on a single 24 GB GPU
(we used a single RTX 4090). No training is performed.

## Data schema

`data/per_question/nonthinking.csv` columns:

| column          | type            | description                                                  |
|-----------------|-----------------|--------------------------------------------------------------|
| `model`         | str             | one of the 12 model identifiers                              |
| `dataset`       | str             | hotpotqa / musique / strategyqa / drop                       |
| `question_id`   | str             | dataset-native ID                                            |
| `op_direct`     | str             | greedy direct answer (T=0)                                   |
| `op_expansion`  | str             | greedy decomposed answer (T=0)                               |
| `nv_answers`    | str (JSON list) | K=10 naive direct-answer SC samples (T=0.7); App. A.4 row    |
| `cot_shorts`    | str (JSON list) | K=10 LLM-extracted short answers from canonical CoT-SC chains (T=0.7, zero-shot CoT; produced by `experiments/cot_sc/run_full.py`; merged via `scripts/merge_cot_sc_into_csv.py`) |
| `ground_truth`  | str             | gold answer                                                  |
| `aliases`       | str (JSON list) | annotator aliases (DROP only; else `[]`)                     |

`data/per_question/thinking.csv` is similar but per (model, question) and
includes the extracted `q1`, `a1`, `q2` from the CoT trace.

`data/baselines/*.csv` are simple `(model, dataset, question_id, signal_value)` triples.

Full schema documentation lives in `data/README.md`.

## Models evaluated

Non-thinking (§3.1, 12 models):

| Identifier         | Provider     | Parameters |
|--------------------|--------------|------------|
| `llama8b`          | Together     | 8B         |
| `gemma3n_4b`       | Together     | 4B         |
| `qwen7b`           | Together     | 7B         |
| `liquidai_24b`     | Together     | 24B        |
| `llama33_70b`      | Together     | 70B        |
| `qwen3_235b`       | Together     | 235B       |
| `deepseek_v3_1`    | Together     | 671B       |
| `cogito_671b`      | Together     | 671B       |
| `gpt_4o_mini`      | OpenRouter   | (closed)   |
| `gemini_2_5_flash` | OpenRouter   | (closed)   |
| `llama_4_maverick` | OpenRouter   | 17B×128E   |
| `mistral_large_2`  | OpenRouter   | 123B       |

Thinking (§3.2, 5 models): `cogito-v2.1`, `deepseek-r1`, `glm-5.1`,
`kimi-k2.6`, `minimax-m2.7` -- all via Together.

## Reproducibility caveats

- **API drift**: closed-source models update silently. Numbers reported in the
  paper used snapshots from March–April 2026; later runs may differ.
- **Together / OpenRouter availability**: open-weight model availability on
  serverless platforms changes. As of release, all twelve non-thinking and
  five thinking models above are reachable via the listed providers.
- **Random seeds**: `analysis/` is fully deterministic. `experiments/` calls at
  `T=0.7` (SC sampling) are non-deterministic by construction; we cache one
  full draw of K=10 samples per (model, question) and ship those in
  `data/per_question/nonthinking.csv`.

## Citation

If you use this code or build on operadic consistency, please cite:

```bibtex
@inproceedings{bottman_liu_richardson:empirics,
  title         = {Operadic consistency: a label-free signal for compositional
                   reasoning failures in {LLM}s},
  author        = {Bottman, Nathaniel and Liu, Yinhong and Richardson, Kyle},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2026},
  note          = {arXiv:2606.13649}
}

@misc{bottman_richardson:theory,
  title         = {Operads for compositional reasoning in {LLM}s},
  author        = {Bottman, Nathaniel and Richardson, Kyle},
  year          = {2026},
  note          = {arXiv:2606.13634}
}
```

## License

Code in this repository is licensed under MIT (see [LICENSE](LICENSE)).
Figures and tabular results (`results/*.pdf`, `results/*.png`, `results/*.txt`)
match the paper and are additionally available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for reuse with
attribution to match the paper's arXiv license.
