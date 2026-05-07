# BioChem-XDR

A 7,002-item benchmark for cross-domain biochemical reasoning in large language models, grounded in verified multi-hop knowledge graph paths across KEGG, Rhea, and UniProt.

## Dataset

Available on HuggingFace: https://huggingface.co/datasets/biochem-xdr/biochem_xdr

| Split | Items | Description |
|---|---|---|
| biochem_xdr_full.json | 7,002 | Complete benchmark |
| biochem_xdr_verified.json | 974 | Multi-signal verified subset |
| biochem_xdr_gold.json | 350 | Expert-verified Gold subset |

## Installation

```bash
pip install -r requirements.txt
```

API keys go in a `.env` file — see `.env.example`.

## Quick Start

```python
from datasets import load_dataset
ds = load_dataset("biochem-xdr/biochem_xdr")
```

Run MCQ evaluation on all 13 main models:

```bash
python evaluation/run_evaluation.py biochem_xdr_full.json --output evaluation_results/
```

Run on local HuggingFace models:

```bash
python evaluation/run_evaluation_local.py biochem_xdr_full.json --output evaluation_results/
```

Reproduce all paper tables and figures:

```bash
bash scripts/reproduce_all.sh
```

## Repository Structure

`kg/` — Knowledge graph construction from KEGG, Rhea, and UniProt.

`generation/` — Full dataset construction pipeline: path sampling, QA generation, quality filtering, short entity extraction, and distractor regeneration. All prompts used verbatim are in `generation/prompts/`.

`evaluation/` — MCQ and open-ended evaluation runners for all 13 models plus reasoning models. Prompts in `evaluation/prompts/`.

`models/` — Exact model identifiers, API strings, and provider endpoints for all evaluated models.

`tiers/` — Difficulty tier calibration using aggregate accuracy across all 13 models.

`analysis/` — Scripts that reproduce all paper tables and figures from pre-computed result files.

`gold/` — Gold subset selection, expert annotation sheet generation, and human baseline computation.

`scripts/` — Shell scripts that reproduce all main results end-to-end.

`configs/` — Evaluation hyperparameters and path configuration.

## Model Identifiers

All exact model API strings are in `models/model_configs.py`. The 13 main MCQ models are in `MCQ_MODELS`. Reasoning models evaluated on the Gold subset are in `REASONING_MODELS`.

## Prompts

All prompts used verbatim in the paper are checked into the repo:

- `generation/prompts/generation_prompt.txt` — question generation
- `generation/prompts/distractor_regen_prompt.txt` — distractor regeneration
- `evaluation/prompts/mcq_system_prompt.txt` and `mcq_user_prompt.txt` — MCQ evaluation
- `evaluation/prompts/judge_prompt.txt` — open-ended GPT-4o-mini judge

## Pre-computed Results

Pre-computed JSONL result files from the paper are in `results/`. These allow verification of all reported numbers without re-running 91,026 model evaluations.

## Reproducing Results

```bash
# Table 1 — main results
python analysis/main_results.py results/ analysis_outputs/

# Co-occurrence trap (Table B.6)
python analysis/appendix_tables.py results/ cooccurrence_results/ \
  data/biochem_xdr_gold.json configs/tier_aggregate_calibrated.csv analysis_outputs/

# All figures
python figures/generate_all_figures.py results/ data/biochem_xdr_full.json \
  configs/tier_aggregate_calibrated.csv figures_output/
```

## License

CC BY 4.0