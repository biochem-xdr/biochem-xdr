# Data

The dataset files are hosted on HuggingFace due to size constraints.

**Dataset:** https://huggingface.co/datasets/biochem-xdr/biochem_xdr

## Splits

| File | Items | Description |
|---|---|---|
| biochem_xdr_full.json | 7,002 | Complete benchmark |
| biochem_xdr_verified.json | 974 | Multi-signal verified subset |
| biochem_xdr_gold.json | 350 | Expert-verified Gold subset |

## Loading

```python
from datasets import load_dataset
ds = load_dataset("biochem-xdr/biochem_xdr")
```

Or download directly:

```bash
wget https://huggingface.co/datasets/biochem-xdr/biochem_xdr/resolve/main/biochem_xdr_full.json
```

## Format

Each item is a JSON object with these fields:

```json
{
  "id": 5886,
  "question": "...",
  "short_correct_answer": "TXNRD2",
  "short_distractor_1": "GLRX2",
  "short_distractor_2": "NNT",
  "short_distractor_3": "FDXR",
  "crossing_count": 9,
  "answer_format": "short_entity",
  "path_text": "NDUFA5 [enzyme_kinetics] --associated_with_disease--> ...",
  "difficulty_tier": "T3"
}
```

## Tier Distribution (full dataset)

| Tier | Threshold | N | % |
|---|---|---|---|
| T1 | ≥70% aggregate accuracy | 1,547 | 22.1% |
| T2 | 40–70% | 1,488 | 21.3% |
| T3 | 10–40% | 2,159 | 30.8% |
| T4 | <10% | 1,808 | 25.8% |