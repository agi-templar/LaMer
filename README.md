# LaMer: Non-Parallel Text Style Transfer with Self-Parallel Supervision

[![ICLR 2022](https://img.shields.io/badge/ICLR-2022-blue.svg)](https://openreview.net/forum?id=tB5MNjkFCf8)
[![arXiv](https://img.shields.io/badge/arXiv-2204.08123-b31b1b.svg)](https://arxiv.org/abs/2204.08123)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of **"Non-Parallel Text Style Transfer with Self-Parallel Supervision"** (ICLR 2022).

> Ruibo Liu, Chongyang Gao, Chenyan Jia, Guangxuan Xu, Soroush Vosoughi

**TL;DR** -- Non-parallel text style transfer datasets contain *hidden parallelism*. LaMer mines roughly parallel sentence pairs using sentence embeddings and scene graphs, trains BART with MLE on the mined pairs, then refines with contrastive imitation learning. It outperforms eight baselines across sentiment, formality, and political stance transfer.

---

## Overview

### Three-Step Pipeline

```
                       Non-Parallel Data
                             |
                             v
               +-----------------------------+
               |  Step 1: Mining Parallels   |
               |                             |
               |  Random         (baseline)  |
               |  S-Emb.    (cosine sim.)    |
               |  S-Emb.+SAS  (scene graph) |  <-- best
               +-----------------------------+
                             |
                             v
                   Mined parallel pairs
                             |
                             v
               +-----------------------------+
               |  Step 2: MLE on BART        |
               |                             |
               |  Conditional token-level    |
               |  loss on target tokens only |
               +-----------------------------+
                             |
                             v
                   Pre-trained BART
                             |
                             v
               +-----------------------------+
               |  Step 3: Imitation Learning |
               |                             |
               |  REINFORCE + contrastive    |
               |  expert vs amateur demos    |
               +-----------------------------+
                             |
                             v
                      Final LaMer Model
```

### Key Ideas

- **Scene Alignment Score (SAS):** Extract scene graphs from sentences, compute F-beta over shared entities to find content-preserving parallels across styles
- **Conditional MLE on BART:** Fine-tune a pre-trained text-to-text LM on the mined pairs, computing loss only on target tokens
- **Contrastive Imitation Learning:** Refine the model with REINFORCE -- contrast the best alignment (expert) against weaker ones (amateur) using semantic coherence (d_SEM) and scene preservation (d_PSV) distances

---

## Project Structure

```
LaMer/
├── LaMer/                         # Main package
│   ├── data/                      # Step 1: Data alignment
│   │   ├── config.py              #   Task hyperparameters (k, p, beta)
│   │   ├── data_aligner.py        #   DataAligner: Random / LM / LM+KG
│   │   ├── scene_graph.py         #   Scene graph extraction + SAS
│   │   └── utils.py               #   Text normalization, batching
│   ├── model/                     # Steps 2--3: Training
│   │   ├── bart_trainer.py        #   BartStyleTransfer: MLE + generation
│   │   └── imitation_learning.py  #   REINFORCE with contrastive loss
│   └── evaluation/                # Evaluation
│       └── metrics.py             #   ACC, BLEU, SIM, FL, i-PINC, GM
├── scripts/                       # Runnable end-to-end scripts
│   ├── download_data.py           #   Dataset download instructions
│   ├── run_alignment.py           #   Step 1
│   ├── run_train.py               #   Steps 2 + 3
│   └── run_inference.py           #   Generate style-transferred text
├── test/                          # Unit tests
├── setup.py
└── README.md
```

---

## Installation

**Requirements:** Python >= 3.8, PyTorch >= 1.9.0, CUDA recommended

```bash
git clone https://github.com/DapangLiu/LaMer.git
cd LaMer

# Install with all dependencies
pip install -e ".[dev]"

# Scene graph parser (needed for LM+KG alignment)
pip install SceneGraphParser
python -m spacy download en_core_web_sm
```

---

## Quick Start

### Step 0: Prepare Data

```bash
python scripts/download_data.py --dataset yelp
```

Each dataset has specific access requirements:

| Dataset | Source | Destination |
|---------|--------|-------------|
| **Yelp Sentiment** (Shen et al., 2017) | [language-style-transfer](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp) | `assets/yelp/raw/` |
| **GYAFC Formality** (Rao & Tetreault, 2018) | [GYAFC-corpus](https://github.com/raosudha89/GYAFC-corpus) (requires license) | `data/GYAFC_Corpus/` |
| **AllSides Political Stance** | [allsides.com](https://www.allsides.com/story/admin) (see paper for extraction) | `data/allsides/` |

### Step 1: Mine Parallel Sentences

```bash
# Recommended: S-Emb. + Scene Alignment Score
python scripts/run_alignment.py --task yelp_pos2neg --method lm_kg

# Alternatives (for ablation)
python scripts/run_alignment.py --task yelp_pos2neg --method lm       # S-Emb. only
python scripts/run_alignment.py --task yelp_pos2neg --method random   # Random baseline
```

**Supported tasks:**

| Task ID | Direction | Domain |
|---------|-----------|--------|
| `yelp_pos2neg` | Positive to Negative | Sentiment |
| `yelp_neg2pos` | Negative to Positive | Sentiment |
| `formal_music_f2i` | Formal to Informal | Formality (Music) |
| `formal_music_i2f` | Informal to Formal | Formality (Music) |
| `formal_family_f2i` | Formal to Informal | Formality (Family) |
| `formal_family_i2f` | Informal to Formal | Formality (Family) |
| `allsides_l2r` | Left to Right | Political Stance |
| `allsides_r2l` | Right to Left | Political Stance |

### Step 2 + 3: Train

```bash
# MLE only (Step 2)
python scripts/run_train.py \
    --aligned_data <path-to-aligned-csv> \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --batch_size 16 --lr 5e-5

# MLE + Imitation Learning (Steps 2 + 3, recommended)
python scripts/run_train.py \
    --aligned_data <path-to-aligned-csv> \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --batch_size 16 --lr 5e-5 \
    --do_il --il_epochs 3 --alpha 0.4 --delta 0.5
```

> `<path-to-aligned-csv>` is the CSV produced by Step 1, e.g. `yelp_lm_kg_tok50_top06_beta001/yelp_p2n_lm_kg_tok50_top06_beta001.csv`

**Alpha values** (controls d_Order vs d_Exist weight in scene preservation):

| Domain | `--alpha` |
|--------|-----------|
| Sentiment | 0.4 |
| Formality | 0.3 |
| Political Stance | 0.1 |

### Step 4: Generate

```bash
# From a file
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --input_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output

# Single sentence
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --text "the food was really great and i loved it"

# Interactive mode
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --interactive
```

### Step 5: Evaluate

```bash
python -m LaMer.evaluation.metrics \
    --source_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output \
    --reference_file assets/yelp/raw/reference.1 \
    --classifier_path checkpoints/style_classifier \
    --target_label 1
```

| Metric | What it measures | Notes |
|--------|------------------|-------|
| **ACC** | Style transfer accuracy | Requires a pre-trained style classifier |
| **BLEU** | Content preservation | Average n-gram BLEU (n=1..4) against human references |
| **SIM** | Semantic similarity | Cosine similarity between source and output embeddings |
| **FL** | Fluency | GPT-2 perplexity (lower is better) |
| **i-PINC** | Net style change | N-gram change beyond copying; penalizes identity copies |
| **GM** | Overall quality | Geometric mean of ACC and BLEU |

---

## Reproducing Paper Results (Table 1)

### Recommended Hyperparameters

From paper Figure 3 -- the starred settings that best balance ACC and BLEU:

| Task | Alignment k | Alignment p | SAS beta | IL alpha |
|------|:-----------:|:-----------:|:--------:|:--------:|
| Sentiment (Yelp) | 200 | 0.6 | 0.01 | 0.4 |
| Formality (GYAFC) | 500 | 0.4 | 0.01 | 0.3 |
| Political Stance | 500 | 0.3 | 0.01 | 0.1 |

### Full Reproduction Script (Yelp Example)

```bash
# 1. Mine parallel pairs
python scripts/run_alignment.py --task yelp_pos2neg --method lm_kg

# 2. Train (MLE + IL)
python scripts/run_train.py \
    --aligned_data yelp_lm_kg_tok50_top06_beta001/yelp_p2n_lm_kg_tok50_top06_beta001.csv \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --do_il --il_epochs 3 --alpha 0.4

# 3. Generate
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --input_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output

# 4. Evaluate
python -m LaMer.evaluation.metrics \
    --source_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output \
    --reference_file assets/yelp/raw/reference.1
```

For formality or political stance, substitute the task ID, aligned data path, and alpha value from the tables above.

### Expected Results (S-Emb. + SAS, from Table 1)

| Task | ACC | BLEU | GM | i-PINC |
|------|:---:|:----:|:--:|:------:|
| Sentiment | 97.0 | 34.1 | 57.5 | 9.6 |
| Formality | 76.5 | 39.2 | 54.8 | 13.3 |
| Political Stance | 82.7 | 30.5 | 50.2 | 13.6 |

---

## Practical Notes

- **GPU memory:** BART-base training requires ~8 GB VRAM with batch size 16. Reduce `--batch_size` if needed.
- **Alignment speed:** LM+KG alignment with scene graph parsing is the slowest step. For quick experiments, start with `--method lm` or use `--num_samples 10000`.
- **Style classifier for ACC:** The ACC metric requires a pre-trained style classifier. You can train a simple TextCNN or fine-tune BERT on the binary style labels from your training data. Without a classifier, ACC will report 0.
- **SceneGraphParser:** If installation fails, LM-only alignment (`--method lm`) still works well (see paper Table 1, "w/ S-Emb." row).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: sng_parser` | `pip install SceneGraphParser` |
| `OSError: en_core_web_sm not found` | `python -m spacy download en_core_web_sm` |
| CUDA out of memory during training | Reduce `--batch_size` (try 8 or 4) |
| Empty alignment output | Check that data files exist at paths in `LaMer/data/config.py` |
| `KeyError: pos_file_name` | Your task config may use different field names. Check `config.py` for the correct task ID. |

---

## Citation

```bibtex
@inproceedings{liu2022lamer,
  title     = {Non-Parallel Text Style Transfer with Self-Parallel Supervision},
  author    = {Liu, Ruibo and Gao, Chongyang and Jia, Chenyan and Xu, Guangxuan and Vosoughi, Soroush},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
