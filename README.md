# LaMer: Non-Parallel Text Style Transfer with Self-Parallel Supervision

[![ICLR 2022](https://img.shields.io/badge/ICLR-2022-blue.svg)](https://openreview.net/forum?id=tB5MNjkFCf8)
[![arXiv](https://img.shields.io/badge/arXiv-2204.08123-b31b1b.svg)](https://arxiv.org/abs/2204.08123)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"Non-Parallel Text Style Transfer with Self-Parallel Supervision"** (ICLR 2022).

> **Authors:** Ruibo Liu, Chongyang Gao, Chenyan Jia, Guangxuan Xu, Soroush Vosoughi

## Overview

LaMer is a self-supervised text style transfer (TST) framework built on large-scale language models. Unlike existing methods that randomly map source-to-target sentences, LaMer **mines roughly parallel expressions** hidden within non-parallel datasets, then uses these as self-supervision for training.

### Three-Step Pipeline

```
Non-Parallel Data
       |
       v
  ┌──────────────────────────────────────┐
  │  Step 1: Mining Parallel Sentences   │
  │                                      │
  │  Random (RD) ─── baseline            │
  │  Sentence Embedding (S-Emb.) ─── cosine similarity    │
  │  S-Emb. + Scene Alignment Score (SAS) ─── best        │
  └──────────────────────────────────────┘
       |
       v   Roughly parallel pairs (src → tgt)
  ┌──────────────────────────────────────┐
  │  Step 2: Conditional MLE Training    │
  │                                      │
  │  Fine-tune BART on mined pairs       │
  │  Loss only on target tokens          │
  └──────────────────────────────────────┘
       |
       v   Pre-trained BART model
  ┌──────────────────────────────────────┐
  │  Step 3: Imitation Learning (IL)     │
  │                                      │
  │  REINFORCE with contrastive loss     │
  │  Expert (best pair) vs Amateur       │
  │  d_SEM (semantic) + d_PSV (scene)    │
  └──────────────────────────────────────┘
       |
       v
  Final LaMer Model
```

### Key Ideas

- **Scene Alignment Score (SAS):** Extracts scene graphs from sentences and computes an F-beta score over shared entities to find content-preserving parallels
- **Conditional MLE:** Uses BART to learn source→target generation from mined pairs
- **Contrastive IL:** Further refines the model by contrasting expert demonstrations (best alignments) against amateur ones using REINFORCE

## Project Structure

```
LaMer/
├── LaMer/                      # Main package
│   ├── data/                   # Step 1: Data alignment
│   │   ├── config.py           # Task-specific hyperparameters
│   │   ├── data_aligner.py     # DataAligner: Random, LM, LM+KG alignment
│   │   ├── scene_graph.py      # Scene graph extraction & SAS computation
│   │   └── utils.py            # Text normalization, batching utilities
│   ├── model/                  # Steps 2 & 3: Training
│   │   ├── bart_trainer.py     # BartStyleTransfer: MLE training & generation
│   │   └── imitation_learning.py  # ImitationLearningTrainer: REINFORCE + contrastive
│   └── evaluation/             # Evaluation metrics
│       └── metrics.py          # ACC, BLEU, SIM, FL, i-PINC, GM
├── scripts/                    # Runnable scripts
│   ├── download_data.py        # Dataset download instructions
│   ├── run_alignment.py        # Step 1: Mine parallel sentences
│   ├── run_train.py            # Steps 2+3: MLE + IL training
│   └── run_inference.py        # Generate style-transferred text
├── test/                       # Unit tests
├── setup.py                    # Package installation
└── README.md
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (recommended for training)

### Install

```bash
# Clone the repository
git clone https://github.com/DapangLiu/LaMer.git
cd LaMer

# Install the package
pip install -e ".[dev]"

# Install SceneGraphParser (for LM+KG alignment)
pip install SceneGraphParser

# Download spaCy model (used by SceneGraphParser)
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Prepare Data

```bash
# See dataset download instructions
python scripts/download_data.py --dataset yelp
```

**Yelp Sentiment** (Shen et al., 2017): Download from [language-style-transfer](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp) and place in `assets/yelp/raw/`.

**GYAFC Formality** (Rao & Tetreault, 2018): Request access from [GYAFC-corpus](https://github.com/raosudha89/GYAFC-corpus) and place in `data/GYAFC_Corpus/`.

**AllSides Political Stance**: See the paper for collection details.

### 2. Mine Parallel Sentences (Step 1)

```bash
# Random baseline alignment
python scripts/run_alignment.py --task yelp_pos2neg --method random

# Sentence Embedding alignment
python scripts/run_alignment.py --task yelp_pos2neg --method lm

# S-Emb. + Scene Alignment Score (recommended)
python scripts/run_alignment.py --task yelp_pos2neg --method lm_kg
```

Available tasks: `yelp_pos2neg`, `yelp_neg2pos`, `formal_music_f2i`, `formal_music_i2f`, `formal_family_f2i`, `formal_family_i2f`, `allsides_l2r`, `allsides_r2l`

### 3. Train the Model (Steps 2 + 3)

```bash
# MLE training only (Step 2)
python scripts/run_train.py \
    --aligned_data yelp_lm_kg_tok50_top06_beta001/yelp_p2n_lm_kg_tok50_top06_beta001.csv \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --batch_size 16 --lr 5e-5

# MLE + Imitation Learning (Steps 2 + 3)
python scripts/run_train.py \
    --aligned_data yelp_lm_kg_tok50_top06_beta001/yelp_p2n_lm_kg_tok50_top06_beta001.csv \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --batch_size 16 --lr 5e-5 \
    --do_il --il_epochs 3 --alpha 0.4 --delta 0.5
```

**Alpha values by task** (controls d_Order vs d_Exist in scene preservation):
- Sentiment: `--alpha 0.4`
- Formality: `--alpha 0.3`
- Political Stance: `--alpha 0.1`

### 4. Generate Style-Transferred Text

```bash
# From file
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --input_file test.src \
    --output_file test.output

# Single sentence
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --text "the food was really great and i loved it"

# Interactive mode
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --interactive
```

### 5. Evaluate

```bash
python -m LaMer.evaluation.metrics \
    --source_file test.src \
    --output_file test.output \
    --reference_file test.ref \
    --classifier_path checkpoints/style_classifier \
    --target_label 1
```

Metrics reported:
| Metric | Description |
|--------|-------------|
| **ACC** | Style transfer accuracy (via pre-trained classifier) |
| **BLEU** | Average n-gram BLEU (1-4) against human references |
| **SIM** | Sentence-level cosine similarity (content preservation) |
| **FL** | GPT-2 perplexity (fluency; lower = better) |
| **i-PINC** | Inverse paraphrase — net style change beyond copying |
| **GM** | Geometric mean of ACC and BLEU |

## Recommended Hyperparameters (Paper Table 1 / Figure 3)

| Task | Alignment k | Alignment p | SAS beta | IL alpha |
|------|------------|------------|----------|----------|
| Sentiment (Yelp) | 200 | 0.6 | 0.01 | 0.4 |
| Formality (GYAFC) | 500 | 0.4 | 0.01 | 0.3 |
| Political Stance | 500 | 0.3 | 0.01 | 0.1 |

## Reproducing Paper Results (Table 1)

```bash
# Full pipeline for Yelp pos→neg
python scripts/run_alignment.py --task yelp_pos2neg --method lm_kg
python scripts/run_train.py \
    --aligned_data yelp_lm_kg_tok50_top06_beta001/yelp_p2n_lm_kg_tok50_top06_beta001.csv \
    --output_dir checkpoints/yelp_p2n \
    --epochs 5 --do_il --il_epochs 3 --alpha 0.4
python scripts/run_inference.py \
    --model_path checkpoints/yelp_p2n/il/final \
    --input_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output
python -m LaMer.evaluation.metrics \
    --source_file assets/yelp/raw/test.pos \
    --output_file results/yelp_p2n.output \
    --reference_file assets/yelp/raw/reference.1
```

## Citation

```bibtex
@inproceedings{liu2022lamer,
  title={Non-Parallel Text Style Transfer with Self-Parallel Supervision},
  author={Liu, Ruibo and Gao, Chongyang and Jia, Chenyan and Xu, Guangxuan and Vosoughi, Soroush},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
