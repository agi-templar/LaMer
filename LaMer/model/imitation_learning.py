#! /usr/bin/env python3
# coding=utf-8

# Licensed under the Apache License, Version 2.0

"""Reinforced Imitation Learning by Contrast (Section 2.4).

After MLE pre-training, we refine the BART model using REINFORCE with a
contrastive loss. The expert demonstration (highest SAS/similarity) is
contrasted against amateur demonstrations.

Usage:
    uv run python -m LaMer.model.imitation_learning \
        --mle_checkpoint checkpoints/yelp_p2n/final \
        --aligned_data assets/yelp/yelp_lm_kg/aligned.csv \
        --output_dir checkpoints/yelp_p2n_il \
        --epochs 3 --alpha 0.4 --delta 0.5
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from LaMer.data.scene_graph import extract_scene_entities


def levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    """Compute Hamming distance (number of differing positions)."""
    return int(np.sum(v1 != v2))


def entity_to_onehot(
    entities: List[str], vocab: List[str]
) -> np.ndarray:
    """Convert entity list to one-hot vector over vocab."""
    vec = np.zeros(len(vocab), dtype=int)
    for e in entities:
        if e in vocab:
            vec[vocab.index(e)] = 1
    return vec


class ImitationLearningTrainer:
    """REINFORCE-based imitation learning for LaMer refinement.

    Implements contrastive loss J_IL (Equation 3) and policy gradient
    (Equation 4) from the paper.
    """

    def __init__(
        self,
        mle_checkpoint: str,
        device: Optional[str] = None,
        alpha: float = 0.4,
        delta: float = 0.5,
    ):
        """
        :param mle_checkpoint: path to MLE-pretrained BART checkpoint
        :param device: computation device
        :param alpha: weight for d_Order vs d_Exist in d_PSV
        :param delta: margin in contrastive loss
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BartTokenizer.from_pretrained(mle_checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(mle_checkpoint)
        self.model.to(self.device)

        # Sentence encoder for d_SEM
        self.sem_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.alpha = alpha
        self.delta = delta

    def compute_d_sem(self, generated: str, demonstration: str) -> float:
        """Sequence-level semantic coherence distance (d_SEM).

        Negative cosine similarity between sentence embeddings.
        """
        embs = self.sem_encoder.encode([generated, demonstration])
        cos_sim = np.dot(embs[0], embs[1]) / (
            np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
        )
        return -cos_sim

    def compute_d_psv(
        self, source: str, generated: str, demonstration: str
    ) -> float:
        """Token-level scene preservation distance (d_PSV).

        d_PSV = alpha * d_Order + (1 - alpha) * d_Exist
        """
        src_ents = extract_scene_entities(source)
        gen_ents = extract_scene_entities(generated)
        demo_ents = extract_scene_entities(demonstration)

        if not src_ents:
            return 0.0

        all_ents = list(set(src_ents + gen_ents + demo_ents))
        if not all_ents:
            return 0.0

        # d_Order: Levenshtein on entity sequences
        d_order = levenshtein_distance(gen_ents, demo_ents)
        max_len = max(len(gen_ents), len(demo_ents), 1)
        d_order_norm = d_order / max_len

        # d_Exist: Hamming on union one-hot vectors
        gen_union = entity_to_onehot(list(set(gen_ents) | set(src_ents)), all_ents)
        demo_union = entity_to_onehot(
            list(set(demo_ents) | set(src_ents)), all_ents
        )
        d_exist = hamming_distance(gen_union, demo_union)
        d_exist_norm = d_exist / max(len(all_ents), 1)

        return self.alpha * d_order_norm + (1 - self.alpha) * d_exist_norm

    def compute_psi(
        self, source: str, generated: str, demonstration: str
    ) -> float:
        """Combined distance psi*(.) = d_SEM + d_PSV."""
        d_sem = self.compute_d_sem(generated, demonstration)
        d_psv = self.compute_d_psv(source, generated, demonstration)
        return d_sem + d_psv

    def _decode_greedy(self, input_ids: torch.Tensor) -> str:
        with torch.no_grad():
            output = self.model.generate(
                input_ids, num_beams=1, do_sample=False, max_length=128
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _decode_sample(self, input_ids: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Sampling decoding for exploration. Returns text and log prob sum."""
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                max_length=128,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)

        log_probs = []
        for step_idx, step_scores in enumerate(output.scores):
            probs = F.log_softmax(step_scores[0], dim=-1)
            token_id = output.sequences[0, step_idx + 1]
            log_probs.append(probs[token_id])
        log_prob_sum = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)

        return text, log_prob_sum

    def train(
        self,
        sources: List[str],
        expert_targets: List[str],
        amateur_targets: List[str],
        output_dir: str,
        epochs: int = 3,
        lr: float = 1e-5,
    ) -> None:
        """Run imitation learning refinement.

        :param sources: source-style sentences
        :param expert_targets: best parallel targets (s^{tgt+})
        :param amateur_targets: other parallel targets (s^{tgt-})
        :param output_dir: directory to save refined model
        :param epochs: number of IL epochs
        :param lr: learning rate
        """
        os.makedirs(output_dir, exist_ok=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            indices = list(range(len(sources)))
            np.random.shuffle(indices)

            pbar = tqdm(indices, desc=f"IL Epoch {epoch + 1}/{epochs}")
            for idx in pbar:
                src = sources[idx]
                expert = expert_targets[idx]
                amateur = amateur_targets[idx]

                input_ids = self.tokenizer(
                    src, return_tensors='pt', truncation=True, max_length=128
                ).input_ids.to(self.device)

                # Sample from current policy
                sampled_text, log_prob = self._decode_sample(input_ids)

                # Greedy baseline
                greedy_text = self._decode_greedy(input_ids)

                # Contrastive J_IL for sample and greedy
                psi_expert_s = self.compute_psi(src, sampled_text, expert)
                psi_amateur_s = self.compute_psi(src, sampled_text, amateur)
                j_il_sample = max(psi_expert_s - psi_amateur_s + self.delta, 0.0)

                psi_expert_g = self.compute_psi(src, greedy_text, expert)
                psi_amateur_g = self.compute_psi(src, greedy_text, amateur)
                j_il_greedy = max(psi_expert_g - psi_amateur_g + self.delta, 0.0)

                # REINFORCE: advantage = sample - greedy baseline
                advantage = j_il_sample - j_il_greedy
                policy_loss = -log_prob * advantage

                if policy_loss.requires_grad:
                    optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += abs(advantage)
                pbar.set_postfix({'adv': f'{advantage:.4f}'})

            avg_loss = total_loss / max(len(sources), 1)
            print(f"IL Epoch {epoch + 1}: avg |advantage| = {avg_loss:.4f}")

        self.model.save_pretrained(os.path.join(output_dir, 'final'))
        self.tokenizer.save_pretrained(os.path.join(output_dir, 'final'))
        print(f"IL-refined model saved to {output_dir}/final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LaMer Imitation Learning')
    parser.add_argument('--mle_checkpoint', type=str, required=True)
    parser.add_argument('--aligned_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=0.5)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.aligned_data)

    # Group by source, pick best as expert, rest as amateur
    grouped = df.groupby('source')
    sources, experts, amateurs = [], [], []
    score_col = 'sas_score' if 'sas_score' in df.columns else 'similarity_score'
    for src, group in grouped:
        if len(group) < 2:
            continue
        best_idx = group[score_col].astype(float).idxmax()
        expert = group.loc[best_idx, 'target']
        amateur_list = group.loc[group.index != best_idx, 'target'].tolist()
        sources.append(str(src))
        experts.append(str(expert))
        amateurs.append(str(amateur_list[0]))

    trainer = ImitationLearningTrainer(
        mle_checkpoint=args.mle_checkpoint,
        alpha=args.alpha,
        delta=args.delta,
    )
    trainer.train(
        sources=sources,
        expert_targets=experts,
        amateur_targets=amateurs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
    )
