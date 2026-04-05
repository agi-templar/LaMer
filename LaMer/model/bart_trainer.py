#! /usr/bin/env python3
# coding=utf-8

# Licensed under the Apache License, Version 2.0
"""Conditional Token-level MLE Training on BART (Section 2.3).

The model takes source-style sentences as input and is trained to generate
target-style parallel sentences found in the mining step. We use BART
(Lewis et al., 2020) as the text-to-text LM backbone.

Usage example:
    uv run python -m LaMer.model.bart_trainer \
        --aligned_data assets/yelp/yelp_lm_kg/aligned.csv \
        --output_dir checkpoints/yelp_p2n \
        --epochs 5 --batch_size 16 --lr 5e-5
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)


class AlignedPairDataset(Dataset):
    """Dataset of (source, target) aligned sentence pairs from mining step."""

    def __init__(
        self,
        data_path: str,
        tokenizer: BartTokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(data_path)
        self.sources = df['source'].astype(str).tolist()
        self.targets = df['target'].astype(str).tolist()

        # Handle cases where target is a JSON list (from LM/LM_KG alignment)
        expanded_src, expanded_tgt = [], []
        for src, tgt in zip(self.sources, self.targets):
            if tgt.startswith('[') and tgt.endswith(']'):
                try:
                    tgt_list = json.loads(tgt.replace("'", '"'))
                    for t in tgt_list:
                        expanded_src.append(src)
                        expanded_tgt.append(str(t))
                except (ValueError, json.JSONDecodeError):
                    expanded_src.append(src)
                    expanded_tgt.append(tgt)
            else:
                expanded_src.append(src)
                expanded_tgt.append(tgt)

        self.sources = expanded_src
        self.targets = expanded_tgt

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.sources[idx]
        tgt = self.targets[idx]

        src_enc = self.tokenizer(
            src,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = tgt_enc['input_ids'].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': labels,
        }


class BartStyleTransfer:
    """BART-based conditional MLE training for text style transfer.

    Per Section 2.3: We choose BART as the LM, concatenate source and target
    with </s><s> tokens, and minimize MLE loss only on the target part.
    """

    def __init__(
        self,
        model_name: str = 'facebook/bart-base',
        device: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def train(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 5e-5,
        warmup_ratio: float = 0.1,
        max_length: int = 128,
        save_every: int = 1,
        eval_data_path: Optional[str] = None,
    ) -> None:
        """Run MLE training on aligned pairs.

        :param data_path: path to the aligned CSV (from mining step)
        :param output_dir: directory to save checkpoints
        :param epochs: number of training epochs
        :param batch_size: training batch size
        :param lr: learning rate
        :param warmup_ratio: fraction of total steps for warmup
        :param max_length: max token length for source/target
        :param save_every: save checkpoint every N epochs
        :param eval_data_path: optional validation data path
        """
        os.makedirs(output_dir, exist_ok=True)

        dataset = AlignedPairDataset(data_path, self.tokenizer, max_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                ckpt_dir = os.path.join(output_dir, f'checkpoint-epoch{epoch + 1}')
                self.save(ckpt_dir)

        self.save(os.path.join(output_dir, 'final'))

    def generate(
        self,
        sentences: List[str],
        num_beams: int = 5,
        max_length: int = 128,
    ) -> List[str]:
        """Generate style-transferred sentences.

        :param sentences: list of source-style sentences
        :param num_beams: beam search width
        :param max_length: max generation length
        :return: list of generated target-style sentences
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for sent in sentences:
                inputs = self.tokenizer(
                    sent, return_tensors='pt', max_length=max_length, truncation=True
                ).to(self.device)
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                )
                decoded = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                results.append(decoded)
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'BartStyleTransfer':
        instance = cls.__new__(cls)
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        instance.tokenizer = BartTokenizer.from_pretrained(path)
        instance.model = BartForConditionalGeneration.from_pretrained(path)
        instance.model.to(instance.device)
        return instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BART MLE Training for LaMer')
    parser.add_argument('--aligned_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    args = parser.parse_args()

    trainer = BartStyleTransfer(model_name=args.model_name)
    trainer.train(
        data_path=args.aligned_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
