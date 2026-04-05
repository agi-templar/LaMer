#! /usr/bin/env python3
# coding=utf-8

# Licensed under the Apache License, Version 2.0

"""TST metrics: ACC, BLEU, SIM, FL, i-PINC, GM (Section 3.2).

Usage:
    python -m LaMer.evaluation.metrics \
        --source_file test.src \
        --output_file test.output \
        --reference_file test.ref \
        --classifier_path checkpoints/style_classifier \
        --task sentiment
"""

import argparse
import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    outputs: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """Compute average BLEU (1 to max_n) against references.

    BLEU-ref as in the paper: average n-gram BLEU from n=1 to n=4.

    :param outputs: generated sentences
    :param references: list of reference lists per output
    :param max_n: max n-gram order
    :return: average BLEU score
    """
    def sentence_bleu_n(output_tokens, ref_token_lists, n):
        output_ng = _ngrams(output_tokens, n)
        max_ref_ng = Counter()
        for ref_tokens in ref_token_lists:
            ref_ng = _ngrams(ref_tokens, n)
            for ng in ref_ng:
                max_ref_ng[ng] = max(max_ref_ng[ng], ref_ng[ng])
        clipped = sum(
            min(count, max_ref_ng[ng])
            for ng, count in output_ng.items()
        )
        total = sum(output_ng.values())
        return clipped / max(total, 1)

    scores_per_n = {n: [] for n in range(1, max_n + 1)}

    for out, refs in zip(outputs, references):
        out_tokens = out.lower().split()
        ref_token_lists = [r.lower().split() for r in refs]

        for n in range(1, max_n + 1):
            if len(out_tokens) >= n:
                scores_per_n[n].append(
                    sentence_bleu_n(out_tokens, ref_token_lists, n)
                )
            else:
                scores_per_n[n].append(0.0)

    avg_bleu_per_n = [np.mean(scores_per_n[n]) for n in range(1, max_n + 1)]
    return float(np.mean(avg_bleu_per_n))


def compute_sim(
    sources: List[str],
    outputs: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
) -> float:
    """Compute SIM: average cosine similarity between source and output.

    Measures content preservation at the sentence level.
    """
    encoder = SentenceTransformer(model_name)
    src_embs = encoder.encode(sources)
    out_embs = encoder.encode(outputs)

    sims = []
    for s, o in zip(src_embs, out_embs):
        cos = np.dot(s, o) / (np.linalg.norm(s) * np.linalg.norm(o) + 1e-8)
        sims.append(cos)
    return float(np.mean(sims))


def compute_fluency(
    outputs: List[str],
    model_name: str = 'gpt2',
) -> float:
    """Compute FL: average perplexity using GPT-2.

    Lower perplexity = more fluent text.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ppls = []
    for text in tqdm(outputs, desc="Computing perplexity"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model_out = model(**inputs, labels=inputs['input_ids'])
        ppl = math.exp(min(model_out.loss.item(), 100))
        ppls.append(ppl)

    return float(np.mean(ppls))


def compute_accuracy(
    outputs: List[str],
    target_label: int,
    classifier_path: Optional[str] = None,
) -> float:
    """Compute ACC: style transfer accuracy.

    Uses a pre-trained style classifier. If no classifier_path is provided,
    returns 0.0 with a warning.

    :param outputs: generated sentences
    :param target_label: the target style label (0 or 1)
    :param classifier_path: path to pre-trained classifier
    :return: accuracy (fraction of outputs classified as target style)
    """
    if classifier_path is None:
        print("WARNING: No classifier provided. Train a TextCNN/BERT "
              "classifier on your task data for ACC computation.")
        return 0.0

    from transformers import pipeline
    classifier = pipeline('text-classification', model=classifier_path)
    correct = 0
    for text in tqdm(outputs, desc="Computing ACC"):
        pred = classifier(text[:512])[0]
        pred_label = int(pred['label'].split('_')[-1])
        if pred_label == target_label:
            correct += 1
    return correct / max(len(outputs), 1)


def compute_i_pinc(
    sources: List[str],
    outputs: List[str],
    max_n: int = 4,
) -> float:
    """Compute i-PINC: inverse paraphrase in n-gram changes.

    Measures net transfer ability by counting n-gram overlap after removing
    shared tokens. i-PINC = 0 means output is an exact copy of source.
    """
    scores = []
    for src, out in zip(sources, outputs):
        src_tokens = src.lower().split()
        out_tokens = out.lower().split()

        pinc_per_n = []
        for n in range(1, max_n + 1):
            src_ng = _ngrams(src_tokens, n)
            out_ng = _ngrams(out_tokens, n)
            if sum(out_ng.values()) == 0:
                pinc_per_n.append(1.0)
                continue
            overlap = sum((src_ng & out_ng).values())
            total = sum(out_ng.values())
            pinc_per_n.append(1.0 - overlap / total)

        scores.append(np.mean(pinc_per_n))

    return float(np.mean(scores))


def compute_gm(acc: float, bleu: float) -> float:
    """Compute GM: geometric mean of ACC and BLEU."""
    return math.sqrt(max(acc, 0) * max(bleu, 0))


def run_full(
    sources: List[str],
    outputs: List[str],
    references: Optional[List[List[str]]] = None,
    target_label: int = 1,
    classifier_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run all metrics and return a results dictionary."""
    results = {}

    results['ACC'] = compute_accuracy(outputs, target_label, classifier_path)

    if references:
        results['BLEU'] = compute_bleu(outputs, references)
    else:
        results['BLEU'] = 0.0

    results['SIM'] = compute_sim(sources, outputs)
    results['FL'] = compute_fluency(outputs)
    results['i-PINC'] = compute_i_pinc(sources, outputs)
    results['GM'] = compute_gm(results['ACC'], results['BLEU'])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LaMer TST Metrics')
    parser.add_argument('--source_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--reference_file', type=str, default=None)
    parser.add_argument('--classifier_path', type=str, default=None)
    parser.add_argument('--target_label', type=int, default=1)
    args = parser.parse_args()

    with open(args.source_file) as f:
        sources = [line.strip() for line in f]
    with open(args.output_file) as f:
        outputs = [line.strip() for line in f]

    references = None
    if args.reference_file:
        with open(args.reference_file) as f:
            references = [[line.strip()] for line in f]

    results = run_full(
        sources, outputs, references,
        target_label=args.target_label,
        classifier_path=args.classifier_path,
    )

    print("\n=== LaMer TST Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
