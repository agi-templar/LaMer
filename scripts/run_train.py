#!/usr/bin/env python3
"""Run LaMer training pipeline (Step 2 + Step 3).

Examples:
    # Step 2: MLE training only
    python scripts/run_train.py \
        --aligned_data yelp_lm_kg/aligned.csv \
        --output_dir checkpoints/yelp_p2n \
        --epochs 5

    # Step 2 + Step 3: MLE + Imitation Learning
    python scripts/run_train.py \
        --aligned_data yelp_lm_kg/aligned.csv \
        --output_dir checkpoints/yelp_p2n \
        --epochs 5 --do_il --il_epochs 3 --alpha 0.4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from LaMer.model.bart_trainer import BartStyleTransfer
from LaMer.model.imitation_learning import ImitationLearningTrainer


def main():
    parser = argparse.ArgumentParser(description='LaMer Training Pipeline')

    # MLE args
    parser.add_argument('--aligned_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)

    # IL args
    parser.add_argument('--do_il', action='store_true', help='Run imitation learning')
    parser.add_argument('--il_epochs', type=int, default=3)
    parser.add_argument('--il_lr', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='d_Order weight (0.4=sentiment, 0.3=formality, 0.1=political)')
    parser.add_argument('--delta', type=float, default=0.5, help='Contrastive margin')

    args = parser.parse_args()

    # Step 2: MLE Training
    print("=" * 60)
    print("Step 2: Conditional Token-level MLE Training on BART")
    print("=" * 60)

    mle_dir = os.path.join(args.output_dir, 'mle')
    trainer = BartStyleTransfer(model_name=args.model_name)
    trainer.train(
        data_path=args.aligned_data,
        output_dir=mle_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )

    if not args.do_il:
        print("\nMLE training complete. Use --do_il to add imitation learning.")
        return

    # Step 3: Imitation Learning
    print("\n" + "=" * 60)
    print("Step 3: Reinforced Imitation Learning by Contrast")
    print("=" * 60)

    mle_checkpoint = os.path.join(mle_dir, 'final')
    il_dir = os.path.join(args.output_dir, 'il')

    # Prepare expert/amateur demonstrations from aligned data
    df = pd.read_csv(args.aligned_data)
    score_col = 'sas_score' if 'sas_score' in df.columns else 'similarity_score'

    grouped = df.groupby('source')
    sources, experts, amateurs = [], [], []
    for src, group in grouped:
        if len(group) < 2:
            continue
        best_idx = group[score_col].astype(float).idxmax()
        expert = group.loc[best_idx, 'target']
        amateur_list = group.loc[group.index != best_idx, 'target'].tolist()
        sources.append(str(src))
        experts.append(str(expert))
        amateurs.append(str(amateur_list[0]))

    print(f"  Expert/amateur pairs: {len(sources)}")

    il_trainer = ImitationLearningTrainer(
        mle_checkpoint=mle_checkpoint,
        alpha=args.alpha,
        delta=args.delta,
    )
    il_trainer.train(
        sources=sources,
        expert_targets=experts,
        amateur_targets=amateurs,
        output_dir=il_dir,
        epochs=args.il_epochs,
        lr=args.il_lr,
    )

    print("\nFull training pipeline complete!")
    print(f"  MLE model:  {mle_checkpoint}")
    print(f"  IL model:   {os.path.join(il_dir, 'final')}")


if __name__ == '__main__':
    main()
