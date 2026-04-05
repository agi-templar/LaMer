#!/usr/bin/env python3
"""Run data alignment (Step 1 of LaMer pipeline).

Examples:
    # Random alignment
    uv run python scripts/run_alignment.py --task yelp_pos2neg --method random

    # LM (Sentence Embedding) alignment
    uv run python scripts/run_alignment.py --task yelp_pos2neg --method lm

    # LM + KG (Scene Graph) alignment
    uv run python scripts/run_alignment.py --task yelp_pos2neg --method lm_kg
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LaMer.data import DataAligner
from LaMer.data.config import Config


def main():
    parser = argparse.ArgumentParser(description='LaMer Data Alignment')
    parser.add_argument(
        '--task', type=str, required=True,
        choices=list(Config.keys()),
        help='Task name (e.g., yelp_pos2neg, formal_music_f2i, allsides_l2r)'
    )
    parser.add_argument(
        '--method', type=str, required=True,
        choices=['random', 'lm', 'lm_kg'],
        help='Alignment method'
    )
    parser.add_argument(
        '--num_samples', type=int, default=-1,
        help='Number of samples to use (-1 = all)'
    )
    parser.add_argument(
        '--no_cache', action='store_true',
        help='Rebuild embeddings from scratch'
    )
    args = parser.parse_args()

    config = Config[args.task]
    aligner = DataAligner(
        config, args.method,
        use_cache=not args.no_cache,
        num_samples=args.num_samples
    )

    print(f"\nSource sentences: {len(aligner.get_src_data())}")
    print(f"Target sentences: {len(aligner.get_tgt_data())}")

    if args.method == 'random':
        df = aligner.align_by_random(
            topk=config.random_topk,
            output_root_dir=config.random_root_name,
            output_file_name=config.random_cache_file,
        )
    elif args.method == 'lm':
        df = aligner.align_by_LM(
            topp=config.lm_topp,
            topk=config.lm_topk,
            output_root_dir=getattr(config, 'lm_root_name', 'lm_output'),
            output_file_name=getattr(config, 'lm_cache_file', 'lm_aligned.csv'),
        )
    elif args.method == 'lm_kg':
        df = aligner.align_by_LM_KG(
            output_root_dir=config.lm_kg_root_name,
            output_file_name=config.lm_kg_cache_file,
        )

    print(f"\nAlignment complete! {len(df)} pairs generated.")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample pairs:")
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
