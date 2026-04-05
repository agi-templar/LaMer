#!/usr/bin/env python3
"""Run inference with a trained LaMer model.

Examples:
    # Generate from a file
    uv run python scripts/run_inference.py \
        --model_path checkpoints/yelp_p2n/il/final \
        --input_file test.src \
        --output_file test.output

    # Interactive mode
    uv run python scripts/run_inference.py \
        --model_path checkpoints/yelp_p2n/il/final \
        --interactive

    # Single sentence
    uv run python scripts/run_inference.py \
        --model_path checkpoints/yelp_p2n/il/final \
        --text "the food was really great and i loved it"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LaMer.model.bart_trainer import BartStyleTransfer


def main():
    parser = argparse.ArgumentParser(description='LaMer Inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    model = BartStyleTransfer.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    if args.text:
        results = model.generate([args.text], args.num_beams, args.max_length)
        print(f"Input:  {args.text}")
        print(f"Output: {results[0]}")

    elif args.input_file:
        with open(args.input_file) as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"Generating for {len(sentences)} sentences...")
        results = model.generate(sentences, args.num_beams, args.max_length)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                for r in results:
                    f.write(r + '\n')
            print(f"Results saved to {args.output_file}")
        else:
            for src, out in zip(sentences, results):
                print(f"  {src}  ->  {out}")

    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            text = input("Input: ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if text:
                result = model.generate([text], args.num_beams, args.max_length)
                print(f"Output: {result[0]}\n")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
