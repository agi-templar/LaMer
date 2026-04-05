#!/usr/bin/env python3
"""Download datasets for LaMer experiments.

Examples:
    python scripts/download_data.py --dataset yelp
    python scripts/download_data.py --dataset gyafc
    python scripts/download_data.py --dataset all
"""

import argparse
import os
import sys


def download_yelp(data_dir: str = 'assets/yelp/raw') -> None:
    """Download the Yelp sentiment dataset (Shen et al., 2017).

    The Yelp dataset contains ~250k negative and ~380k positive reviews.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    if (os.path.exists(os.path.join(data_dir, 'train.pos'))
            and os.path.exists(os.path.join(data_dir, 'train.neg'))):
        print(f"Yelp data already exists at {data_dir}")
        return

    print("Downloading Yelp sentiment dataset...")
    print("Source: https://github.com/shentianxiao/language-style-transfer")
    print()
    print("Please download manually from:")
    print("  https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp")
    print()
    print(f"Place the following files in {data_dir}/:")
    print("  - train.pos  (positive reviews)")
    print("  - train.neg  (negative reviews)")
    print("  - test.pos   (positive test set)")
    print("  - test.neg   (negative test set)")
    print("  - reference.0 and reference.1 (human references)")


def download_gyafc(data_dir: str = 'data/GYAFC_Corpus') -> None:
    """Download the GYAFC formality dataset (Rao & Tetreault, 2018).

    Requires signing an agreement to access the data.
    Family Relationships domain: ~52k train, ~5k dev, ~2.5k test
    """
    os.makedirs(data_dir, exist_ok=True)

    print("GYAFC Corpus (Grammarly's Yahoo Answers Formality Corpus)")
    print("Source: https://github.com/raosudha89/GYAFC-corpus")
    print()
    print("This dataset requires a license agreement.")
    print("Please follow the instructions at the above URL to request access.")
    print()
    print(f"Place the data in {data_dir}/ with this structure:")
    print("  Family_Relationships/train/formal")
    print("  Family_Relationships/train/informal")
    print("  Entertainment_Music/train/formal")
    print("  Entertainment_Music/train/informal")


def download_allsides(data_dir: str = 'data/allsides') -> None:
    """Info for the AllSides political stance dataset.

    The paper collects 2,298 articles from allsides.com (6/1/2012 - 4/1/2021)
    and splits them into ~56k sentence pairs.
    """
    os.makedirs(data_dir, exist_ok=True)

    print("AllSides Political Stance Dataset")
    print("Source: https://www.allsides.com/story/admin")
    print()
    print("The paper provides links to articles (not raw text) due to TOS.")
    print("Please check the official LaMer repo for the extraction code:")
    print("  https://github.com/DapangLiu/LaMer")
    print()
    print(f"Place the data in {data_dir}/ with this structure:")
    print("  left_output/  (liberal-leaning sentences, one per line)")
    print("  right_output/ (conservative-leaning sentences, one per line)")


def main():
    parser = argparse.ArgumentParser(description='Download LaMer datasets')
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['yelp', 'gyafc', 'allsides', 'all'],
        help='Dataset to download'
    )
    args = parser.parse_args()

    if args.dataset in ('yelp', 'all'):
        download_yelp()
        print()

    if args.dataset in ('gyafc', 'all'):
        download_gyafc()
        print()

    if args.dataset in ('allsides', 'all'):
        download_allsides()


if __name__ == '__main__':
    main()
