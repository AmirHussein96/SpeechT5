#!/usr/bin/env python3
import argparse
from pathlib import Path


def train_dev_split(
    data: Path,
    output_dir: Path,
    ratio: float = 0.95,
):
    """
    Split the raw text data into training and development sets.
    """
    with open(data, "r") as f:
        lines = f.readlines()

    n = len(lines)
    n_train = int(n * ratio)

    train_texts = lines[:n_train]
    dev_texts = lines[n_train:]

    with open(output_dir / "text_train.txt", "w") as f:
        f.writelines(train_texts)  # Write lines directly without adding extra newlines
    
    with open(output_dir / "text_valid.txt", "w") as f:
        f.writelines(dev_texts)  # Write lines directly without adding extra newlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data", type=Path, required=True,
                        help="Path to the data tsv file.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Path to the output tsv file.")
    parser.add_argument("-r", "--ratio", type=float, default=0.95,
                        help="Ratio of the training set.")

    args = parser.parse_args()

    train_dev_split(
        data=args.data,
        output_dir=args.output_dir,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()
