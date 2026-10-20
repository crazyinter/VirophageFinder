#!/usr/bin/env python3
"""Contrastive pre-training script for VirophageFinder."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virophagefinder.utils.io import load_config, ensure_dir
from virophagefinder.training.pretrain import ContrastivePretrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contrastive pre-training for VirophageFinder"
    )
    parser.add_argument(
        "--unlabeled_contigs",
        type=str,
        required=True,
        help="FASTA file containing unlabeled viral contigs for pre-training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_config.yaml",
        help="Path to pre-training configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for pre-trained checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("VirophageFinder Contrastive Pre-training")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    print(f"Config loaded from {args.config}")

    # Create output directory
    ensure_dir(args.output_dir)

    # Create pretrainer
    pretrainer = ContrastivePretrainer(
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run pre-training
    print(f"\nPre-training on: {args.unlabeled_contigs}")
    pretrainer.train(args.unlabeled_contigs)

    print("\n" + "=" * 60)
    print("Pre-training complete!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
