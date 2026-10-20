#!/usr/bin/env python3
"""Prediction script for VirophageFinder.

Predicts virophage contigs from input FASTA files.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virophagefinder.inference.predict import VirophagePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict virophage contigs using VirophageFinder"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input FASTA file containing contigs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file path (default: predictions.csv)",
    )
    parser.add_argument(
        "--output_fasta",
        type=str,
        default=None,
        help="Optional output FASTA for predicted virophage contigs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=500,
        help="Minimum contig length to process (default: 500)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("VirophageFinder Prediction")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print()

    # Create predictor
    predictor = VirophagePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    # Run prediction
    df = predictor.predict_fasta(
        fasta_path=args.input,
        output_path=args.output,
        min_length=args.min_length,
        batch_size=args.batch_size,
    )

    # Extract virophage contigs to FASTA if requested
    if args.output_fasta:
        predictor.extract_virophage_contigs(
            fasta_path=args.input,
            output_fasta=args.output_fasta,
            min_probability=args.threshold,
        )

    # Print top predictions
    print("\nTop 10 predictions:")
    print(df.head(10)[["contig_id", "length", "probability", "n_genes"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print(f"Results: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
