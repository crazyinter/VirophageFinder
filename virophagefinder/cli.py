#!/usr/bin/env python3
"""Command-line interface for VirophageFinder."""

import os
import sys
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """VirophageFinder: Multi-modal deep learning framework for virophage identification."""
    pass


@cli.command()
@click.option("--virophage_genomes", required=True, help="FASTA with virophage genomes")
@click.option("--negative_dir", required=True, help="Directory with negative genomes (L1-L4 subdirs)")
@click.option("--output_dir", required=True, help="Output directory")
@click.option("--seed", default=42, help="Random seed")
def prepare(virophage_genomes, negative_dir, output_dir, seed):
    """Prepare training datasets."""
    from scripts.prepare_data import main as prepare_main
    sys.argv = [
        "prepare_data.py",
        "--virophage_genomes", virophage_genomes,
        "--negative_dir", negative_dir,
        "--output_dir", output_dir,
        "--seed", str(seed),
    ]
    prepare_main()


@cli.command()
@click.option("--unlabeled_contigs", required=True, help="FASTA with unlabeled viral contigs")
@click.option("--config", default="configs/pretrain_config.yaml", help="Config file")
@click.option("--output_dir", required=True, help="Output directory")
@click.option("--device", default="cuda", help="Device")
def pretrain(unlabeled_contigs, config, output_dir, device):
    """Run contrastive pre-training."""
    from scripts.pretrain import main as pretrain_main
    sys.argv = [
        "pretrain.py",
        "--unlabeled_contigs", unlabeled_contigs,
        "--config", config,
        "--output_dir", output_dir,
        "--device", device,
    ]
    pretrain_main()


@cli.command()
@click.option("--train_data", required=True, help="Training data directory")
@click.option("--val_data", required=True, help="Validation data directory")
@click.option("--output_dir", required=True, help="Output directory")
@click.option("--config", default="configs/config.yaml", help="Config file")
@click.option("--pretrained_checkpoint", default=None, help="Pre-trained checkpoint")
@click.option("--device", default="cuda", help="Device")
def train(train_data, val_data, output_dir, config, pretrained_checkpoint, device):
    """Supervised fine-tuning."""
    from scripts.train import main as train_main
    sys.argv = [
        "train.py",
        "--train_data", train_data,
        "--val_data", val_data,
        "--output_dir", output_dir,
        "--config", config,
        "--device", device,
    ]
    if pretrained_checkpoint:
        sys.argv.extend(["--pretrained_checkpoint", pretrained_checkpoint])
    train_main()


@cli.command()
@click.option("--input", "input_fasta", required=True, help="Input FASTA file")
@click.option("--checkpoint", required=True, help="Model checkpoint")
@click.option("--output", default="predictions.csv", help="Output CSV")
@click.option("--threshold", default=0.5, help="Classification threshold")
@click.option("--device", default="cuda", help="Device")
def predict(input_fasta, checkpoint, output, threshold, device):
    """Predict virophage contigs."""
    from scripts.predict import main as predict_main
    sys.argv = [
        "predict.py",
        "--input", input_fasta,
        "--checkpoint", checkpoint,
        "--output", output,
        "--threshold", str(threshold),
        "--device", device,
    ]
    predict_main()


def main():
    cli()


if __name__ == "__main__":
    main()
