#!/usr/bin/env python3
"""Supervised fine-tuning script for VirophageFinder."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virophagefinder.utils.io import load_config, ensure_dir
from virophagefinder.data.dataloader import create_dataloaders
from virophagefinder.models.virophagefinder import VirophageFinderModel
from virophagefinder.training.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VirophageFinder model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data directory (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint (from contrastive pre-training)",
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
    """Set random seeds for reproducibility."""
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
    print("VirophageFinder Training")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    print(f"Config loaded from {args.config}")

    # Create output directory
    ensure_dir(args.output_dir)

    # Create dataloaders
    print("\nCreating dataloaders...")
    data_cfg = config["data"]
    loaders = create_dataloaders(
        train_dir=args.train_data,
        val_dir=args.val_data,
        test_dir=args.test_data,
        batch_size=config["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        max_contig_length=data_cfg.get("max_contig_length", 50000),
        min_contig_length=data_cfg.get("min_contig_length", 500),
        max_proteins=config["model"]["protein"].get("max_proteins_per_contig", 50),
        augment_train=data_cfg.get("augmentation", {}).get("enabled", True),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val: {len(loaders['val'].dataset)} samples")
    if "test" in loaders:
        print(f"  Test: {len(loaders['test'].dataset)} samples")

    # Build model
    print("\nBuilding model...")
    model = VirophageFinderModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        output_dir=args.output_dir,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=args.device,
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    # Final evaluation on test set
    if "test" in loaders:
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        # Load best model
        best_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(trainer.device)

        # Evaluate
        trainer.val_loader = loaders["test"]
        test_metrics = trainer.validate()

        from virophagefinder.evaluation.metrics import print_metrics
        print_metrics(test_metrics, "Test Set Metrics")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
