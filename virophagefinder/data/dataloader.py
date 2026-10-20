"""DataLoader utilities for VirophageFinder."""

from typing import Dict, Optional

from torch.utils.data import DataLoader

from .dataset import VirophageDataset, collate_fn


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_contig_length: int = 50000,
    min_contig_length: int = 500,
    max_proteins: int = 50,
    augment_train: bool = True,
    augment_config: Optional[Dict] = None,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders.

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Optional test data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_contig_length: Maximum contig length
        min_contig_length: Minimum contig length
        max_proteins: Max proteins per contig
        augment_train: Whether to augment training data
        augment_config: Augmentation configuration
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    train_dataset = VirophageDataset(
        data_dir=train_dir,
        max_contig_length=max_contig_length,
        min_contig_length=min_contig_length,
        max_proteins=max_proteins,
        augment=augment_train,
        augment_config=augment_config,
    )

    val_dataset = VirophageDataset(
        data_dir=val_dir,
        max_contig_length=max_contig_length,
        min_contig_length=min_contig_length,
        max_proteins=max_proteins,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    loaders = {"train": train_loader, "val": val_loader}

    if test_dir:
        test_dataset = VirophageDataset(
            data_dir=test_dir,
            max_contig_length=max_contig_length,
            min_contig_length=min_contig_length,
            max_proteins=max_proteins,
            augment=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
        )
        loaders["test"] = test_loader

    return loaders
