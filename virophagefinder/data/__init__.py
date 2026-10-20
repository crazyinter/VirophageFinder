"""Data module for VirophageFinder."""

from .augmentation import SequenceAugmentor

__all__ = ["SequenceAugmentor", "VirophageDataset", "collate_fn", "create_dataloaders"]

# Lazy imports for torch-dependent modules
def __getattr__(name):
    if name in ("VirophageDataset", "collate_fn"):
        from .dataset import VirophageDataset, collate_fn
        return VirophageDataset if name == "VirophageDataset" else collate_fn
    if name == "create_dataloaders":
        from .dataloader import create_dataloaders
        return create_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
