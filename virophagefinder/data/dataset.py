"""Dataset classes for VirophageFinder training and evaluation."""

import os
import json
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from Bio import SeqIO

from ..utils.gene_prediction import predict_genes
from .augmentation import SequenceAugmentor


class VirophageDataset(Dataset):
    """Dataset for virophage contig classification.

    Each sample contains:
    - nucleotide sequence
    - predicted protein sequences
    - gene coordinates (for graph construction)
    - label (1 = virophage, 0 = non-virophage)
    - auxiliary labels (phylum, protein types)
    """

    def __init__(
        self,
        data_dir: str,
        max_contig_length: int = 50000,
        min_contig_length: int = 500,
        max_proteins: int = 50,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        label_file: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Directory containing FASTA files and metadata
            max_contig_length: Maximum contig length in bp
            min_contig_length: Minimum contig length in bp
            max_proteins: Maximum number of proteins to process per contig
            augment: Whether to apply data augmentation
            augment_config: Configuration for augmentation
            label_file: Optional JSON file with labels (if not in FASTA headers)
        """
        self.data_dir = data_dir
        self.max_contig_length = max_contig_length
        self.min_contig_length = min_contig_length
        self.max_proteins = max_proteins
        self.augment = augment

        if augment and augment_config:
            self.augmentor = SequenceAugmentor(**augment_config)
        elif augment:
            self.augmentor = SequenceAugmentor()
        else:
            self.augmentor = None

        # Load samples
        self.samples = self._load_samples(data_dir, label_file)

    def _load_samples(self, data_dir: str, label_file: Optional[str]) -> List[Dict]:
        """Load all contig samples from directory."""
        samples = []
        labels = {}

        if label_file and os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)

        # Find all FASTA files
        fasta_files = []
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith((".fasta", ".fa", ".fna")):
                    fasta_files.append(os.path.join(root, fname))

        for fasta_path in fasta_files:
            for record in SeqIO.parse(fasta_path, "fasta"):
                seq = str(record.seq).upper()
                if len(seq) < self.min_contig_length:
                    continue

                # Truncate if too long
                if len(seq) > self.max_contig_length:
                    seq = seq[: self.max_contig_length]

                # Parse label from header or label file
                if record.id in labels:
                    label = labels[record.id]["label"]
                    phylum = labels[record.id].get("phylum", -1)
                    negative_tier = labels[record.id].get("negative_tier", -1)
                else:
                    # Parse from header: >contig_id|label|phylum
                    parts = record.description.split("|")
                    label = int(parts[1]) if len(parts) > 1 else 0
                    phylum = int(parts[2]) if len(parts) > 2 else -1
                    negative_tier = int(parts[3]) if len(parts) > 3 else -1

                samples.append(
                    {
                        "id": record.id,
                        "sequence": seq,
                        "label": label,
                        "phylum": phylum,
                        "negative_tier": negative_tier,
                        "length": len(seq),
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sequence = sample["sequence"]

        # Apply augmentation
        gene_coords = None
        if self.augmentor and self.augment:
            # Predict genes first for gene-deletion augmentation
            gene_coords, _ = predict_genes(sequence)
            sequence = self.augmentor(sequence, gene_coords)

        # Predict genes (or re-predict after augmentation)
        gene_coords, protein_seqs = predict_genes(sequence)

        # Limit number of proteins
        if len(protein_seqs) > self.max_proteins:
            # Keep longest proteins (more informative)
            sorted_idx = sorted(
                range(len(protein_seqs)), key=lambda i: len(protein_seqs[i]), reverse=True
            )
            keep_idx = sorted(sorted_idx[: self.max_proteins])
            protein_seqs = [protein_seqs[i] for i in keep_idx]
            gene_coords = [gene_coords[i] for i in keep_idx]

        # Protein type labels (for auxiliary loss): 0=MCP, 1=ATPase, 2=protease, 3=other
        # In practice, this would come from annotation; here default to other
        protein_types = [3] * len(protein_seqs)

        return {
            "id": sample["id"],
            "sequence": sequence,
            "proteins": protein_seqs,
            "gene_coords": gene_coords,
            "protein_types": protein_types,
            "label": torch.tensor(sample["label"], dtype=torch.float),
            "phylum": torch.tensor(sample["phylum"], dtype=torch.long),
            "length": sample["length"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for variable-length sequences and proteins.

    Pads protein sequences and gene coordinate lists to the maximum in batch.
    """
    # Simple fields
    ids = [b["id"] for b in batch]
    sequences = [b["sequence"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    phylums = torch.stack([b["phylum"] for b in batch])
    lengths = [b["length"] for b in batch]

    # Protein sequences: variable number per contig
    proteins_list = [b["proteins"] for b in batch]
    gene_coords_list = [b["gene_coords"] for b in batch]
    protein_types_list = [b["protein_types"] for b in batch]

    # Max proteins in this batch
    max_proteins = max(len(p) for p in proteins_list) if proteins_list else 0

    # Pad protein lists
    padded_proteins = []
    padded_gene_coords = []
    padded_protein_types = []
    protein_masks = []

    for proteins, coords, ptypes in zip(proteins_list, gene_coords_list, protein_types_list):
        n = len(proteins)
        mask = [1] * n + [0] * (max_proteins - n)
        protein_masks.append(mask)

        # Pad proteins with empty string
        padded_proteins.append(proteins + [""] * (max_proteins - n))
        # Pad gene coords with (0,0,+)
        padded_gene_coords.append(coords + [(0, 0, "+")] * (max_proteins - n))
        # Pad protein types with -1 (ignore index)
        padded_protein_types.append(ptypes + [-1] * (max_proteins - n))

    return {
        "ids": ids,
        "sequences": sequences,
        "proteins": padded_proteins,
        "gene_coords": padded_gene_coords,
        "protein_types": torch.tensor(padded_protein_types, dtype=torch.long),
        "protein_masks": torch.tensor(protein_masks, dtype=torch.float),
        "labels": labels,
        "phylums": phylums,
        "lengths": lengths,
        "max_proteins": max_proteins,
    }
