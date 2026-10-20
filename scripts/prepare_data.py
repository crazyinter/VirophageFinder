#!/usr/bin/env python3
"""Prepare training/validation/test datasets for VirophageFinder.

Processes virophage and negative genome FASTA files into augmented,
partitioned datasets ready for training.
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict, Tuple

from Bio import SeqIO
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virophagefinder.data.augmentation import SequenceAugmentor
from virophagefinder.utils.gene_prediction import predict_genes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare VirophageFinder training datasets"
    )
    parser.add_argument(
        "--virophage_genomes",
        type=str,
        required=True,
        help="FASTA file containing virophage genomes",
    )
    parser.add_argument(
        "--negative_dir",
        type=str,
        required=True,
        help="Directory containing negative genome FASTA files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--n_fragments",
        type=int,
        default=20,
        help="Number of fragments per genome (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def load_genomes(fasta_path: str) -> List[Tuple[str, str]]:
    """Load genomes from FASTA file."""
    genomes = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        genomes.append((record.id, str(record.seq).upper()))
    return genomes


def load_negative_genomes(negative_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load negative genomes organized by tier (L1-L4 subdirectories)."""
    negatives = {}
    for tier in ["L1", "L2", "L3", "L4"]:
        tier_dir = os.path.join(negative_dir, tier)
        if os.path.exists(tier_dir):
            genomes = []
            for fname in os.listdir(tier_dir):
                if fname.endswith((".fasta", ".fa", ".fna")):
                    genomes.extend(load_genomes(os.path.join(tier_dir, fname)))
            negatives[tier] = genomes
            print(f"  {tier}: {len(genomes)} genomes")
    return negatives


def partition_by_genus(
    genomes: List[Tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Partition genomes by genus (extracted from ID) to avoid data leakage.

    Assumes genome IDs contain genus information, e.g., 'Genus_species_strain'.
    Falls back to random partition if genus cannot be extracted.
    """
    rng = random.Random(seed)

    # Group by genus
    genera = {}
    for genome_id, seq in genomes:
        # Extract genus from ID (first word before underscore)
        parts = genome_id.replace("_", " ").split()
        genus = parts[0] if parts else genome_id
        if genus not in genera:
            genera[genus] = []
        genera[genus].append((genome_id, seq))

    genus_list = list(genera.keys())
    rng.shuffle(genus_list)

    n_train = int(len(genus_list) * train_ratio)
    n_val = int(len(genus_list) * val_ratio)

    train_genera = genus_list[:n_train]
    val_genera = genus_list[n_train : n_train + n_val]
    test_genera = genus_list[n_train + n_val :]

    train = [g for genus in train_genera for g in genera[genus]]
    val = [g for genus in val_genera for g in genera[genus]]
    test = [g for genus in test_genera for g in genera[genus]]

    print(f"  Partition: {len(train)} train, {len(val)} val, {len(test)} test genomes")
    print(f"  Genera: {len(train_genera)} train, {len(val_genera)} val, {len(test_genera)} test")

    return train, val, test


def generate_augmented_contigs(
    genomes: List[Tuple[str, str]],
    label: int,
    negative_tier: int,
    augmentor: SequenceAugmentor,
    n_fragments: int = 20,
    phylum: int = -1,
) -> List[Dict]:
    """Generate augmented contig fragments from genomes.

    Args:
        genomes: List of (genome_id, sequence) tuples
        label: 1 for virophage, 0 for negative
        negative_tier: Negative sample tier (L1-L4), -1 for virophage
        augmentor: SequenceAugmentor instance
        n_fragments: Number of fragments per genome
        phylum: Virophage phylum/family label (for auxiliary loss)

    Returns:
        List of sample dictionaries
    """
    samples = []
    length_bins = [500, 1000, 3000, 5000, 10000]

    for genome_id, seq in genomes:
        # Predict genes for augmentation
        gene_coords, _ = predict_genes(seq)

        # Generate fragments
        fragments = augmentor.fragment(seq, length_bins, n_fragments)

        for frag_idx, (frag_seq, start, end) in enumerate(fragments):
            # Apply augmentation
            frag_gene_coords = [
                (max(0, s - start), min(len(frag_seq), e - start), strand)
                for s, e, strand in gene_coords
                if s >= start and e <= end
            ]
            aug_seq = augmentor(frag_seq, frag_gene_coords)

            sample_id = f"{genome_id}_frag{frag_idx}"
            samples.append(
                {
                    "id": sample_id,
                    "sequence": aug_seq,
                    "label": label,
                    "phylum": phylum,
                    "negative_tier": negative_tier,
                    "source_genome": genome_id,
                    "original_start": start,
                    "original_end": end,
                }
            )

    return samples


def save_dataset(samples: List[Dict], output_path: str, label_file: str):
    """Save dataset as FASTA + JSON labels."""
    # Save FASTA
    fasta_path = os.path.join(output_path, "contigs.fasta")
    with open(fasta_path, "w") as f:
        for sample in samples:
            f.write(f">{sample['id']}\n")
            f.write(f"{sample['sequence']}\n")

    # Save labels
    labels = {}
    for sample in samples:
        labels[sample["id"]] = {
            "label": sample["label"],
            "phylum": sample["phylum"],
            "negative_tier": sample["negative_tier"],
            "source_genome": sample["source_genome"],
        }
    with open(label_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Saved {len(samples)} contigs to {output_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("VirophageFinder Dataset Preparation")
    print("=" * 60)

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # Load virophage genomes
    print("\n[1/4] Loading virophage genomes...")
    virophage_genomes = load_genomes(args.virophage_genomes)
    print(f"  Loaded {len(virophage_genomes)} virophage genomes")

    # Load negative genomes
    print("\n[2/4] Loading negative genomes...")
    negative_genomes = load_negative_genomes(args.negative_dir)
    total_neg = sum(len(v) for v in negative_genomes.values())
    print(f"  Total: {total_neg} negative genomes")

    # Partition by genus
    print("\n[3/4] Partitioning datasets (by genus to avoid leakage)...")
    vir_train, vir_val, vir_test = partition_by_genus(
        virophage_genomes, args.train_ratio, args.val_ratio, args.seed
    )

    # Initialize augmentor
    augmentor = SequenceAugmentor(
        mutation_rates=[0.01, 0.03, 0.05],
        reverse_complement=True,
        gene_deletion_prob=0.1,
        seed=args.seed,
    )

    # Generate augmented contigs
    print("\n[4/4] Generating augmented contigs...")

    for split_name, vir_split in [("train", vir_train), ("val", vir_val), ("test", vir_test)]:
        print(f"\n  Processing {split_name} set...")

        # Virophage samples
        vir_samples = generate_augmented_contigs(
            vir_split, label=1, negative_tier=-1,
            augmentor=augmentor, n_fragments=args.n_fragments,
        )

        # Negative samples (sample proportionally from each tier)
        neg_samples = []
        tier_ratios = {"L1": 0.40, "L2": 0.25, "L3": 0.20, "L4": 0.15}
        n_neg_total = len(vir_samples) * 3  # 3:1 negative to positive ratio

        for tier, ratio in tier_ratios.items():
            if tier in negative_genomes and negative_genomes[tier]:
                n_tier = int(n_neg_total * ratio)
                tier_genomes = random.sample(
                    negative_genomes[tier],
                    min(n_tier, len(negative_genomes[tier])),
                )
                tier_samples = generate_augmented_contigs(
                    tier_genomes, label=0,
                    negative_tier=int(tier[1]),
                    augmentor=augmentor,
                    n_fragments=max(1, n_tier // max(len(tier_genomes), 1)),
                )
                neg_samples.extend(tier_samples)

        # Combine and shuffle
        all_samples = vir_samples + neg_samples
        random.shuffle(all_samples)

        # Save
        split_dir = os.path.join(args.output_dir, split_name)
        save_dataset(
            all_samples,
            split_dir,
            os.path.join(split_dir, "labels.json"),
        )

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
