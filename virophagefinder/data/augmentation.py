"""Sequence augmentation for virophage genome fragments."""

import random
from typing import Optional

import numpy as np
from Bio.Seq import Seq


class SequenceAugmentor:
    """Augmentation strategies for genomic sequences.

    Supports:
    - Random point mutation at specified rates
    - Reverse complement
    - Random gene region deletion (simulating fragmented contigs)
    - Random fragmentation
    """

    def __init__(
        self,
        mutation_rates: Optional[list] = None,
        reverse_complement: bool = True,
        gene_deletion_prob: float = 0.1,
        seed: int = 42,
    ):
        self.mutation_rates = mutation_rates or [0.01, 0.03, 0.05]
        self.reverse_complement = reverse_complement
        self.gene_deletion_prob = gene_deletion_prob
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def __call__(self, sequence: str, gene_coords: Optional[list] = None) -> str:
        """Apply random augmentation to a sequence.

        Args:
            sequence: DNA sequence string (A/T/G/C)
            gene_coords: List of (start, end, strand) tuples for predicted genes

        Returns:
            Augmented DNA sequence
        """
        seq = sequence

        # 1. Point mutation (applied with probability proportional to rate selection)
        if self.rng.random() < 0.5:
            rate = self.rng.choice(self.mutation_rates)
            seq = self._mutate(seq, rate)

        # 2. Reverse complement
        if self.reverse_complement and self.rng.random() < 0.5:
            seq = str(Seq(seq).reverse_complement())

        # 3. Gene deletion (simulate contigs lacking marker genes)
        if gene_coords and self.rng.random() < self.gene_deletion_prob:
            seq = self._delete_random_genes(seq, gene_coords)

        return seq

    def _mutate(self, sequence: str, mutation_rate: float) -> str:
        """Introduce random point mutations."""
        bases = ["A", "T", "G", "C"]
        seq_list = list(sequence)
        n_mutations = int(len(seq_list) * mutation_rate)
        positions = self.np_rng.choice(len(seq_list), size=n_mutations, replace=False)
        for pos in positions:
            original = seq_list[pos]
            alternatives = [b for b in bases if b != original]
            seq_list[pos] = self.rng.choice(alternatives)
        return "".join(seq_list)

    def _delete_random_genes(self, sequence: str, gene_coords: list) -> str:
        """Delete 1-2 random gene regions, replacing with Ns."""
        if not gene_coords:
            return sequence
        n_delete = min(self.rng.randint(1, 2), len(gene_coords))
        selected = self.rng.sample(gene_coords, n_delete)
        seq_list = list(sequence)
        for start, end, _ in selected:
            start = max(0, start)
            end = min(len(seq_list), end)
            seq_list[start:end] = ["N"] * (end - start)
        return "".join(seq_list)

    def fragment(
        self, sequence: str, length_bins: Optional[list] = None, n_fragments: int = 20
    ) -> list:
        """Generate random fragments from a full genome.

        Args:
            sequence: Full genome sequence
            length_bins: List of target fragment lengths in bp
            n_fragments: Number of fragments to generate

        Returns:
            List of (fragment_sequence, start, end) tuples
        """
        length_bins = length_bins or [500, 1000, 3000, 5000, 10000]
        fragments = []
        seq_len = len(sequence)

        for _ in range(n_fragments):
            target_len = self.rng.choice(length_bins)
            if target_len >= seq_len:
                fragments.append((sequence, 0, seq_len))
                continue
            start = self.rng.randint(0, seq_len - target_len)
            end = start + target_len
            fragments.append((sequence[start:end], start, end))

        return fragments
