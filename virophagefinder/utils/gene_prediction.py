"""Gene prediction utilities for metagenomic contigs."""

import os
import subprocess
import tempfile
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq


def predict_genes(
    sequence: str,
    tool: str = "prodigal",
    min_protein_length: int = 30,
) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """Predict genes from a nucleotide sequence.

    Args:
        sequence: DNA sequence string
        tool: Gene prediction tool ('prodigal' or 'heuristic')
        min_protein_length: Minimum amino acid length for predicted proteins

    Returns:
        Tuple of:
        - gene_coords: List of (start, end, strand) tuples (0-indexed)
        - proteins: List of protein sequence strings
    """
    if len(sequence) < 100:
        return [], []

    if tool == "prodigal":
        try:
            return _predict_prodigal(sequence, min_protein_length)
        except (FileNotFoundError, subprocess.CalledProcessError, Exception):
            # Fall back to heuristic if Prodigal is not available
            return _predict_heuristic(sequence, min_protein_length)
    else:
        return _predict_heuristic(sequence, min_protein_length)


def _predict_prodigal(
    sequence: str, min_protein_length: int
) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """Predict genes using Prodigal (metagenomic mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input FASTA
        input_path = os.path.join(tmpdir, "input.fasta")
        output_path = os.path.join(tmpdir, "output.faa")
        with open(input_path, "w") as f:
            f.write(">contig\n")
            f.write(sequence + "\n")

        # Run Prodigal
        cmd = [
            "prodigal",
            "-i", input_path,
            "-a", output_path,
            "-p", "meta",  # metagenomic mode
            "-q",  # quiet
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            return _predict_heuristic(sequence, min_protein_length)

        # Parse output
        gene_coords = []
        proteins = []
        for record in SeqIO.parse(output_path, "fasta"):
            # Parse coordinates from header: >contig_1 # start # end # strand # ...
            parts = record.description.split("#")
            if len(parts) >= 4:
                try:
                    start = int(parts[1].strip()) - 1  # convert to 0-indexed
                    end = int(parts[2].strip())
                    strand = parts[3].strip()
                    protein = str(record.seq)
                    if len(protein) >= min_protein_length:
                        gene_coords.append((start, end, strand))
                        proteins.append(protein)
                except (ValueError, IndexError):
                    continue

        return gene_coords, proteins


def _predict_heuristic(
    sequence: str, min_protein_length: int
) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """Heuristic gene prediction using six-frame translation.

    Finds ORFs (Open Reading Frames) with start and stop codons.
    Used as fallback when Prodigal is unavailable.
    """
    gene_coords = []
    proteins = []
    seq = sequence.upper()

    start_codons = {"ATG", "GTG", "TTG"}
    stop_codons = {"TAA", "TAG", "TGA"}

    for strand, seq_strand in [("+", seq), ("-", str(Seq(seq).reverse_complement()))]:
        for frame in range(3):
            i = frame
            while i < len(seq_strand) - 2:
                codon = seq_strand[i : i + 3]
                if codon in start_codons:
                    # Find next stop codon
                    j = i + 3
                    while j < len(seq_strand) - 2:
                        next_codon = seq_strand[j : j + 3]
                        if next_codon in stop_codons:
                            # Translate ORF
                            orf_seq = seq_strand[i:j]
                            protein = str(Seq(orf_seq).translate())
                            if len(protein) >= min_protein_length:
                                # Convert coordinates
                                if strand == "+":
                                    start, end = i, j + 3
                                else:
                                    start = len(seq) - (j + 3)
                                    end = len(seq) - i
                                gene_coords.append((start, end, strand))
                                proteins.append(protein)
                            break
                        j += 3
                    i = j + 3
                else:
                    i += 3

    # Sort by genomic position
    if gene_coords:
        paired = sorted(zip(gene_coords, proteins), key=lambda x: x[0][0])
        gene_coords = [p[0] for p in paired]
        proteins = [p[1] for p in paired]

    return gene_coords, proteins
