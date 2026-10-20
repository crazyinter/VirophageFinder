"""IO utilities for VirophageFinder."""

import os
from typing import Dict, List, Tuple

import yaml
from Bio import SeqIO


def read_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Read sequences from a FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        List of (record_id, sequence) tuples
    """
    records = []
    for record in SeqIO.parse(filepath, "fasta"):
        records.append((record.id, str(record.seq).upper()))
    return records


def write_fasta(
    sequences: List[Tuple[str, str]],
    filepath: str,
    description: str = "",
):
    """Write sequences to a FASTA file.

    Args:
        sequences: List of (record_id, sequence) tuples
        filepath: Output file path
        description: Optional description to append to headers
    """
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    records = []
    for seq_id, seq in sequences:
        desc = description if description else ""
        records.append(SeqRecord(Seq(seq), id=seq_id, description=desc))

    SeqIO.write(records, filepath, "fasta")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(dirpath: str):
    """Create directory if it doesn't exist."""
    os.makedirs(dirpath, exist_ok=True)


def count_fasta_records(filepath: str) -> int:
    """Count number of records in a FASTA file."""
    count = 0
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count
