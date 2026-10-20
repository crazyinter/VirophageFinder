"""Prediction pipeline for VirophageFinder."""

import os
from typing import Dict, List, Optional

import torch
import pandas as pd
from Bio import SeqIO

from ..models.virophagefinder import VirophageFinderModel
from ..utils.gene_prediction import predict_genes


class VirophagePredictor:
    """Inference pipeline for predicting virophage contigs from FASTA input."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: str = "cuda",
        threshold: float = 0.5,
        max_proteins: int = 50,
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Model configuration (loaded from checkpoint if None)
            device: Device for inference
            threshold: Classification threshold
            max_proteins: Maximum proteins to process per contig
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_proteins = max_proteins

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if config is None:
            config = checkpoint.get("config", {})

        # Build model
        self.model = VirophageFinderModel(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Device: {self.device}, Threshold: {threshold}")

    def predict_fasta(
        self,
        fasta_path: str,
        output_path: Optional[str] = None,
        min_length: int = 500,
        batch_size: int = 8,
    ) -> pd.DataFrame:
        """Predict virophage contigs from a FASTA file.

        Args:
            fasta_path: Path to input FASTA file
            output_path: Optional path to save CSV output
            min_length: Minimum contig length to process
            batch_size: Batch size for inference

        Returns:
            DataFrame with predictions for each contig
        """
        # Load contigs
        contigs = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) >= min_length:
                contigs.append({"id": record.id, "sequence": seq, "length": len(seq)})

        print(f"Loaded {len(contigs)} contigs (>= {min_length} bp)")

        # Process in batches
        all_results = []
        for i in range(0, len(contigs), batch_size):
            batch = contigs[i : i + batch_size]
            batch_results = self._predict_batch(batch)
            all_results.extend(batch_results)

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(contigs)} contigs")

        # Create DataFrame
        df = pd.DataFrame(all_results)
        df["prediction"] = (df["probability"] >= self.threshold).astype(int)
        df = df.sort_values("probability", ascending=False).reset_index(drop=True)

        # Save output
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        # Summary
        n_virophage = (df["prediction"] == 1).sum()
        print(f"\nPrediction summary:")
        print(f"  Total contigs: {len(df)}")
        print(f"  Virophage predictions: {n_virophage} ({100*n_virophage/len(df):.1f}%)")
        print(f"  Mean probability: {df['probability'].mean():.4f}")

        return df

    def _predict_batch(self, batch: List[Dict]) -> List[Dict]:
        """Predict a single batch of contigs."""
        sequences = [b["sequence"] for b in batch]
        ids = [b["id"] for b in batch]
        lengths = [b["length"] for b in batch]

        # Predict genes for each contig
        proteins_batch = []
        gene_coords_batch = []
        protein_masks = []

        for seq in sequences:
            coords, proteins = predict_genes(seq)
            # Limit proteins
            if len(proteins) > self.max_proteins:
                sorted_idx = sorted(
                    range(len(proteins)), key=lambda i: len(proteins[i]), reverse=True
                )
                keep = sorted(sorted_idx[: self.max_proteins])
                proteins = [proteins[j] for j in keep]
                coords = [coords[j] for j in keep]

            proteins_batch.append(proteins)
            gene_coords_batch.append(coords)
            protein_masks.append([1] * len(proteins))

        # Pad to max in batch
        max_prot = max(len(p) for p in proteins_batch) if proteins_batch else 1
        for i in range(len(proteins_batch)):
            n = len(proteins_batch[i])
            proteins_batch[i] = proteins_batch[i] + [""] * (max_prot - n)
            gene_coords_batch[i] = gene_coords_batch[i] + [(0, 0, "+")] * (max_prot - n)
            protein_masks[i] = protein_masks[i] + [0] * (max_prot - n)

        protein_masks = torch.tensor(protein_masks, dtype=torch.float, device=self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                sequences=sequences,
                proteins=proteins_batch,
                gene_coords=gene_coords_batch,
                protein_masks=protein_masks,
                device=self.device,
            )

        probs = outputs["probs"].squeeze(-1).cpu().numpy()
        gate_weights = outputs["gate_weights"].cpu().numpy()

        results = []
        for j in range(len(batch)):
            results.append(
                {
                    "contig_id": ids[j],
                    "length": lengths[j],
                    "probability": float(probs[j]),
                    "gate_protein": float(gate_weights[j, 0]),
                    "gate_nucleotide": float(gate_weights[j, 1]),
                    "gate_graph": float(gate_weights[j, 2]),
                    "n_genes": int(protein_masks[j].sum().item()),
                }
            )

        return results

    def extract_virophage_contigs(
        self,
        fasta_path: str,
        output_fasta: str,
        min_probability: float = 0.5,
    ) -> int:
        """Extract predicted virophage contigs to a new FASTA file.

        Args:
            fasta_path: Input FASTA
            output_fasta: Output FASTA for virophage contigs
            min_probability: Minimum probability threshold

        Returns:
            Number of extracted contigs
        """
        df = self.predict_fasta(fasta_path)
        virophage_ids = set(df[df["probability"] >= min_probability]["contig_id"])

        n_extracted = 0
        with open(output_fasta, "w") as out_f:
            for record in SeqIO.parse(fasta_path, "fasta"):
                if record.id in virophage_ids:
                    SeqIO.write(record, out_f, "fasta")
                    n_extracted += 1

        print(f"Extracted {n_extracted} virophage contigs to {output_fasta}")
        return n_extracted
