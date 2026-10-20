"""Gated multi-modal fusion module."""

from typing import Dict

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Gated multi-modal fusion with adaptive modality weighting.

    Learns to weight each modality based on input characteristics,
    allowing the model to down-weight unreliable modalities (e.g.,
    protein features for very short contigs).
    """

    def __init__(
        self,
        protein_dim: int = 512,
        nucleotide_dim: int = 512,
        graph_dim: int = 256,
        common_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.common_dim = common_dim

        # Individual modality projections to common dimension
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )
        self.nucleotide_proj = nn.Sequential(
            nn.Linear(nucleotide_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # Gating network: computes 3 weights from concatenated modalities
        total_dim = protein_dim + nucleotide_dim + graph_dim
        self.gate = nn.Sequential(
            nn.Linear(total_dim, common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, 3),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein_feat: torch.Tensor,
        nucleotide_feat: torch.Tensor,
        graph_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse three modality features.

        Args:
            protein_feat: (batch_size, protein_dim)
            nucleotide_feat: (batch_size, nucleotide_dim)
            graph_feat: (batch_size, graph_dim)

        Returns:
            Dictionary with:
                - fused: (batch_size, common_dim) fused representation
                - gate_weights: (batch_size, 3) learned modality weights
        """
        # Project to common dimension
        p_proj = self.protein_proj(protein_feat)
        n_proj = self.nucleotide_proj(nucleotide_feat)
        g_proj = self.graph_proj(graph_feat)

        # Compute gate weights from raw (unprojected) concatenation
        concat = torch.cat([protein_feat, nucleotide_feat, graph_feat], dim=-1)
        gate_weights = torch.softmax(self.gate(concat), dim=-1)

        # Weighted fusion
        fused = (
            gate_weights[:, 0:1] * p_proj
            + gate_weights[:, 1:2] * n_proj
            + gate_weights[:, 2:3] * g_proj
        )

        fused = self.dropout(fused)

        return {"fused": fused, "gate_weights": gate_weights}
