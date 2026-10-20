"""Classification heads for VirophageFinder."""

from typing import Dict, List

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Main and auxiliary classification heads.

    Main head: binary virophage/non-virophage classification
    Auxiliary head 1: virophage family/phylum classification (for positive samples)
    Auxiliary head 2: protein-level classification (MCP/ATPase/protease/other)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_classes: int = 1,
        num_phylum_classes: int = 5,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        # Build main classifier MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.main_classifier = nn.Sequential(*layers)

        # Auxiliary phylum classifier
        phylum_layers = []
        prev_dim = input_dim
        for hidden_dim in [128]:
            phylum_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        phylum_layers.append(nn.Linear(prev_dim, num_phylum_classes))
        self.phylum_classifier = nn.Sequential(*phylum_layers)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            fused_features: (batch_size, input_dim) from fusion module

        Returns:
            Dictionary with:
                - logits: (batch_size, 1) main classification logits
                - probs: (batch_size, 1) sigmoid probabilities
                - phylum_logits: (batch_size, num_phylum_classes) auxiliary predictions
        """
        logits = self.main_classifier(fused_features)
        probs = torch.sigmoid(logits)
        phylum_logits = self.phylum_classifier(fused_features)

        return {
            "logits": logits,
            "probs": probs,
            "phylum_logits": phylum_logits,
        }
