"""Loss functions for VirophageFinder training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, 1) raw logits
            targets: (batch_size,) binary labels

        Returns:
            Focal loss scalar
        """
        logits = logits.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class VirophageLoss(nn.Module):
    """Combined loss for VirophageFinder.

    Main: Focal loss for virophage/non-virophage binary classification
    Auxiliary 1: Cross-entropy for protein-level classification (MCP/ATPase/protease/other)
    Auxiliary 2: Cross-entropy for virophage phylum/family classification
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        protein_aux_weight: float = 0.2,
        phylum_aux_weight: float = 0.1,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.protein_aux_weight = protein_aux_weight
        self.phylum_aux_weight = phylum_aux_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        protein_types: torch.Tensor,
        phylums: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            outputs: Model output dictionary
            labels: (batch_size,) binary virophage labels
            protein_types: (batch_size, max_proteins) protein type labels (-1 = ignore)
            phylums: (batch_size,) phylum labels (-1 = ignore for non-virophages)

        Returns:
            Dictionary with total loss and component losses
        """
        # Main focal loss
        main_loss = self.focal_loss(outputs["logits"], labels)

        # Protein auxiliary loss (only for valid proteins)
        protein_logits = outputs["protein_logits"]  # (B, max_proteins, 4)
        B, P, C = protein_logits.shape
        protein_loss = self.ce_loss(
            protein_logits.reshape(B * P, C),
            protein_types.reshape(B * P),
        )

        # Phylum auxiliary loss (only for virophage samples)
        phylum_logits = outputs["phylum_logits"]  # (B, num_phylum)
        phylum_loss = self.ce_loss(phylum_logits, phylums)

        # Total loss
        total_loss = (
            main_loss
            + self.protein_aux_weight * protein_loss
            + self.phylum_aux_weight * phylum_loss
        )

        return {
            "total": total_loss,
            "main": main_loss,
            "protein_aux": protein_loss,
            "phylum_aux": phylum_loss,
        }


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for pre-training."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss for positive pairs.

        Args:
            z_i: (batch_size, dim) embeddings of first view
            z_j: (batch_size, dim) embeddings of second view

        Returns:
            InfoNCE loss scalar
        """
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )

        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Mask out self-similarity
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)).float()

        # Numerator: positive similarities
        numerator = torch.exp(positives / self.temperature)

        # Denominator: sum of all similarities (excluding self)
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        # Loss
        loss = -torch.log(numerator / denominator.sum(dim=1))
        return loss.mean()
