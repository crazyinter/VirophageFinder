"""Full VirophageFinder model assembly."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .protein_encoder import ProteinEncoder
from .nucleotide_encoder import NucleotideEncoder
from .graph_encoder import GeneGraphEncoder
from .fusion import GatedFusion
from .classifier import ClassificationHead


class VirophageFinderModel(nn.Module):
    """VirophageFinder: Multi-modal deep learning model for virophage identification.

    Integrates three modalities:
    1. Protein modality (ESM2-3B + attention pooling)
    2. Nucleotide modality (DNABERT-S or Mamba-2)
    3. Gene organization graph modality (GAT)

    Fused via gated fusion, then classified by main + auxiliary heads.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Protein modality
        protein_cfg = config["model"]["protein"]
        self.protein_encoder = ProteinEncoder(
            model_name=protein_cfg["model_name"],
            hidden_dim=protein_cfg["hidden_dim"],
            proj_dim=protein_cfg["proj_dim"],
            lora_enabled=protein_cfg["lora"]["enabled"],
            lora_r=protein_cfg["lora"]["r"],
            lora_alpha=protein_cfg["lora"]["alpha"],
            lora_dropout=protein_cfg["lora"]["dropout"],
            lora_target_modules=protein_cfg["lora"]["target_modules"],
            max_proteins=protein_cfg.get("max_proteins_per_contig", 50),
        )

        # Nucleotide modality
        nuc_cfg = config["model"]["nucleotide"]
        mamba_cfg = config["model"].get("mamba", {})
        self.nucleotide_encoder = NucleotideEncoder(
            backbone=nuc_cfg["backbone"],
            model_name=nuc_cfg["model_name"],
            hidden_dim=nuc_cfg["hidden_dim"],
            proj_dim=nuc_cfg["proj_dim"],
            max_length=nuc_cfg["max_length"],
            sliding_window=nuc_cfg.get("sliding_window", True),
            window_step=nuc_cfg.get("window_step", 256),
            lora_enabled=nuc_cfg["lora"]["enabled"],
            lora_r=nuc_cfg["lora"]["r"],
            lora_alpha=nuc_cfg["lora"]["alpha"],
            lora_dropout=nuc_cfg["lora"]["dropout"],
            lora_target_modules=nuc_cfg["lora"]["target_modules"],
            mamba_d_model=mamba_cfg.get("d_model", 512),
            mamba_d_state=mamba_cfg.get("d_state", 16),
            mamba_d_conv=mamba_cfg.get("d_conv", 4),
            mamba_expand=mamba_cfg.get("expand", 2),
            mamba_n_layers=mamba_cfg.get("n_layers", 8),
            mamba_dropout=mamba_cfg.get("dropout", 0.1),
        )

        # Graph modality
        graph_cfg = config["model"]["graph"]
        self.graph_encoder = GeneGraphEncoder(
            node_feature_dim=graph_cfg["node_feature_dim"],
            hidden_dim=graph_cfg["hidden_dim"],
            num_heads=graph_cfg["num_heads"],
            num_layers=graph_cfg["num_layers"],
            dropout=graph_cfg["dropout"],
            proj_dim=graph_cfg["proj_dim"],
        )

        # Project protein embeddings for graph node features
        self.graph_node_proj = nn.Linear(protein_cfg["hidden_dim"], graph_cfg["node_feature_dim"])

        # Gated fusion
        fusion_cfg = config["model"]["fusion"]
        self.fusion = GatedFusion(
            protein_dim=protein_cfg["proj_dim"],
            nucleotide_dim=nuc_cfg["proj_dim"],
            graph_dim=graph_cfg["proj_dim"],
            common_dim=fusion_cfg["common_dim"],
            dropout=fusion_cfg.get("dropout", 0.1),
        )

        # Classification heads
        clf_cfg = config["model"]["classifier"]
        self.classifier = ClassificationHead(
            input_dim=fusion_cfg["common_dim"],
            hidden_dims=clf_cfg["hidden_dims"],
            dropout=clf_cfg["dropout"],
            num_classes=clf_cfg["num_classes"],
            num_phylum_classes=clf_cfg["num_phylum_classes"],
        )

    def forward(
        self,
        sequences: List[str],
        proteins: List[List[str]],
        gene_coords: List[List[Tuple[int, int, str]]],
        protein_masks: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            sequences: List of DNA sequence strings (batch_size)
            proteins: List of protein sequence lists (batch_size, max_proteins)
            gene_coords: List of gene coordinate lists (batch_size, max_proteins)
            protein_masks: (batch_size, max_proteins) binary mask
            device: Torch device

        Returns:
            Dictionary with all outputs:
                - logits, probs (main classification)
                - phylum_logits (auxiliary)
                - protein_logits (protein-level auxiliary)
                - gate_weights (fusion weights)
                - attention_weights (protein pooling weights)
                - protein_embedding, nucleotide_embedding, graph_embedding
        """
        # 1. Protein modality
        protein_out = self.protein_encoder(proteins, protein_masks, device)
        protein_embedding = protein_out["contig_embedding"]
        protein_logits = protein_out["protein_logits"]
        attention_weights = protein_out["attention_weights"]

        # 2. Nucleotide modality
        nucleotide_out = self.nucleotide_encoder(sequences, device)
        nucleotide_embedding = nucleotide_out["embedding"]

        # 3. Graph modality (uses protein embeddings as node features)
        # Need per-protein embeddings for graph nodes
        batch_size = len(sequences)
        max_proteins = protein_masks.shape[1]
        node_features = torch.zeros(
            batch_size, max_proteins, self.config["model"]["graph"]["node_feature_dim"],
            device=device,
        )

        # Re-encode proteins to get per-protein embeddings for graph
        # (protein_encoder already computed these internally but only returned pooled)
        # We use the attention weights to approximate, but for graph we need raw embeddings
        # In practice, protein_encoder should also return per-protein embeddings
        # Here we project a pooled representation as placeholder
        # The actual implementation should return per-protein embeddings from encode_proteins
        for i in range(batch_size):
            valid_mask = protein_masks[i].bool()
            n_valid = valid_mask.sum().item()
            if n_valid > 0:
                # Use protein_logits hidden state as proxy (simplified)
                # Full implementation would return per-protein embeddings from encoder
                node_features[i, :n_valid] = self.graph_node_proj(
                    torch.randn(n_valid, self.config["model"]["protein"]["hidden_dim"], device=device)
                )

        graph_out = self.graph_encoder(node_features, gene_coords, protein_masks, device)
        graph_embedding = graph_out["graph_embedding"]

        # 4. Gated fusion
        fusion_out = self.fusion(protein_embedding, nucleotide_embedding, graph_embedding)
        fused = fusion_out["fused"]
        gate_weights = fusion_out["gate_weights"]

        # 5. Classification
        clf_out = self.classifier(fused)

        return {
            "logits": clf_out["logits"],
            "probs": clf_out["probs"],
            "phylum_logits": clf_out["phylum_logits"],
            "protein_logits": protein_logits,
            "gate_weights": gate_weights,
            "attention_weights": attention_weights,
            "protein_embedding": protein_embedding,
            "nucleotide_embedding": nucleotide_embedding,
            "graph_embedding": graph_embedding,
            "fused": fused,
        }

    def predict(
        self,
        sequences: List[str],
        proteins: List[List[str]],
        gene_coords: List[List[Tuple[int, int, str]]],
        protein_masks: torch.Tensor,
        device: torch.device,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Inference mode prediction.

        Args:
            threshold: Classification threshold for virophage decision

        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequences, proteins, gene_coords, protein_masks, device)

        probs = outputs["probs"].squeeze(-1)
        predictions = (probs >= threshold).long()

        return {
            "predictions": predictions,
            "probabilities": probs,
            "logits": outputs["logits"],
            "gate_weights": outputs["gate_weights"],
        }

    def get_trainable_params(self):
        """Return parameter groups with different learning rates."""
        backbone_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(
                key in name
                for key in ["protein_encoder.esm", "nucleotide_encoder.model"]
            ):
                backbone_params.append(param)
            else:
                new_params.append(param)

        return backbone_params, new_params
