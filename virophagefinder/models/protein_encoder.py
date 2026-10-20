"""Protein modality encoder using ESM2 with LoRA fine-tuning."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import LoraConfig, get_peft_model


class ProteinEncoder(nn.Module):
    """Encode protein sequences using ESM2 with attention pooling.

    Processes multiple proteins per contig and aggregates them into a
    contig-level representation via learned attention weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t36_3B_UR50D",
        hidden_dim: int = 2560,
        proj_dim: int = 512,
        lora_enabled: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        max_proteins: int = 50,
        max_protein_length: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_proteins = max_proteins
        self.max_protein_length = max_protein_length

        # Load ESM2 model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)

        # Apply LoRA for parameter-efficient fine-tuning
        if lora_enabled:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.esm = get_peft_model(self.esm, lora_config)

        # Attention pooling: learn weights for each protein
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Projection to common dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Auxiliary protein classifier (MCP/ATPase/protease/other)
        self.protein_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 4),
        )

    def encode_proteins(
        self, proteins: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of protein sequences.

        Args:
            proteins: List of protein sequence strings
            device: Torch device

        Returns:
            protein_embeddings: (num_proteins, hidden_dim) tensor
            protein_logits: (num_proteins, 4) auxiliary classification logits
        """
        if not proteins or all(p == "" for p in proteins):
            # Return zeros if no proteins
            embeddings = torch.zeros(1, self.hidden_dim, device=device)
            logits = torch.zeros(1, 4, device=device)
            return embeddings, logits

        # Filter empty proteins
        valid_proteins = [p for p in proteins if p and len(p) > 0]
        if not valid_proteins:
            embeddings = torch.zeros(1, self.hidden_dim, device=device)
            logits = torch.zeros(1, 4, device=device)
            return embeddings, logits

        # Truncate long proteins
        valid_proteins = [p[: self.max_protein_length] for p in valid_proteins]

        # Tokenize
        inputs = self.tokenizer(
            valid_proteins,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_protein_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.esm(**inputs)

        # Use mean pooling over sequence length (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * attention_mask).sum(
            dim=1
        ) / attention_mask.sum(dim=1).clamp(min=1)

        # Auxiliary classification
        protein_logits = self.protein_classifier(embeddings)

        return embeddings, protein_logits

    def forward(
        self,
        proteins: List[List[str]],
        protein_masks: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of contigs.

        Args:
            proteins: Batch of protein lists, shape (batch_size, max_proteins)
            protein_masks: (batch_size, max_proteins) binary mask
            device: Torch device

        Returns:
            Dictionary with:
                - contig_embedding: (batch_size, proj_dim) aggregated protein features
                - protein_logits: (batch_size, max_proteins, 4) auxiliary predictions
                - attention_weights: (batch_size, max_proteins) learned weights
        """
        batch_size = len(proteins)
        max_proteins = protein_masks.shape[1]

        all_contig_embeddings = []
        all_protein_logits = []
        all_attention_weights = []

        for i in range(batch_size):
            # Get valid proteins for this contig
            valid_mask = protein_masks[i].bool()
            valid_proteins = [proteins[i][j] for j in range(max_proteins) if valid_mask[j]]

            # Encode proteins
            protein_embeddings, protein_logits = self.encode_proteins(
                valid_proteins, device
            )

            # Attention pooling over proteins
            attn_scores = self.attention(protein_embeddings).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=0)
            contig_embedding = (attn_weights.unsqueeze(-1) * protein_embeddings).sum(dim=0)

            # Project
            contig_embedding = self.projection(contig_embedding)

            # Pad protein logits to max_proteins
            padded_logits = torch.zeros(max_proteins, 4, device=device)
            n_valid = min(len(valid_proteins), max_proteins)
            if n_valid > 0:
                padded_logits[:n_valid] = protein_logits[:n_valid]

            # Pad attention weights
            padded_attn = torch.zeros(max_proteins, device=device)
            if n_valid > 0:
                padded_attn[:n_valid] = attn_weights[:n_valid]

            all_contig_embeddings.append(contig_embedding)
            all_protein_logits.append(padded_logits)
            all_attention_weights.append(padded_attn)

        return {
            "contig_embedding": torch.stack(all_contig_embeddings),
            "protein_logits": torch.stack(all_protein_logits),
            "attention_weights": torch.stack(all_attention_weights),
        }
