"""Nucleotide modality encoder using DNABERT-S or Mamba-2."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


class NucleotideEncoder(nn.Module):
    """Encode nucleotide sequences using a genomic foundation model.

    Supports two backbones:
    - DNABERT-S: BPE-tokenized Transformer with sliding window for long sequences
    - Mamba-2: State space model with linear complexity for long sequences
    """

    def __init__(
        self,
        backbone: str = "dnabert_s",
        model_name: str = "zhihan1996/DNABERT-S",
        hidden_dim: int = 768,
        proj_dim: int = 512,
        max_length: int = 512,
        sliding_window: bool = True,
        window_step: int = 256,
        lora_enabled: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        # Mamba-2 specific
        mamba_d_model: int = 512,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 8,
        mamba_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_length = max_length
        self.sliding_window = sliding_window
        self.window_step = window_step

        if backbone == "dnabert_s":
            self._init_dnabert(
                model_name, hidden_dim, lora_enabled, lora_r, lora_alpha,
                lora_dropout, lora_target_modules,
            )
        elif backbone == "mamba2":
            self._init_mamba2(
                mamba_d_model, mamba_d_state, mamba_d_conv,
                mamba_expand, mamba_n_layers, mamba_dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Projection to common dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def _init_dnabert(
        self, model_name, hidden_dim, lora_enabled, lora_r, lora_alpha,
        lora_dropout, lora_target_modules,
    ):
        """Initialize DNABERT-S model with LoRA."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if lora_enabled:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_config)

    def _init_mamba2(
        self, d_model, d_state, d_conv, expand, n_layers, dropout,
    ):
        """Initialize Mamba-2 backbone (simplified implementation)."""
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
            from mamba_ssm.ops.triton.layernorm import RMSNorm
        except ImportError:
            raise ImportError(
                "mamba-ssm is required for Mamba-2 backbone. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        self.tokenizer = None  # Mamba uses raw one-hot or learned embedding
        self.embedding = nn.Embedding(6, d_model)  # A,T,G,C,N,pad

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            self.norms.append(RMSNorm(d_model))

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = d_model  # override for Mamba

    def _encode_dnabert(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """Encode sequences using DNABERT-S with sliding window for long sequences."""
        batch_embeddings = []

        for seq in sequences:
            # Tokenize
            tokens = self.tokenizer(
                seq,
                return_tensors=None,
                add_special_tokens=True,
                truncation=False,
            )
            input_ids = tokens["input_ids"]

            # Sliding window for long sequences
            if len(input_ids) > self.max_length and self.sliding_window:
                window_embeddings = []
                for start in range(0, len(input_ids) - self.max_length + 1, self.window_step):
                    end = start + self.max_length
                    window_ids = torch.tensor(
                        [input_ids[start:end]], device=device
                    )
                    attention_mask = torch.ones_like(window_ids)

                    with torch.no_grad() if not self.training else torch.enable_grad():
                        outputs = self.model(
                            input_ids=window_ids, attention_mask=attention_mask
                        )
                    # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token representation
                    cls_emb = outputs.last_hidden_state[0, 0, :]
                    window_embeddings.append(cls_emb)

                if window_embeddings:
                    # Average pooling over windows
                    seq_embedding = torch.stack(window_embeddings).mean(dim=0)
                else:
                    seq_embedding = torch.zeros(self.hidden_dim, device=device)
            else:
                # Single window
                inputs = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad() if not self.training else torch.enable_grad():
                    outputs = self.model(**inputs)

                seq_embedding = outputs.last_hidden_state[0, 0, :]

            batch_embeddings.append(seq_embedding)

        return torch.stack(batch_embeddings)

    def _encode_mamba2(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """Encode sequences using Mamba-2 (full length, no windowing)."""
        # Map nucleotides to indices: A=0, T=1, G=2, C=3, N=4, pad=5
        base_map = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
        max_len = max(len(s) for s in sequences)

        batch_ids = []
        batch_mask = []
        for seq in sequences:
            ids = [base_map.get(c, 4) for c in seq.upper()]
            mask = [1] * len(ids)
            # Pad
            pad_len = max_len - len(ids)
            ids += [5] * pad_len
            mask += [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)

        input_ids = torch.tensor(batch_ids, device=device)
        attention_mask = torch.tensor(batch_mask, device=device, dtype=torch.float)

        # Embedding
        x = self.embedding(input_ids)

        # Mamba layers with residual
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x) + residual

        # Mean pooling over valid positions
        mask_expanded = attention_mask.unsqueeze(-1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return x

    def forward(self, sequences: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequences: List of DNA sequence strings
            device: Torch device

        Returns:
            Dictionary with 'embedding' (batch_size, proj_dim)
        """
        if self.backbone == "dnabert_s":
            raw_embedding = self._encode_dnabert(sequences, device)
        elif self.backbone == "mamba2":
            raw_embedding = self._encode_mamba2(sequences, device)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        projected = self.projection(raw_embedding)
        return {"embedding": projected, "raw_embedding": raw_embedding}
