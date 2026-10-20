"""Contrastive pre-training for VirophageFinder."""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from Bio import SeqIO

from ..models.protein_encoder import ProteinEncoder
from ..models.nucleotide_encoder import NucleotideEncoder
from .losses import InfoNCELoss


class ContrastiveDataset(Dataset):
    """Dataset for contrastive pre-training on unlabeled viral contigs.

    Generates positive pairs from:
    1. Two non-overlapping fragments of the same genome
    2. Homologous protein pairs
    """

    def __init__(
        self,
        fasta_path: str,
        max_length: int = 10000,
        min_length: int = 1000,
    ):
        self.contigs = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            if min_length <= len(seq) <= max_length:
                self.contigs.append((record.id, seq))

        print(f"Loaded {len(self.contigs)} contigs for contrastive pre-training")

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        contig_id, seq = self.contigs[idx]

        # Generate two views (fragments) of the same contig
        seq_len = len(seq)
        if seq_len > 2000:
            # Two non-overlapping fragments
            frag_len = min(seq_len // 2, 5000)
            view1_start = 0
            view1_end = frag_len
            view2_start = seq_len - frag_len
            view2_end = seq_len
            view1 = seq[view1_start:view1_end]
            view2 = seq[view2_start:view2_end]
        else:
            # Short contig: use same sequence with different augmentation
            view1 = seq
            view2 = self._augment(seq)

        return {"id": contig_id, "view1": view1, "view2": view2}

    def _augment(self, seq: str) -> str:
        """Simple augmentation: reverse complement or random mutation."""
        import random
        if random.random() < 0.5:
            # Reverse complement
            comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
            return "".join(comp.get(c, "N") for c in reversed(seq))
        else:
            # 1% random mutation
            bases = ["A", "T", "G", "C"]
            seq_list = list(seq)
            n_mut = max(1, int(len(seq_list) * 0.01))
            positions = random.sample(range(len(seq_list)), n_mut)
            for pos in positions:
                seq_list[pos] = random.choice(bases)
            return "".join(seq_list)


def contrastive_collate_fn(batch):
    """Collate function for contrastive dataset."""
    ids = [b["id"] for b in batch]
    view1 = [b["view1"] for b in batch]
    view2 = [b["view2"] for b in batch]
    return {"ids": ids, "view1": view1, "view2": view2}


class ContrastivePretrainer:
    """Contrastive pre-training using InfoNCE loss.

    Pre-trains nucleotide and protein encoders on unlabeled viral sequences
    before supervised fine-tuning.
    """

    def __init__(
        self,
        config: Dict,
        output_dir: str,
        device: str = "cuda",
    ):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize encoders (shared weights with downstream model)
        model_cfg = config["model"]
        self.protein_encoder = ProteinEncoder(
            model_name=model_cfg["protein"]["model_name"],
            hidden_dim=model_cfg["protein"]["hidden_dim"],
            proj_dim=model_cfg["protein"]["proj_dim"],
            lora_enabled=model_cfg["protein"]["lora"]["enabled"],
            lora_r=model_cfg["protein"]["lora"]["r"],
        ).to(self.device)

        self.nucleotide_encoder = NucleotideEncoder(
            backbone=model_cfg["nucleotide"]["backbone"],
            model_name=model_cfg["nucleotide"]["model_name"],
            hidden_dim=model_cfg["nucleotide"]["hidden_dim"],
            proj_dim=model_cfg["nucleotide"]["proj_dim"],
            max_length=model_cfg["nucleotide"]["max_length"],
        ).to(self.device)

        # Contrastive projection head
        proj_cfg = model_cfg["projection"]
        self.projection_head = nn.Sequential(
            nn.Linear(proj_cfg["input_dim"], proj_cfg["hidden_dim"]),
            nn.GELU(),
            nn.Linear(proj_cfg["hidden_dim"], proj_cfg["output_dim"]),
        ).to(self.device)

        # Loss
        self.criterion = InfoNCELoss(temperature=config["training"]["infonce"]["temperature"])

        # Optimizer
        all_params = (
            list(self.protein_encoder.parameters())
            + list(self.nucleotide_encoder.parameters())
            + list(self.projection_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        train_cfg = config["training"]
        total_steps = 1000  # estimated
        warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.use_amp = train_cfg.get("mixed_precision", "fp16") in ["fp16", "bf16"]
        self.scaler = GradScaler(enabled=self.use_amp)
        self.epochs = train_cfg["epochs"]

    def encode_view(self, sequences: List[str]) -> torch.Tensor:
        """Encode a view through nucleotide + protein encoders + projection."""
        # Nucleotide encoding
        nuc_out = self.nucleotide_encoder(sequences, self.device)
        nuc_emb = nuc_out["embedding"]

        # Protein encoding (simplified: predict genes and encode)
        from ..utils.gene_prediction import predict_genes
        proteins_batch = []
        protein_masks = []
        max_prot = 20
        for seq in sequences:
            _, prots = predict_genes(seq)
            prots = prots[:max_prot]
            proteins_batch.append(prots + [""] * (max_prot - len(prots)))
            protein_masks.append([1] * len(prots) + [0] * (max_prot - len(prots)))

        protein_masks = torch.tensor(protein_masks, dtype=torch.float, device=self.device)
        prot_out = self.protein_encoder(proteins_batch, protein_masks, self.device)
        prot_emb = prot_out["contig_embedding"]

        # Concatenate and project
        combined = torch.cat([nuc_emb, prot_emb], dim=-1)
        projected = self.projection_head(combined)
        return projected

    def train(self, fasta_path: str):
        """Run contrastive pre-training."""
        dataset = ContrastiveDataset(fasta_path)
        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=contrastive_collate_fn,
            num_workers=4,
            drop_last=True,
        )

        print(f"Starting contrastive pre-training for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            self.protein_encoder.train()
            self.nucleotide_encoder.train()
            self.projection_head.train()

            total_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                with autocast(enabled=self.use_amp):
                    z1 = self.encode_view(batch["view1"])
                    z2 = self.encode_view(batch["view2"])
                    loss = self.criterion(z1, z2)
                    loss = loss / self.config["training"].get("gradient_accumulation_steps", 1)

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config["training"].get("gradient_accumulation_steps", 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.protein_encoder.parameters())
                        + list(self.nucleotide_encoder.parameters())
                        + list(self.projection_head.parameters()),
                        1.0,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(loader)
            print(f"\nEpoch {epoch}/{self.epochs} - Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            self._save_checkpoint(epoch, avg_loss)

        print("Contrastive pre-training complete")

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save pre-trained encoder weights."""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "protein_encoder_state_dict": self.protein_encoder.state_dict(),
            "nucleotide_encoder_state_dict": self.nucleotide_encoder.state_dict(),
            "projection_head_state_dict": self.projection_head.state_dict(),
            "config": self.config,
        }
        path = os.path.join(self.output_dir, f"pretrain_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        # Also save as best (latest)
        torch.save(checkpoint, os.path.join(self.output_dir, "best_model.pt"))
