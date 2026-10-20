"""Supervised training loop for VirophageFinder."""

import os
import json
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from ..evaluation.metrics import compute_metrics
from .losses import VirophageLoss


class Trainer:
    """Trainer for VirophageFinder supervised fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: str,
        pretrained_checkpoint: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        # Load pretrained checkpoint if provided
        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            self._load_pretrained(pretrained_checkpoint)

        self.model.to(self.device)

        # Loss
        loss_cfg = config["training"]["loss"]
        self.criterion = VirophageLoss(
            focal_alpha=loss_cfg["focal_alpha"],
            focal_gamma=loss_cfg["focal_gamma"],
            protein_aux_weight=loss_cfg["protein_aux_weight"],
            phylum_aux_weight=loss_cfg["phylum_aux_weight"],
        )

        # Optimizer with hierarchical learning rates
        backbone_params, new_params = model.get_trainable_params()
        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": train_cfg["backbone_learning_rate"]},
                {"params": new_params, "lr": train_cfg["learning_rate"]},
            ],
            weight_decay=train_cfg["weight_decay"],
            betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        )

        # Learning rate scheduler
        total_steps = (
            len(train_loader)
            // train_cfg.get("gradient_accumulation_steps", 1)
            * train_cfg["epochs"]
        )
        warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", "fp16") in ["fp16", "bf16"]
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training state
        self.epochs = train_cfg["epochs"]
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.early_stopping_patience = train_cfg["early_stopping_patience"]
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.history = {"train_loss": [], "val_metrics": []}

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights (from contrastive pre-training)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load with strict=False to allow missing classifier heads
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained checkpoint: {checkpoint_path}")
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            labels = batch["labels"].to(self.device)
            protein_types = batch["protein_types"].to(self.device)
            phylums = batch["phylums"].to(self.device)
            protein_masks = batch["protein_masks"].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    sequences=batch["sequences"],
                    proteins=batch["proteins"],
                    gene_coords=batch["gene_coords"],
                    protein_masks=protein_masks,
                    device=self.device,
                )
                loss_dict = self.criterion(outputs, labels, protein_types, phylums)
                loss = loss_dict["total"] / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss_dict["total"].item()
            n_batches += 1

            if batch_idx % self.config["output"]["log_interval"] == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss_dict['total'].item():.4f} "
                    f"(main: {loss_dict['main'].item():.4f}, "
                    f"prot: {loss_dict['protein_aux'].item():.4f}, "
                    f"phyl: {loss_dict['phylum_aux'].item():.4f})"
                )

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation and compute metrics."""
        self.model.eval()
        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            labels = batch["labels"].to(self.device)
            protein_masks = batch["protein_masks"].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    sequences=batch["sequences"],
                    proteins=batch["proteins"],
                    gene_coords=batch["gene_coords"],
                    protein_masks=protein_masks,
                    device=self.device,
                )

            probs = outputs["probs"].squeeze(-1).cpu()
            all_probs.append(probs)
            all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metrics = compute_metrics(all_labels, all_probs)
        return metrics

    def train(self):
        """Full training loop with early stopping."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            epoch_time = time.time() - start_time
            print(
                f"\nEpoch {epoch}/{self.epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f} - "
                f"Val MCC: {val_metrics['mcc']:.4f} - "
                f"Val AUC: {val_metrics['auc_roc']:.4f}"
            )

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_metrics"].append(val_metrics)

            # Early stopping and checkpointing
            monitor_metric = val_metrics["f1"]
            if monitor_metric > self.best_metric:
                self.best_metric = monitor_metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  -> New best model saved (F1={monitor_metric:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Save latest checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best=False)

        # Save training history
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best Val F1: {self.best_metric:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(self.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.output_dir, "latest_model.pt")

        torch.save(checkpoint, path)
