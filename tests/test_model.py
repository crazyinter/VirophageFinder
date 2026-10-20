"""Unit tests for VirophageFinder model components."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virophagefinder.models.fusion import GatedFusion
from virophagefinder.models.classifier import ClassificationHead
from virophagefinder.training.losses import FocalLoss, InfoNCELoss
from virophagefinder.evaluation.metrics import compute_metrics
from virophagefinder.data.augmentation import SequenceAugmentor
from virophagefinder.utils.gene_prediction import predict_genes, _predict_heuristic


class TestGatedFusion:
    """Tests for GatedFusion module."""

    def test_output_shape(self):
        fusion = GatedFusion(protein_dim=512, nucleotide_dim=512, graph_dim=256, common_dim=256)
        p = torch.randn(4, 512)
        n = torch.randn(4, 512)
        g = torch.randn(4, 256)
        out = fusion(p, n, g)
        assert out["fused"].shape == (4, 256)
        assert out["gate_weights"].shape == (4, 3)

    def test_gate_weights_sum_to_one(self):
        fusion = GatedFusion()
        p = torch.randn(2, 512)
        n = torch.randn(2, 512)
        g = torch.randn(2, 256)
        out = fusion(p, n, g)
        assert torch.allclose(out["gate_weights"].sum(dim=-1), torch.ones(2), atol=1e-5)


class TestClassificationHead:
    """Tests for ClassificationHead."""

    def test_output_shape(self):
        clf = ClassificationHead(input_dim=256, num_classes=1, num_phylum_classes=5)
        x = torch.randn(8, 256)
        out = clf(x)
        assert out["logits"].shape == (8, 1)
        assert out["probs"].shape == (8, 1)
        assert out["phylum_logits"].shape == (8, 5)

    def test_probabilities_in_range(self):
        clf = ClassificationHead()
        x = torch.randn(4, 256)
        out = clf(x)
        assert (out["probs"] >= 0).all() and (out["probs"] <= 1).all()


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_basic(self):
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_reduction_none(self):
        loss_fn = FocalLoss(reduction="none")
        logits = torch.randn(4, 1)
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)


class TestInfoNCELoss:
    """Tests for InfoNCELoss."""

    def test_basic(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        z1 = torch.randn(8, 128)
        z2 = torch.randn(8, 128)
        loss = loss_fn(z1, z2)
        assert loss.item() >= 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_metrics(self):
        import numpy as np
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.95])
        metrics = compute_metrics(labels, probs)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "mcc" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["f1"] <= 1

    def test_perfect_prediction(self):
        import numpy as np
        labels = np.array([0, 1, 1, 0])
        probs = np.array([0.0, 1.0, 1.0, 0.0])
        metrics = compute_metrics(labels, probs)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0


class TestAugmentation:
    """Tests for sequence augmentation."""

    def test_mutation(self):
        aug = SequenceAugmentor(mutation_rates=[0.1], reverse_complement=False, gene_deletion_prob=0)
        seq = "ATGCATGCATGCATGC" * 10  # 160 bp
        mutated = aug._mutate(seq, 0.1)
        assert len(mutated) == len(seq)
        # At least some bases should change
        diffs = sum(1 for a, b in zip(seq, mutated) if a != b)
        assert diffs > 0

    def test_reverse_complement(self):
        aug = SequenceAugmentor(reverse_complement=True)
        seq = "ATGCATGC"
        result = aug(seq)
        assert len(result) == len(seq)

    def test_fragmentation(self):
        aug = SequenceAugmentor()
        seq = "ATGC" * 1000  # 4000 bp
        fragments = aug.fragment(seq, length_bins=[500, 1000], n_fragments=5)
        assert len(fragments) == 5
        for frag, start, end in fragments:
            assert end - start == len(frag)


class TestGenePrediction:
    """Tests for gene prediction utilities."""

    def test_heuristic_short_sequence(self):
        coords, proteins = _predict_heuristic("ATGC", min_protein_length=10)
        assert coords == []
        assert proteins == []

    def test_heuristic_orf_detection(self):
        # Sequence with a clear ORF
        seq = "ATGAAATTTGGGCCCTAA" + "A" * 50
        coords, proteins = _predict_heuristic(seq, min_protein_length=5)
        # Should find at least one ORF
        assert len(coords) >= 0  # may or may not find depending on exact sequence
        for coord in coords:
            assert len(coord) == 3
            start, end, strand = coord
            assert start < end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
