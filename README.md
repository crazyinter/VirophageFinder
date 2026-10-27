# VirophageFinder
A multi-modal deep learning framework for accurate identification of virophages in metagenomic data.

## Overview

VirophageFinder integrates three complementary modalities for virophage contig classification:
- **Protein modality**: ESM2-3B protein language model embeddings with LoRA fine-tuning
- **Nucleotide modality**: DNABERT-S genomic foundation model (or Mamba-2 for long sequences)
- **Graph modality**: Graph Attention Network over gene organization structure

These are combined through a gated fusion mechanism, with contrastive pre-training on unlabeled viral sequences to overcome limited labeled virophage genomes.

## Installation

```bash
# Clone the repository
git clone https://github.com/crazyinter/VirophageFinder.git
cd VirophageFinder

# Create conda environment
conda create -n virophagefinder python=3.10 -y
conda activate virophagefinder

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare training data

```bash
python scripts/prepare_data.py \
    --virophage_genomes data/virophage_genomes.fasta \
    --negative_genomes data/negative_genomes/ \
    --output_dir data/processed/
```

### 2. Contrastive pre-training (optional but recommended)

```bash
python scripts/pretrain.py \
    --unlabeled_contigs data/unlabeled_viral_contigs.fasta \
    --config configs/pretrain_config.yaml \
    --output_dir checkpoints/pretrain/
```

### 3. Supervised fine-tuning

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --train_data data/processed/train/ \
    --val_data data/processed/val/ \
    --output_dir checkpoints/finetune/ \
    --pretrained_checkpoint checkpoints/pretrain/best_model.pt
```

### 4. Predict on metagenomic contigs

```bash
python scripts/predict.py \
    --input contigs.fasta \
    --checkpoint checkpoints/finetune/best_model.pt \
    --output predictions.csv \
    --threshold 0.5
```

## Project Structure

```
VirophageFinder/
├── configs/                    # Configuration files
├── scripts/                    # Command-line entry points
├── tests/                      # Unit tests
└── virophagefinder/
    ├── data/                   # Dataset, augmentation, dataloader
    ├── models/                 # Model architectures
    │   ├── protein_encoder.py  # ESM2 protein modality
    │   ├── nucleotide_encoder.py # DNABERT-S / Mamba-2
    │   ├── graph_encoder.py    # GAT gene organization graph
    │   ├── fusion.py           # Gated multi-modal fusion
    │   └── virophagefinder.py  # Full model assembly
    ├── training/               # Pre-training and fine-tuning loops
    ├── inference/              # Prediction pipeline
    ├── evaluation/             # Metrics and benchmarking
    └── utils/                  # Gene prediction, IO utilities
```




