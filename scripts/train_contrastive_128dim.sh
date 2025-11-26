#!/bin/bash
#
# Train Contrastive VQ-VAE with 128-dim Projection
#
# This script trains a contrastive VQ-VAE with 128-dim projection dimension
# (matching masked VQ-VAE) for better clustering compared to 64-dim.
#
# Saves to: experiments/3_contrastive_vqvae_128dim/
#
# Usage:
#   ./scripts/train_contrastive_128dim.sh
#

# Exit on error
set -e

# Configuration
VQVAE_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
OUTPUT_DIR="experiments/3_contrastive_vqvae_128dim"
EXPERIMENT_NAME="contrastive_128dim_run_1"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-4
TEMPERATURE=0.5
MASK_PROB=0.15
DROP_PROB=0.10
PROJ_DIM=128  # Key: 128-dim to match masked VQ-VAE

# Hardware
N_GPU=2
GPU_IDS="0,1"

echo "=========================================="
echo "Contrastive VQ-VAE Training (128-dim)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  VQ-VAE checkpoint: $VQVAE_CHECKPOINT"
echo "  Data: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo ""
echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Temperature: $TEMPERATURE"
echo "  Projection dim: $PROJ_DIM (matches masked VQ-VAE)"
echo "  Mask probability: $MASK_PROB"
echo "  Drop probability: $DROP_PROB"
echo ""
echo "Hardware:"
echo "  GPUs: $N_GPU (IDs: $GPU_IDS)"
echo ""

# Check if checkpoint exists
if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found: $VQVAE_CHECKPOINT"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting contrastive training with 128-dim projection..."
echo ""

python -u scripts/contrastive_finetune_128dim.py \
    --data-path "$DATA_PATH" \
    --checkpoint-path "$VQVAE_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --temperature "$TEMPERATURE" \
    --mask-prob "$MASK_PROB" \
    --drop-prob "$DROP_PROB" \
    --proj-dim "$PROJ_DIM" \
    --n-gpu "$N_GPU" \
    --gpu-ids "$GPU_IDS" \
    --save-freq 5 \
    --num-workers 4

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  • checkpoints/best_model.pt"
echo "  • checkpoints/checkpoint_epoch_*.pt"
echo "  • visualizations/training_curves.png"
echo "  • visualizations/embedding_space_umap.png"
echo "  • visualizations/final_embeddings.npy"
echo ""
echo "Next steps:"
echo "  1. Evaluate with genus-level taxonomy:"
echo "     python scripts/label_contrastive_128dim_clusters_with_taxa.py"
echo ""
echo "  2. Compare clustering quality:"
echo "     - Masked VQ-VAE: 128-dim, no projection, no normalization"
echo "     - Contrastive 64-dim: 64-dim, projection + normalization"
echo "     - Contrastive 128-dim: 128-dim, projection + normalization (NEW)"
echo ""
echo "Expected improvement:"
echo "  Higher dimensionality (128 vs 64) should preserve more information"
echo "  Better clustering quality than 64-dim contrastive"
echo "  May still have normalization effects vs masked VQ-VAE"
echo ""
