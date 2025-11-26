#!/bin/bash
#
# Train Contrastive VQ-VAE (10 epochs)
# SimCLR-style contrastive fine-tuning using best Standard VQ-VAE
#
# Input: experiments/1_standard_vqvae/checkpoints/best_model.pt
# Output: experiments/3_contrastive_vqvae/checkpoints/best_model.pt
#

set -e  # Exit on error

echo "========================================="
echo "TRAINING CONTRASTIVE VQ-VAE (Step 3/3)"
echo "========================================="
echo ""

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Check if standard VQ-VAE checkpoint exists
STANDARD_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"

if [ ! -f "$STANDARD_CHECKPOINT" ]; then
    echo "ERROR: Standard VQ-VAE checkpoint not found!"
    echo "Expected: $STANDARD_CHECKPOINT"
    echo ""
    echo "Please run ./scripts/1_train_standard.sh first."
    exit 1
fi

echo "Configuration:"
echo "  • Base model: $STANDARD_CHECKPOINT"
echo "  • Data: data/cleaned_reads.fastq"
echo "  • Epochs: 50"
echo "  • Batch size: 32"
echo "  • GPUs: 0,1 (free GPUs)"
echo "  • Contrastive temp: 0.07"
echo "  • Output: experiments/3_contrastive_vqvae/"
echo ""

# Set environment variables for memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python -u scripts/contrastive_finetune.py \
    --checkpoint-path "$STANDARD_CHECKPOINT" \
    --data-path data/cleaned_reads.fastq \
    --output-dir experiments/3_contrastive_vqvae \
    --experiment-name contrastive_vqvae_50epochs \
    --gpu-ids 2,3 \
    --n-gpu 2 \
    --batch-size 32 \
    --epochs 50 \
    --train-split 0.9 \
    --seed 42 \
    --max-seq-length 150 \
    --k-mer 6 \
    --proj-dim 64 \
    --temperature 0.07 \
    --learning-rate 1e-4 \
    --save-freq 1 \
    --use-wandb \
    --wandb-project vqvae-genomics

echo ""
echo "========================================="
echo "✓ Contrastive VQ-VAE training complete!"
echo "========================================="
echo ""
echo "Best checkpoint: experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
echo ""
echo "All models trained successfully!"
echo "Next step: Evaluate individual models with scripts/4_evaluate_*.sh"
echo ""
