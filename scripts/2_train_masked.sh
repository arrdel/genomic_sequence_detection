#!/bin/bash
#
# Train Masked VQ-VAE (50 epochs)
# BERT-style masked language modeling
#
# Output: experiments/2_masked_vqvae/checkpoints/best_model.pt
#

set -e  # Exit on error

echo "========================================="
echo "TRAINING MASKED VQ-VAE (Step 2/3)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  • Data: data/cleaned_reads.fastq"
echo "  • Epochs: 50"
echo "  • Batch size: 64"
echo "  • GPUs: 2"
echo "  • Masking: 20% of tokens"
echo "  • Output: experiments/2_masked_vqvae/"
echo ""

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Set environment variables for memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python -u scripts/mqvae_train.py \
    --data-path data/cleaned_reads.fastq \
    --output-dir experiments/2_masked_vqvae \
    --experiment-name masked_vqvae_50epochs \
    --gpu-ids 0,1 \
    --n-gpu 2 \
    --batch-size 32 \
    --epochs 50 \
    --test-split 0.1 \
    --seed 42 \
    --max-seq-length 150 \
    --k-mer 6 \
    --vocab-size 4099 \
    --num-codes 512 \
    --code-dim 64 \
    --embed-dim 128 \
    --hidden-dim 256 \
    --learning-rate 2e-4 \
    --commitment-cost 0.1 \
    --mask-prob 0.2 \
    --save-freq 5 \
    --wandb-project vqvae-genomics

echo ""
echo "========================================="
echo "✓ Masked VQ-VAE training complete!"
echo "========================================="
echo ""
echo "Best checkpoint: experiments/2_masked_vqvae/checkpoints/best_model.pt"
echo "Next step: Run ./scripts/3_train_contrastive.sh"
echo ""
