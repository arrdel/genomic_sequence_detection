#!/bin/bash
#
# Evaluate Masked VQ-VAE
# Computes reconstruction metrics and masked prediction accuracy
#
# Input: experiments/2_masked_vqvae/checkpoints/best_model.pt
# Output: experiments/2_masked_vqvae/results/
#

set -e  # Exit on error

echo "========================================="
echo "EVALUATING MASKED VQ-VAE"
echo "========================================="
echo ""

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Check if checkpoint exists
CHECKPOINT="experiments/2_masked_vqvae/checkpoints/best_model.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Masked VQ-VAE checkpoint not found!"
    echo "Expected: $CHECKPOINT"
    echo ""
    echo "Please run ./scripts/2_train_masked.sh first."
    exit 1
fi

echo "Configuration:"
echo "  • Checkpoint: $CHECKPOINT"
echo "  • Data: data/cleaned_reads.fastq"
echo "  • Output: experiments/2_masked_vqvae/results/"
echo ""

# Run evaluation
python -u scripts/mqvae_evaluate.py \
    --checkpoint-path "$CHECKPOINT" \
    --data-path data/cleaned_reads.fastq \
    --output-dir experiments/2_masked_vqvae/results \
    --num-samples 10000 \
    --batch-size 128 \
    --max-seq-length 150 \
    --k-mer 6 \
    --mask-prob 0.2 \
    --gpu-id 0

echo ""
echo "========================================="
echo "✓ Masked VQ-VAE evaluation complete!"
echo "========================================="
echo ""
echo "Results saved to: experiments/2_masked_vqvae/results/"
echo "  • evaluation_results.json"
echo "  • sequence_reconstructions.txt"
echo ""
