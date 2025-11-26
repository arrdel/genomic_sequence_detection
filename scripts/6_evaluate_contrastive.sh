#!/bin/bash
#
# Evaluate Contrastive VQ-VAE
# Computes embedding quality metrics (alignment, uniformity, clustering)
#
# Input: experiments/3_contrastive_vqvae/checkpoints/best_model.pt
# Output: experiments/3_contrastive_vqvae/results/
#

set -e  # Exit on error

echo "========================================="
echo "EVALUATING CONTRASTIVE VQ-VAE"
echo "========================================="
echo ""

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Check if checkpoints exist
CONTRASTIVE_CHECKPOINT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
STANDARD_CHECKPOINT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/experiments/1_standard_vqvae/checkpoints/best_model.pt"

if [ ! -f "$CONTRASTIVE_CHECKPOINT" ]; then
    echo "ERROR: Contrastive VQ-VAE checkpoint not found!"
    echo "Expected: $CONTRASTIVE_CHECKPOINT"
    echo ""
    echo "Please run ./scripts/3_train_contrastive.sh first."
    exit 1
fi

if [ ! -f "$STANDARD_CHECKPOINT" ]; then
    echo "ERROR: Standard VQ-VAE checkpoint not found!"
    echo "Expected: $STANDARD_CHECKPOINT"
    echo ""
    echo "Contrastive model requires the base standard VQ-VAE checkpoint."
    echo "Please ensure ./scripts/1_train_standard.sh was run successfully."
    exit 1
fi

echo "Configuration:"
echo "  • Contrastive checkpoint: $CONTRASTIVE_CHECKPOINT"
echo "  • Base checkpoint: $STANDARD_CHECKPOINT"
echo "  • Data: data/cleaned_reads.fastq"
echo "  • Output: experiments/3_contrastive_vqvae/results/"
echo ""

# Run evaluation
python -u scripts/contrastive_evaluate.py \
    --checkpoint-path "$CONTRASTIVE_CHECKPOINT" \
    --vqvae-checkpoint-path "$STANDARD_CHECKPOINT" \
    --data-path data/cleaned_reads.fastq \
    --output-dir experiments/3_contrastive_vqvae/results \
    --num-eval-samples 10000 \
    --batch-size 128 \
    --max-seq-length 150 \
    --k-mer 6 \
    --gpu-id 0

echo ""
echo "========================================="
echo "✓ Contrastive VQ-VAE evaluation complete!"
echo "========================================="
echo ""
echo "Results saved to: experiments/3_contrastive_vqvae/results/"
echo "  • evaluation_results.json"
echo "  • embeddings_umap.png"
echo ""
