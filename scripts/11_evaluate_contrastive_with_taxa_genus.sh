#!/bin/bash
#
# Evaluate Contrastive VQ-VAE with Genus-Level Taxonomic Labels
# 
# This script runs the contrastive VQ-VAE evaluation with genus-level taxonomy
# to provide cleaner, more interpretable biological patterns compared to species-level.
#
# Usage:
#   ./scripts/11_evaluate_contrastive_with_taxa_genus.sh
#

# Exit on error
set -e

# Configuration
CHECKPOINT="experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
VQVAE_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
KRAKEN_CLASSIFICATIONS="data/sequence_classification/kraken_classifications.txt"
KRAKEN_REPORT="data/sequence_classification/kraken_report.txt"
OUTPUT_DIR="experiments/3_contrastive_vqvae/results_genus_labeled"
K_MER=6
NUM_CLUSTERS=10
NUM_EVAL_SAMPLES=10000
BATCH_SIZE=128

echo "=========================================="
echo "Contrastive VQ-VAE: Genus-Level Taxonomic Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Contrastive checkpoint: $CHECKPOINT"
echo "  VQ-VAE checkpoint: $VQVAE_CHECKPOINT"
echo "  Data: $DATA_PATH"
echo "  Kraken classifications: $KRAKEN_CLASSIFICATIONS"
echo "  Kraken report: $KRAKEN_REPORT"
echo "  Output directory: $OUTPUT_DIR"
echo "  K-mer size: $K_MER"
echo "  Number of clusters: $NUM_CLUSTERS"
echo "  Evaluation samples: $NUM_EVAL_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Taxonomy level: GENUS"
echo ""

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Contrastive checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found: $VQVAE_CHECKPOINT"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$KRAKEN_CLASSIFICATIONS" ]; then
    echo "ERROR: Kraken classifications not found: $KRAKEN_CLASSIFICATIONS"
    exit 1
fi

if [ ! -f "$KRAKEN_REPORT" ]; then
    echo "ERROR: Kraken report not found: $KRAKEN_REPORT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting genus-level taxonomic evaluation..."
echo ""

python scripts/label_contrastive_clusters_with_taxa.py \
    --checkpoint-path "$CHECKPOINT" \
    --vqvae-checkpoint-path "$VQVAE_CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --kraken-classifications "$KRAKEN_CLASSIFICATIONS" \
    --kraken-report "$KRAKEN_REPORT" \
    --output-dir "$OUTPUT_DIR" \
    --k-mer "$K_MER" \
    --num-clusters "$NUM_CLUSTERS" \
    --num-eval-samples "$NUM_EVAL_SAMPLES" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  • embeddings.npy - Extracted embeddings from contrastive model"
echo "  • cluster_labels.npy - K-means cluster assignments"
echo "  • embeddings_umap_genus_labeled.png - UMAP visualization with genus labels"
echo "  • embeddings_tsne_genus_labeled.png - t-SNE visualization with genus labels"
echo "  • cluster_genus_composition.png - Bar charts of top genera per cluster"
echo "  • cluster_genus_labels.json - Detailed cluster information and metrics"
echo ""
echo "Next steps:"
echo "  1. Compare genus-level patterns with species-level (results_labeled/)"
echo "  2. Examine cluster purity (entropy values)"
echo "  3. Identify biologically meaningful clusters"
echo "  4. Use unified comparison to evaluate across all models"
echo ""
