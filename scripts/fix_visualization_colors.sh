#!/bin/bash
#
# Quick Fix: Generate Cluster Labels and 4-Model Visualizations
#
# This script:
# 1. Generates cluster labels from existing embeddings (if missing)
# 2. Creates 4-model UMAP and t-SNE visualizations
#
# Use this if you already ran the comparison but the labels are missing.
#

set -e

echo "=========================================="
echo "QUICK FIX: Generate Labels & Visualizations"
echo "=========================================="
echo ""

# Check if embeddings exist
COMPARISON_DIR="experiments/4_final_comparison"
CONTRASTIVE_128_DIR="experiments/3_contrastive_vqvae_128dim/results_genus_labeled"

if [ ! -f "$COMPARISON_DIR/standard_embeddings.npy" ]; then
    echo "ERROR: Embeddings not found in $COMPARISON_DIR"
    echo "Please run ./scripts/7_compare_all.sh first"
    exit 1
fi

echo "Step 1: Generating cluster labels from embeddings..."
echo ""

python scripts/generate_cluster_labels.py \
    --embeddings-dir "$COMPARISON_DIR" \
    --num-clusters 10

echo ""
echo "Step 2: Creating 4-model visualizations..."
echo ""

python scripts/create_4model_visualization.py \
    --standard-embeddings "$COMPARISON_DIR/standard_embeddings.npy" \
    --masked-embeddings "$COMPARISON_DIR/masked_embeddings.npy" \
    --contrastive-64-embeddings "$COMPARISON_DIR/contrastive_embeddings.npy" \
    --contrastive-128-embeddings "$CONTRASTIVE_128_DIR/embeddings.npy" \
    --standard-labels "$COMPARISON_DIR/standard_labels.npy" \
    --masked-labels "$COMPARISON_DIR/masked_labels.npy" \
    --contrastive-64-labels "$COMPARISON_DIR/contrastive_labels.npy" \
    --contrastive-128-labels "$CONTRASTIVE_128_DIR/cluster_labels.npy" \
    --output-dir "$COMPARISON_DIR" \
    --sample-size 5000

echo ""
echo "=========================================="
echo "✓ COMPLETE!"
echo "=========================================="
echo ""
echo "Generated visualizations:"
echo "  • $COMPARISON_DIR/4model_umap_comparison.png"
echo "  • $COMPARISON_DIR/4model_tsne_comparison.png"
echo ""
echo "All 4 models now have colored clusters!"
echo ""
