#!/bin/bash
#
# Visualize Contrastive Learning Relationships (128-dim)
#
# Creates visualizations showing:
# - 2D embedding space with colored clusters
# - Positive pairs (green arrows) - same sequence, different augmentations
# - Negative pairs (red dashes) - different sequences
# - Similarity heatmap
#
# Usage:
#   ./scripts/visualize_contrastive_128dim.sh
#

set -e

echo "=========================================="
echo "Contrastive Learning Relationship Visualization (128-dim)"
echo "=========================================="
echo ""

# Configuration
CONTRASTIVE_CHECKPOINT="experiments/3_contrastive_vqvae_128dim/checkpoints/best_model.pt"
VQVAE_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
OUTPUT_DIR="experiments/3_contrastive_vqvae_128dim/relationship_viz"
PROJ_DIM=128
NUM_SAMPLES=500
NUM_PAIRS=50
NUM_DISPLAY_PAIRS=30
NUM_CLUSTERS=10

echo "Configuration:"
echo "  Contrastive checkpoint: $CONTRASTIVE_CHECKPOINT"
echo "  VQ-VAE checkpoint: $VQVAE_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Projection dimension: $PROJ_DIM (matches masked VQ-VAE)"
echo "  Samples: $NUM_SAMPLES"
echo "  Pairs to generate: $NUM_PAIRS"
echo "  Pairs to display: $NUM_DISPLAY_PAIRS"
echo "  Clusters: $NUM_CLUSTERS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CONTRASTIVE_CHECKPOINT" ]; then
    echo "ERROR: Contrastive 128-dim checkpoint not found: $CONTRASTIVE_CHECKPOINT"
    echo "Please train the 128-dim contrastive model first:"
    echo "  ./scripts/train_contrastive_128dim.sh"
    exit 1
fi

if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found: $VQVAE_CHECKPOINT"
    echo "Please train the VQ-VAE model first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting visualization..."
echo ""

python -u scripts/visualize_contrastive_relationships.py \
    --checkpoint-path "$CONTRASTIVE_CHECKPOINT" \
    --vqvae-checkpoint-path "$VQVAE_CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --proj-dim "$PROJ_DIM" \
    --num-samples "$NUM_SAMPLES" \
    --num-pairs "$NUM_PAIRS" \
    --num-display-pairs "$NUM_DISPLAY_PAIRS" \
    --num-clusters "$NUM_CLUSTERS" \
    --seed 42

echo ""
echo "=========================================="
echo "✓ Visualization Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated visualizations:"
echo "  • contrastive_relationships_tsne.png"
echo "  • contrastive_relationships_umap.png (if UMAP installed)"
echo "  • similarity_heatmap.png"
echo ""
echo "Visualization shows:"
echo "  ✓ Colored clusters in 2D embedding space"
echo "  ✓ Green arrows = Positive pairs (same sequence, augmented)"
echo "  ✓ Red dashes = Negative pairs (different sequences)"
echo "  ✓ Similarity heatmap grouped by clusters"
echo ""
echo "Compare with 64-dim version:"
echo "  experiments/3_contrastive_vqvae/relationship_viz/"
echo ""
