#!/bin/bash
#
# Evaluate Contrastive VQ-VAE (128-dim) with Genus-Level Taxonomic Labels
#
# Usage:
#   ./scripts/12_evaluate_contrastive_128dim_with_taxa.sh
#

set -e

CHECKPOINT="experiments/3_contrastive_vqvae_128dim/checkpoints/best_model.pt"
VQVAE_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
KRAKEN_CLASSIFICATIONS="data/sequence_classification/kraken_classifications.txt"
KRAKEN_REPORT="data/sequence_classification/kraken_report.txt"
OUTPUT_DIR="experiments/3_contrastive_vqvae_128dim/results_genus_labeled"
K_MER=6
NUM_CLUSTERS=10
NUM_EVAL_SAMPLES=10000
BATCH_SIZE=128

echo "=========================================="
echo "Contrastive VQ-VAE (128-dim): Genus-Level Taxonomic Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Contrastive checkpoint: $CHECKPOINT"
echo "  VQ-VAE checkpoint: $VQVAE_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Projection dimension: 128 (matches masked VQ-VAE)"
echo "  Number of clusters: $NUM_CLUSTERS"
echo "  Evaluation samples: $NUM_EVAL_SAMPLES"
echo "  Taxonomy level: GENUS"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Contrastive checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found: $VQVAE_CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting genus-level taxonomic evaluation..."
echo ""

python scripts/label_contrastive_128dim_clusters_with_taxa.py \
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
echo "  • embeddings.npy                          - 128-dim normalized embeddings"
echo "  • cluster_labels.npy                      - K-means cluster assignments"
echo "  • embeddings_umap_genus_labeled.png       - UMAP visualization with genus labels"
echo "  • embeddings_tsne_genus_labeled.png       - t-SNE visualization with genus labels"
echo "  • cluster_genus_composition.png           - Bar chart of top genera per cluster"
echo "  • cluster_genus_labels.json               - Detailed metrics and statistics"
echo ""
echo "Visualizations:"
echo "  UMAP: $OUTPUT_DIR/embeddings_umap_genus_labeled.png"
echo "  t-SNE: $OUTPUT_DIR/embeddings_tsne_genus_labeled.png"
echo "  Composition: $OUTPUT_DIR/cluster_genus_composition.png"
echo ""
echo "Comparison Summary:"
echo "  Masked VQ-VAE:          128-dim, no projection, no normalization"
echo "  Contrastive 64-dim:     64-dim, projection + normalization"
echo "  Contrastive 128-dim:    128-dim, projection + normalization (NEW)"
echo ""
echo "Expected Results:"
echo "  • Better clustering than 64-dim (higher dimensionality)"
echo "  • May still show uniformity effects vs masked (normalization)"
echo "  • Lower entropy than 64-dim contrastive"
echo "  • Higher cluster purity than 64-dim contrastive"
echo ""
