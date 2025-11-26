#!/bin/bash

# Shell script to evaluate Masked VQ-VAE with genus-level taxonomic labels
# Generates UMAP, t-SNE, and cluster composition visualizations

set -e

# Configuration
CHECKPOINT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/experiments/2_masked_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
KRAKEN_CLASSIFICATIONS="data/sequence_classification/kraken_classifications.txt"
KRAKEN_REPORT="data/sequence_classification/kraken_report.txt"
OUTPUT_DIR="experiments/2_masked_vqvae/results_genus_labeled"
K_MER=6
NUM_EVAL_SAMPLES=10000
NUM_CLUSTERS=10
BATCH_SIZE=128

echo "============================================"
echo "MASKED VQ-VAE: GENUS-LEVEL EVALUATION"
echo "============================================"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA_PATH"
echo "Kraken classifications: $KRAKEN_CLASSIFICATIONS"
echo "Kraken report: $KRAKEN_REPORT"
echo "Output: $OUTPUT_DIR"
echo "Taxonomy level: GENUS"
echo "Number of clusters: $NUM_CLUSTERS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please train the masked VQ-VAE first."
    exit 1
fi

# Check if Kraken files exist
if [ ! -f "$KRAKEN_CLASSIFICATIONS" ]; then
    echo "ERROR: Kraken classifications not found at $KRAKEN_CLASSIFICATIONS"
    exit 1
fi

if [ ! -f "$KRAKEN_REPORT" ]; then
    echo "ERROR: Kraken report not found at $KRAKEN_REPORT"
    exit 1
fi

# Run evaluation
python scripts/label_masked_clusters_with_taxa.py \
  --checkpoint-path "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --kraken-classifications "$KRAKEN_CLASSIFICATIONS" \
  --kraken-report "$KRAKEN_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --k-mer "$K_MER" \
  --num-eval-samples "$NUM_EVAL_SAMPLES" \
  --num-clusters "$NUM_CLUSTERS" \
  --batch-size "$BATCH_SIZE" \
  --gpu-id 0

echo ""
echo "============================================"
echo "EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Output files in: $OUTPUT_DIR"
echo "  - embeddings_umap_genus_labeled.png   : UMAP with genus labels"
echo "  - embeddings_tsne_genus_labeled.png   : t-SNE with genus labels"
echo "  - cluster_genus_composition.png       : Top 5 genera per cluster"
echo "  - cluster_genus_labels.json           : Detailed cluster information"
echo "  - embeddings.npy                      : Extracted embeddings"
echo "  - cluster_labels.npy                  : K-means cluster assignments"
echo ""
