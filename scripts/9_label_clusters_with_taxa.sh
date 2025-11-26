#!/bin/bash

# Shell script to label existing clusters with taxonomic information
# This script takes the 10 K-means clusters from contrastive_evaluate.py
# and assigns biological labels based on Kraken2 taxonomy

set -e

# Configuration
EMBEDDINGS_NPY="experiments/3_contrastive_vqvae/results/embeddings.npy"
CLUSTER_LABELS_NPY="experiments/3_contrastive_vqvae/results/cluster_labels.npy"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
KRAKEN_CLASSIFICATIONS="data/sequence_classification/kraken_classifications.txt"
KRAKEN_REPORT="data/sequence_classification/kraken_report.txt"
OUTPUT_DIR="experiments/3_contrastive_vqvae/results_labeled"
K_MER=6
NUM_EVAL_SAMPLES=100000  # Must match the number used in contrastive_evaluate.py

echo "============================================"
echo "LABELING CLUSTERS WITH TAXONOMY"
echo "============================================"
echo ""
echo "Embeddings: $EMBEDDINGS_NPY"
echo "Cluster labels: $CLUSTER_LABELS_NPY"
echo "Data: $DATA_PATH"
echo "Kraken classifications: $KRAKEN_CLASSIFICATIONS"
echo "Kraken report: $KRAKEN_REPORT"
echo "Output: $OUTPUT_DIR"
echo ""

# Run the labeling script
python scripts/label_clusters_with_taxa.py \
  --embeddings-npy "$EMBEDDINGS_NPY" \
  --cluster-labels-npy "$CLUSTER_LABELS_NPY" \
  --data-path "$DATA_PATH" \
  --kraken-classifications "$KRAKEN_CLASSIFICATIONS" \
  --kraken-report "$KRAKEN_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --k-mer "$K_MER" \
  --num-eval-samples "$NUM_EVAL_SAMPLES"

echo ""
echo "============================================"
echo "LABELING COMPLETE!"
echo "============================================"
echo ""
echo "Output files in: $OUTPUT_DIR"
echo "  - embeddings_umap_labeled.png"
echo "  - embeddings_tsne_labeled.png"
echo "  - cluster_taxonomy_composition.png"
echo "  - cluster_labels_with_taxa.json"
echo ""
