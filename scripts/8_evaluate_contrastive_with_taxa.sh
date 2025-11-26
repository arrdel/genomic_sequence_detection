#!/bin/bash

################################################################################
# Contrastive VQ-VAE Evaluation with Taxonomic Labels
# 
# This script evaluates the contrastive VQ-VAE model and generates visualizations
# with biological taxonomic labels instead of numeric cluster IDs.
#
# Output: experiments/3_contrastive_vqvae/results_with_taxa/
################################################################################

set -e  # Exit on error

echo "================================================================================================"
echo "CONTRASTIVE VQ-VAE EVALUATION WITH TAXONOMIC LABELS"
echo "================================================================================================"
echo ""

# Configuration
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Paths
CONTRASTIVE_CHECKPOINT="experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
VQVAE_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/checkpoint_epoch_50.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
KRAKEN_CLASSIFICATIONS="data/sequence_classification/kraken_classifications.txt"
KRAKEN_REPORT="data/sequence_classification/kraken_report.txt"
OUTPUT_DIR="experiments/3_contrastive_vqvae/results_with_taxa"

# Evaluation parameters
NUM_EVAL_SAMPLES=100000
NUM_CLUSTERS=10  # Match the 10 clusters from standard contrastive evaluation
MIN_TAXA_COUNT=20
BATCH_SIZE=64
GPU_ID=0

echo "Configuration:"
echo "  Contrastive checkpoint: $CONTRASTIVE_CHECKPOINT"
echo "  VQ-VAE checkpoint: $VQVAE_CHECKPOINT"
echo "  Data: $DATA_PATH"
echo "  Kraken classifications: $KRAKEN_CLASSIFICATIONS"
echo "  Kraken report: $KRAKEN_REPORT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of samples: $NUM_EVAL_SAMPLES"
echo "  Number of clusters: $NUM_CLUSTERS"
echo "  Min taxa count for viz: $MIN_TAXA_COUNT"
echo "  GPU: $GPU_ID"
echo ""

# Check if files exist
echo "Checking input files..."
for file in "$CONTRASTIVE_CHECKPOINT" "$VQVAE_CHECKPOINT" "$DATA_PATH" "$KRAKEN_CLASSIFICATIONS" "$KRAKEN_REPORT"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    echo "  ✓ $file"
done
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "================================================================================================"
echo "RUNNING EVALUATION"
echo "================================================================================================"
echo ""

python scripts/contrastive_evaluate_with_taxa.py \
    --checkpoint-path "$CONTRASTIVE_CHECKPOINT" \
    --vqvae-checkpoint-path "$VQVAE_CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --kraken-classifications "$KRAKEN_CLASSIFICATIONS" \
    --kraken-report "$KRAKEN_REPORT" \
    --output-dir "$OUTPUT_DIR" \
    --num-eval-samples $NUM_EVAL_SAMPLES \
    --num-clusters $NUM_CLUSTERS \
    --min-taxa-count $MIN_TAXA_COUNT \
    --batch-size $BATCH_SIZE \
    --gpu-id $GPU_ID \
    --k-mer 6 \
    --figsize 16 12

echo ""
echo "================================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - evaluation_results_with_taxa.json      (Comprehensive results)"
echo "  - embeddings_umap_clusters.png           (UMAP: 10 clusters with taxonomy labels)"
echo "  - embeddings_tsne_clusters.png           (t-SNE: 10 clusters with taxonomy labels)"
echo "  - embeddings_umap_taxa.png               (UMAP: colored by taxonomy)"
echo "  - embeddings_tsne_taxa.png               (t-SNE: colored by taxonomy)"
echo "  - cluster_taxonomy_composition.png       (Taxonomy breakdown per cluster)"
echo "  - embeddings.npy                         (Raw embeddings)"
echo "  - cluster_labels.npy                     (K-means cluster assignments)"
echo "  - taxa_ids.npy                           (Taxonomic IDs)"
echo "  - seq_ids.txt                            (Sequence IDs)"
echo "  - taxa_names.txt                         (Taxonomic names)"
echo ""
echo "================================================================================================"
