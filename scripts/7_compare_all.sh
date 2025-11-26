#!/bin/bash
#
# Unified Model Comparison
# Compares all three VQ-VAE approaches on the same test set
#
# Inputs:
#   - experiments/1_standard_vqvae/checkpoints/best_model.pt
#   - experiments/2_masked_vqvae/checkpoints/best_model.pt
#   - experiments/3_contrastive_vqvae/checkpoints/best_model.pt
#
# Output: experiments/4_final_comparison/
#

set -e  # Exit on error

echo "========================================="
echo "UNIFIED MODEL COMPARISON"
echo "========================================="
echo ""
echo "Comparing four VQ-VAE approaches:"
echo "  1. Standard VQ-VAE (full reconstruction)"
echo "  2. Masked VQ-VAE (BERT-style MLM)"
echo "  3. Contrastive VQ-VAE (SimCLR-style, 64-dim)"
echo "  4. Contrastive VQ-VAE 128-dim (matches masked dimensionality)"
echo ""

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Check if all checkpoints exist
STANDARD_CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
MASKED_CHECKPOINT="experiments/2_masked_vqvae/checkpoints/best_model.pt"
CONTRASTIVE_CHECKPOINT="experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
CONTRASTIVE_128_CHECKPOINT="experiments/3_contrastive_vqvae_128dim/checkpoints/best_model.pt"

MISSING=0

if [ ! -f "$STANDARD_CHECKPOINT" ]; then
    echo "❌ Standard VQ-VAE checkpoint not found: $STANDARD_CHECKPOINT"
    MISSING=1
fi

if [ ! -f "$MASKED_CHECKPOINT" ]; then
    echo "❌ Masked VQ-VAE checkpoint not found: $MASKED_CHECKPOINT"
    MISSING=1
fi

if [ ! -f "$CONTRASTIVE_CHECKPOINT" ]; then
    echo "❌ Contrastive VQ-VAE checkpoint not found: $CONTRASTIVE_CHECKPOINT"
    MISSING=1
fi

if [ ! -f "$CONTRASTIVE_128_CHECKPOINT" ]; then
    echo "❌ Contrastive VQ-VAE 128-dim checkpoint not found: $CONTRASTIVE_128_CHECKPOINT"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: One or more checkpoints are missing!"
    echo ""
    echo "Training workflow:"
    echo "  1. ./scripts/1_train_standard.sh"
    echo "  2. ./scripts/2_train_masked.sh"
    echo "  3. ./scripts/3_train_contrastive.sh"
    echo "  4. ./scripts/train_contrastive_128dim.sh"
    echo ""
    exit 1
fi

echo "✓ All checkpoints found"
echo ""
echo "Configuration:"
echo "  • Standard: $STANDARD_CHECKPOINT"
echo "  • Masked: $MASKED_CHECKPOINT"
echo "  • Contrastive 64-dim: $CONTRASTIVE_CHECKPOINT"
echo "  • Contrastive 128-dim: $CONTRASTIVE_128_CHECKPOINT"
echo "  • Data: data/cleaned_reads.fastq"
echo "  • Samples: 10,000"
echo "  • Output: experiments/4_final_comparison/"
echo ""

# Run unified comparison (3 models: Standard, Masked, Contrastive 64-dim)
echo "Running comparison for first 3 models..."
python -u scripts/evaluation.py \
    --standard-checkpoint "$STANDARD_CHECKPOINT" \
    --masked-checkpoint "$MASKED_CHECKPOINT" \
    --contrastive-checkpoint "$CONTRASTIVE_CHECKPOINT" \
    --data-path data/cleaned_reads.fastq \
    --output-dir experiments/4_final_comparison \
    --num-samples 10000 \
    --batch-size 64 \
    --max-seq-length 150 \
    --k-mer 6 \
    --num-clusters 10 \
    --gpu-id 2

echo ""
echo "✓ 3-model comparison complete"
echo ""

# Run evaluation for 128-dim contrastive separately
echo "Running evaluation for Contrastive 128-dim..."
./scripts/12_evaluate_contrastive_128dim_with_taxa.sh

echo ""
echo "✓ Contrastive 128-dim evaluation complete"
echo ""

# Generate 4-model side-by-side visualizations
echo "Generating 4-model UMAP and t-SNE visualizations..."
python -u scripts/create_4model_visualization.py \
    --standard-embeddings experiments/4_final_comparison/standard_embeddings.npy \
    --masked-embeddings experiments/4_final_comparison/masked_embeddings.npy \
    --contrastive-64-embeddings experiments/4_final_comparison/contrastive_embeddings.npy \
    --contrastive-128-embeddings experiments/3_contrastive_vqvae_128dim/results_genus_labeled/embeddings.npy \
    --standard-labels experiments/4_final_comparison/standard_labels.npy \
    --masked-labels experiments/4_final_comparison/masked_labels.npy \
    --contrastive-64-labels experiments/4_final_comparison/contrastive_labels.npy \
    --contrastive-128-labels experiments/3_contrastive_vqvae_128dim/results_genus_labeled/cluster_labels.npy \
    --output-dir experiments/4_final_comparison \
    --sample-size 5000

echo ""
echo "✓ 4-model visualizations complete"
echo ""

echo ""
echo "========================================="
echo "✓ ALL COMPARISONS COMPLETE!"
echo "========================================="
echo ""
echo "Results locations:"
echo "  • experiments/4_final_comparison/           - 3-model unified comparison"
echo "  • experiments/3_contrastive_vqvae_128dim/   - Contrastive 128-dim evaluation"
echo ""
echo "3-Model Comparison Files (Standard, Masked, Contrastive 64-dim):"
echo "  • comparison_results.json     - All metrics in JSON format"
echo "  • comparison_table.txt        - Formatted comparison table"
echo "  • comparison_table.csv        - CSV format table"
echo "  • metrics_comparison.png      - Bar chart visualization"
echo "  • umap_comparison.png         - UMAP embeddings (3 models)"
echo "  • tsne_comparison.png         - t-SNE embeddings (3 models)"
echo ""
echo "4-Model Visualizations (All Models Side-by-Side):"
echo "  • 4model_umap_comparison.png  - UMAP comparison of all 4 models"
echo "  • 4model_tsne_comparison.png  - t-SNE comparison of all 4 models"
echo ""
echo "Contrastive 128-dim Files:"
echo "  • results_genus_labeled/embeddings_umap_genus_labeled.png"
echo "  • results_genus_labeled/embeddings_tsne_genus_labeled.png"
echo "  • results_genus_labeled/cluster_genus_composition.png"
echo "  • results_genus_labeled/cluster_genus_labels.json"
echo ""
echo "Comparison Summary:"
echo "  Model                    | Dim  | Projection | Normalization"
echo "  -------------------------|------|------------|---------------"
echo "  Standard VQ-VAE          | 128  | None       | No"
echo "  Masked VQ-VAE            | 128  | None       | No"
echo "  Contrastive 64-dim       | 64   | Yes        | Yes (L2)"
echo "  Contrastive 128-dim      | 128  | Yes        | Yes (L2)"
echo ""
echo "Expected clustering quality (best to worst):"
echo "  1. Masked VQ-VAE (128-dim, reconstruction objective)"
echo "  2. Contrastive 128-dim (higher capacity than 64-dim)"
echo "  3. Standard VQ-VAE (128-dim, but different training)"
echo "  4. Contrastive 64-dim (compressed, uniformity effects)"
echo ""
