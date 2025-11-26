#!/bin/bash
#
# Analyze Masked VQ-VAE Codebook
#
# Creates visualizations showing:
# - Usage frequency distribution (log scale)
# - Heatmap of code activation patterns
# - Example k-mers from top-5 codes
# - Detailed statistics
#
# Usage:
#   ./scripts/analyze_masked_codebook.sh
#

set -e

echo "=========================================="
echo "Masked VQ-VAE Codebook Analysis"
echo "=========================================="
echo ""

# Configuration
CHECKPOINT="experiments/2_masked_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
OUTPUT_DIR="experiments/2_masked_vqvae/codebook_analysis"
NUM_SAMPLES=5000
BATCH_SIZE=128
TOP_CODES=5

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Samples to analyze: $NUM_SAMPLES"
echo "  Top codes to show: $TOP_CODES"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please train the masked VQ-VAE first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting codebook analysis..."
echo ""

python -u scripts/analyze_codebook.py \
    --checkpoint-path "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --top-codes "$TOP_CODES" \
    --seed 42

echo ""
echo "=========================================="
echo "✓ Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  • usage_frequency_distribution.png  - Linear & log scale usage plots"
echo "  • code_activation_heatmap.png       - Top 100 codes activation patterns"
echo "  • top_codes_kmers.png               - K-mer examples from top-5 codes"
echo "  • codebook_statistics.txt           - Detailed statistics report"
echo ""
echo "Note: Masked VQ-VAE typically shows lower codebook utilization"
echo "      due to fine-tuning on masked prediction objective."
echo ""
echo "Compare with standard VQ-VAE:"
echo "  experiments/1_standard_vqvae/codebook_analysis/"
echo ""
