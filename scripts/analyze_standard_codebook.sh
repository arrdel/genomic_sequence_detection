#!/bin/bash
#
# Analyze Standard VQ-VAE Codebook
#
# Creates visualizations showing:
# - Usage frequency distribution (log scale)
# - Heatmap of code activation patterns
# - Example k-mers from top-5 codes
# - Detailed statistics
#
# Usage:
#   ./scripts/analyze_standard_codebook.sh
#

set -e

echo "=========================================="
echo "Standard VQ-VAE Codebook Analysis"
echo "=========================================="
echo ""

# Configuration
CHECKPOINT="experiments/1_standard_vqvae/checkpoints/best_model.pt"
DATA_PATH="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq"
OUTPUT_DIR="experiments/1_standard_vqvae/codebook_analysis"
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
    echo "Please train the standard VQ-VAE first."
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
echo "Open the visualizations to see:"
echo "  ✓ Which codes are most/least used"
echo "  ✓ Code activation patterns in embedding space"
echo "  ✓ What k-mers each code represents"
echo "  ✓ Overall codebook utilization metrics"
echo ""
