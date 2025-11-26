#!/bin/bash
#
# MASTER TRAINING SCRIPT
# Trains all three VQ-VAE models sequentially from scratch
#
# This script orchestrates the complete training pipeline:
#   1. Standard VQ-VAE (50 epochs) → experiments/1_standard_vqvae/
#   2. Masked VQ-VAE (50 epochs)   → experiments/2_masked_vqvae/
#   3. Contrastive VQ-VAE (50 epochs, uses best standard model) → experiments/3_contrastive_vqvae/
#
# Total estimated time: ~8-12 hours on 2 GPUs
#

set -e  # Exit on error

# Set project root
PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================="
echo "COMPLETE VQ-VAE TRAINING PIPELINE"
echo "========================================="
echo ""
echo "This will train all three models from scratch:"
echo "  1. Standard VQ-VAE     (50 epochs, ~4-5 hours)"
echo "  2. Masked VQ-VAE       (50 epochs, ~4-5 hours)"
echo "  3. Contrastive VQ-VAE  (50 epochs, ~4-5 hours)"
echo ""
echo "Total estimated time: 12-15 hours on 2 GPUs"
echo ""
echo "GPU Configuration:"
echo "  • Using GPUs: 0, 1 (batch size: 32)"
echo "  • Memory management: PYTORCH_CUDA_ALLOC_CONF enabled"
echo ""
echo "Output directories:"
echo "  • experiments/1_standard_vqvae/"
echo "  • experiments/2_masked_vqvae/"
echo "  • experiments/3_contrastive_vqvae/"
echo ""
echo -e "${YELLOW}Press Ctrl+C within 10 seconds to cancel...${NC}"
sleep 10

# ============================================================================
# STEP 1: Train Standard VQ-VAE
# ============================================================================

echo ""
echo "========================================="
echo "STEP 1/3: Training Standard VQ-VAE"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

./scripts/1_train_standard.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Standard VQ-VAE training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 1 complete!${NC}"
echo ""

# ============================================================================
# STEP 2: Train Masked VQ-VAE
# ============================================================================

echo ""
echo "========================================="
echo "STEP 2/3: Training Masked VQ-VAE"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

./scripts/2_train_masked.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Masked VQ-VAE training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 2 complete!${NC}"
echo ""

# ============================================================================
# STEP 3: Train Contrastive VQ-VAE
# ============================================================================

echo ""
echo "========================================="
echo "STEP 3/3: Training Contrastive VQ-VAE"
echo "========================================="
echo ""
echo "Start time: $(date)"
echo ""

./scripts/3_train_contrastive.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Contrastive VQ-VAE training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Step 3 complete!${NC}"
echo ""

# ============================================================================
# COMPLETE
# ============================================================================

echo ""
echo "========================================="
echo "✓✓✓ ALL TRAINING COMPLETE! ✓✓✓"
echo "========================================="
echo ""
echo "End time: $(date)"
echo ""
echo "Trained models:"
echo "  ✓ Standard VQ-VAE:     experiments/1_standard_vqvae/checkpoints/best_model.pt"
echo "  ✓ Masked VQ-VAE:       experiments/2_masked_vqvae/checkpoints/best_model.pt"
echo "  ✓ Contrastive VQ-VAE:  experiments/3_contrastive_vqvae/checkpoints/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate individual models:"
echo "     ./scripts/4_evaluate_standard.sh"
echo "     ./scripts/5_evaluate_masked.sh"
echo "     ./scripts/6_evaluate_contrastive.sh"
echo ""
echo "  2. Run unified comparison:"
echo "     ./scripts/7_compare_all.sh"
echo ""
echo "========================================="
echo ""
