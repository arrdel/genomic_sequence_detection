#!/bin/bash
#
# PROJECT SETUP VERIFICATION
# Run this to verify all files and directories are in place
#

set -e

PROJECT_ROOT="/home/adelechinda/home/semester_projects/fall_25/deep_learning/project"
cd "$PROJECT_ROOT"

echo "========================================="
echo "VQ-VAE PROJECT SETUP VERIFICATION"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# Check data files
echo "Checking data files..."
if [ -f "data/cleaned_reads.fastq" ]; then
    SIZE=$(du -h data/cleaned_reads.fastq | cut -f1)
    echo -e "${GREEN}✓${NC} data/cleaned_reads.fastq ($SIZE)"
else
    echo -e "${RED}✗${NC} data/cleaned_reads.fastq NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

# Check experiment directories
echo ""
echo "Checking experiment directories..."
for exp in "1_standard_vqvae" "2_masked_vqvae" "3_contrastive_vqvae" "4_final_comparison"; do
    if [ -d "experiments/$exp" ]; then
        echo -e "${GREEN}✓${NC} experiments/$exp/"
        for subdir in "checkpoints" "logs" "results"; do
            if [ -d "experiments/$exp/$subdir" ]; then
                echo -e "  ${GREEN}✓${NC} $subdir/"
            else
                echo -e "  ${RED}✗${NC} $subdir/ NOT FOUND"
                ERRORS=$((ERRORS + 1))
            fi
        done
    else
        echo -e "${RED}✗${NC} experiments/$exp/ NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check training scripts
echo ""
echo "Checking training scripts..."
for script in "train_all_models.sh" "1_train_standard.sh" "2_train_masked.sh" "3_train_contrastive.sh"; do
    if [ -x "scripts/$script" ]; then
        echo -e "${GREEN}✓${NC} scripts/$script (executable)"
    elif [ -f "scripts/$script" ]; then
        echo -e "${YELLOW}⚠${NC} scripts/$script (not executable, run: chmod +x scripts/$script)"
    else
        echo -e "${RED}✗${NC} scripts/$script NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check evaluation scripts
echo ""
echo "Checking evaluation scripts..."
for script in "4_evaluate_standard.sh" "5_evaluate_masked.sh" "6_evaluate_contrastive.sh" "7_compare_all.sh"; do
    if [ -x "scripts/$script" ]; then
        echo -e "${GREEN}✓${NC} scripts/$script (executable)"
    elif [ -f "scripts/$script" ]; then
        echo -e "${YELLOW}⚠${NC} scripts/$script (not executable, run: chmod +x scripts/$script)"
    else
        echo -e "${RED}✗${NC} scripts/$script NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check Python scripts
echo ""
echo "Checking Python scripts..."
for py_script in "vqvae_train.py" "mqvae_train.py" "contrastive_finetune.py" "vqvae_evaluate.py" "mqvae_evaluate.py" "contrastive_evaluate.py" "evaluation.py"; do
    if [ -f "scripts/$py_script" ]; then
        echo -e "${GREEN}✓${NC} scripts/$py_script"
    else
        echo -e "${RED}✗${NC} scripts/$py_script NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check documentation
echo ""
echo "Checking documentation..."
for doc in "TRAINING_WORKFLOW.md" "REORGANIZATION_SUMMARY.md" "QUICK_REFERENCE.md" "README.md"; do
    if [ -f "$doc" ]; then
        echo -e "${GREEN}✓${NC} $doc"
    else
        echo -e "${RED}✗${NC} $doc NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check source code
echo ""
echo "Checking source code..."
if [ -f "src/models/vqvae.py" ]; then
    echo -e "${GREEN}✓${NC} src/models/vqvae.py"
else
    echo -e "${RED}✗${NC} src/models/vqvae.py NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "src/data/tokenizer.py" ]; then
    echo -e "${GREEN}✓${NC} src/data/tokenizer.py"
else
    echo -e "${RED}✗${NC} src/data/tokenizer.py NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

# Check dependencies
echo ""
echo "Checking Python environment..."
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null && echo -e "${GREEN}✓${NC} PyTorch installed" || echo -e "${RED}✗${NC} PyTorch NOT installed"
python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null && echo -e "${GREEN}✓${NC} NumPy installed" || echo -e "${RED}✗${NC} NumPy NOT installed"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)" 2>/dev/null && echo -e "${GREEN}✓${NC} Scikit-learn installed" || echo -e "${RED}✗${NC} Scikit-learn NOT installed"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓${NC} $GPU_COUNT GPU(s) available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "  - $line"
    done
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found (GPU check skipped)"
fi

# Summary
echo ""
echo "========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo "========================================="
    echo ""
    echo "Your project is ready for training!"
    echo ""
    echo "Next steps:"
    echo "  1. Review TRAINING_WORKFLOW.md for complete guide"
    echo "  2. Review QUICK_REFERENCE.md for quick commands"
    echo "  3. Start training:"
    echo "     ./scripts/train_all_models.sh"
else
    echo -e "${RED}✗ $ERRORS ERROR(S) FOUND${NC}"
    echo "========================================="
    echo ""
    echo "Please fix the errors above before proceeding."
fi
echo ""
