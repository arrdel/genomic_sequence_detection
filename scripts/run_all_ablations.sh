#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run all ablation studies, each as a separate torchrun process.
# This avoids SIGSEGV from accumulated state when running many
# sequential model create/destroy cycles in one Python process.
#
# Uses --resume to skip already-completed configs.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

CONDA_ENV="/path/to/conda/env"
PROJECT_DIR="$HOME/projects/genomic_sequence_detection"
DATA="$HOME/data/contrastive/processed/subset_1M.fastq"
EXP_DIR="$HOME/data/contrastive/experiments/ablations"

source ~/miniconda3/bin/activate
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=2,3,4,5

STUDIES=(codebook_size code_dim kmer_size loss_components masking)

echo "════════════════════════════════════════════════════════"
echo " ABLATION: Running each study as a separate process"
echo " Studies: ${STUDIES[*]}"
echo "════════════════════════════════════════════════════════"

for STUDY in "${STUDIES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ">>> Starting study: ${STUDY}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    torchrun --nproc_per_node=4 --master_port=29501 \
        scripts/run_ablation_study.py \
        --data-path "${DATA}" \
        --output-dir "${EXP_DIR}" \
        --study "${STUDY}" \
        --override-batch-size 512 \
        --resume

    echo ">>> Finished study: ${STUDY}"
done

echo ""
echo "✓ ALL ABLATION STUDIES DONE"
