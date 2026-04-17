#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Launch ablation studies on ai-lab1 (GPUs 2,3,4,5 via DDP)
#
# Usage:
#   bash scripts/launch_ablation_study.sh              # run ALL studies
#   bash scripts/launch_ablation_study.sh codebook_size # run one study
#   bash scripts/launch_ablation_study.sh all 5         # override to 5 epochs (smoke test)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

STUDY="${1:-all}"
OVERRIDE_EPOCHS="${2:-}"

CONDA_ENV="/path/to/conda/env"
PROJECT="/path/to/project"
DATA="/path/to/data/processed/subset_1M.fastq"
OUT_DIR="/path/to/data/experiments/ablations"

export CUDA_VISIBLE_DEVICES=2,3,4,5

echo "═══════════════════════════════════════════════════════════════"
echo "  Ablation Study: ${STUDY}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}  (4-GPU DDP)"
echo "  Data: ${DATA}"
echo "  Output: ${OUT_DIR}"
[ -n "${OVERRIDE_EPOCHS}" ] && echo "  Override epochs: ${OVERRIDE_EPOCHS}"
echo "═══════════════════════════════════════════════════════════════"

source ~/miniconda3/bin/activate
conda activate "${CONDA_ENV}"
cd "${PROJECT}"

EXTRA_ARGS=""
[ -n "${OVERRIDE_EPOCHS}" ] && EXTRA_ARGS="--override-epochs ${OVERRIDE_EPOCHS}"

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    scripts/run_ablation_study.py \
        --data-path "${DATA}" \
        --output-dir "${OUT_DIR}" \
        --study "${STUDY}" \
        --override-batch-size 512 \
        ${EXTRA_ARGS}

echo ""
echo "✓ Ablation study '${STUDY}' complete.  Results in: ${OUT_DIR}"
