#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Launch full downstream evaluation on ai-lab1
# Runs on CPU (no GPU needed — just numpy / sklearn)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

CONDA_ENV="/home/adelechinda/local_conda_envs/contrastive_env"
PROJECT="/home/adelechinda/projects/genomic_sequence_detection"
EXP_DIR="/home/adelechinda/data/contrastive/experiments"
OUT_DIR="${EXP_DIR}/full_evaluation"

echo "═══════════════════════════════════════════════════════════════"
echo "  Full Downstream Evaluation — all 8 models × 5 metric suites"
echo "═══════════════════════════════════════════════════════════════"

source ~/miniconda3/bin/activate
conda activate "${CONDA_ENV}"
cd "${PROJECT}"

python -u scripts/run_full_downstream_eval.py \
    --experiments-dir "${EXP_DIR}" \
    --output-dir "${OUT_DIR}" \
    --max-samples 10000 \
    --clustering-methods kmeans \
    --n-clusters 5 10 15 20 \
    --seeds 42 123 456

echo ""
echo "✓ Done.  Results in: ${OUT_DIR}"
