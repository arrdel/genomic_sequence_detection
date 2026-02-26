#!/bin/bash
# ==========================================================================
# masked_vq_train.sh  —  Complete overnight training pipeline
#
#  Phase 1: Core models  (VQ-VAE, Masked VQ-VAE, Contrastive 64d, 128d)
#  Phase 2: Extract embeddings from core models
#  Phase 3: Baselines    (Autoencoder, Transformer VAE, DNABERT-2, K-mer PCA)
#  Phase 4: Ablations    (codebook K, code dim D, mask prob, temperature)
#  Phase 5: Multi-seed   (5 seeds × 3 models for significance)
#  Phase 6: Evaluation   (unified + per-model detailed)
#
#  GPU: uses GPUs 4 and 5 only
#
#  Usage:
#    nohup bash scripts/masked_vq_train.sh &
# ==========================================================================
set +e  # keep going if a step fails

# ---------- activate conda ----------
eval "$(conda shell.bash hook)"
conda activate vqvae_env
echo "Python: $(which python)"

# ---------- paths ----------
DATA_DIR="/media/scratch/adele/contrastive"
DATA="${DATA_DIR}/processed/cleaned_reads.fastq"
EXP="${DATA_DIR}/experiments"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

# ---------- GPU config ----------
# All 6 GPUs are visible; we target physical GPUs 4 and 5.
N_GPU=2
GPUS="4,5"
GPU0=4                 # first GPU — used for single-GPU scripts

BS=32
SEED=42
N_EMB=10000            # samples for embedding extraction

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${EXP}/logs"; mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/masked_vq_train_${TS}.log"

cd "${PROJ}"
exec > >(tee -a "${LOG}") 2>&1

echo "=========================================="
echo " masked_vq_train  |  $(date)"
echo "=========================================="
echo " data:  ${DATA}"
echo " exp:   ${EXP}"
echo " gpus:  ${GPUS} (${N_GPU})"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " conda: $(conda info --envs | grep '*')"
echo " log:   ${LOG}"
echo ""

T0=$(date +%s)
declare -a PASS FAIL

step() {
    local nm="$1"; shift
    echo ""; echo "--- ${nm} --- $(date) ---"
    local s=$(date +%s)
    "$@"; local rc=$?
    local d=$(( $(date +%s) - s ))
    if [ ${rc} -eq 0 ]; then
        echo "  OK   ${nm}  (${d}s)"; PASS+=("${nm}")
    else
        echo "  FAIL ${nm}  rc=${rc}  (${d}s)"; FAIL+=("${nm}")
    fi
}

best() { find "$1" -name best_model.pt -type f 2>/dev/null | head -1; }

# ==========================================================================
#  PHASE 1  —  Core Models
# ==========================================================================
echo ""; echo "==== PHASE 1: Core Models ===="

step "P1 VQ-VAE" \
  python -u scripts/vqvae_train.py \
    --data-path "${DATA}" \
    --output-dir "${EXP}/1_standard_vqvae" \
    --experiment-name standard_vqvae \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
    --batch-size ${BS} --epochs 50 --seed ${SEED} --no-wandb

step "P1 Masked VQ-VAE" \
  python -u scripts/mqvae_train.py \
    --data-path "${DATA}" \
    --output-dir "${EXP}/2_masked_vqvae" \
    --experiment-name masked_vqvae \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
    --batch-size ${BS} --epochs 50 --seed ${SEED} \
    --mask-prob 0.2 --no-wandb

VQ=$(best "${EXP}/1_standard_vqvae")

if [ -n "${VQ}" ]; then
  step "P1 Contrastive 64d" \
    python -u scripts/contrastive_finetune.py \
      --data-path "${DATA}" \
      --checkpoint-path "${VQ}" \
      --output-dir "${EXP}/3_contrastive_64dim" \
      --experiment-name contrastive_64dim \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size 64 --epochs 10 --proj-dim 64 --seed ${SEED}

  step "P1 Contrastive 128d" \
    python -u scripts/contrastive_finetune_128dim.py \
      --data-path "${DATA}" \
      --checkpoint-path "${VQ}" \
      --output-dir "${EXP}/4_contrastive_128dim" \
      --experiment-name contrastive_128dim \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size 64 --epochs 10 --proj-dim 128 --seed ${SEED}
else
  echo "  SKIP contrastive — no VQ-VAE checkpoint"
  FAIL+=("P1 Contrastive 64d (skip)") FAIL+=("P1 Contrastive 128d (skip)")
fi

# ==========================================================================
#  PHASE 2  —  Extract Embeddings
# ==========================================================================
echo ""; echo "==== PHASE 2: Extract Embeddings ===="

CK=$(best "${EXP}/1_standard_vqvae")
[ -n "${CK}" ] && step "P2 Embed VQ-VAE" \
  python -u scripts/extract_embeddings.py \
    --model-type vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/1_standard_vqvae/standard_vqvae_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

CK=$(best "${EXP}/2_masked_vqvae")
[ -n "${CK}" ] && step "P2 Embed Masked" \
  python -u scripts/extract_embeddings.py \
    --model-type masked_vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/2_masked_vqvae/masked_vqvae_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

VQ=$(best "${EXP}/1_standard_vqvae")

CK=$(best "${EXP}/3_contrastive_64dim")
[ -n "${CK}" ] && [ -n "${VQ}" ] && step "P2 Embed Contrastive 64d" \
  python -u scripts/extract_embeddings.py \
    --model-type contrastive --checkpoint "${CK}" \
    --vqvae-checkpoint "${VQ}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/3_contrastive_64dim/contrastive_64dim_embeddings.npy" \
    --num-samples ${N_EMB} --proj-dim 64 --gpu-id ${GPU0}

CK=$(best "${EXP}/4_contrastive_128dim")
[ -n "${CK}" ] && [ -n "${VQ}" ] && step "P2 Embed Contrastive 128d" \
  python -u scripts/extract_embeddings.py \
    --model-type contrastive --checkpoint "${CK}" \
    --vqvae-checkpoint "${VQ}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/4_contrastive_128dim/contrastive_128dim_embeddings.npy" \
    --num-samples ${N_EMB} --proj-dim 128 --gpu-id ${GPU0}

# ==========================================================================
#  PHASE 3  —  Baselines
# ==========================================================================
echo ""; echo "==== PHASE 3: Baselines ===="

step "P3 Autoencoder" \
  python -u scripts/train_autoencoder.py \
    --data-path "${DATA}" \
    --output-dir "${EXP}/baselines/autoencoder" \
    --epochs 50 --batch-size ${BS} \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}" --seed ${SEED}

step "P3 Transformer VAE" \
  python -u scripts/train_transformer_vae.py \
    --data-path "${DATA}" \
    --output-dir "${EXP}/baselines/transformer_vae" \
    --epochs 50 --batch-size ${BS} \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}" --seed ${SEED}

step "P3 DNABERT-2" \
  python -u scripts/run_experiment.py baseline-dnabert2 \
    --data-path "${DATA}" \
    --output-dir "${EXP}/baselines/dnabert2" \
    --batch-size ${BS} --seed ${SEED} --num-samples ${N_EMB} \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}"

step "P3 K-mer PCA" \
  python -u scripts/run_experiment.py baseline-kmer-pca \
    --data-path "${DATA}" \
    --output-dir "${EXP}/baselines/kmer_pca" \
    --batch-size ${BS} --seed ${SEED} --num-samples ${N_EMB} \
    --n-gpu ${N_GPU} --gpu-ids "${GPUS}"

CK=$(best "${EXP}/baselines/autoencoder")
[ -n "${CK}" ] && step "P3 Embed AE" \
  python -u scripts/extract_embeddings.py \
    --model-type autoencoder --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/baselines/autoencoder/autoencoder_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

CK=$(best "${EXP}/baselines/transformer_vae")
[ -n "${CK}" ] && step "P3 Embed TVAE" \
  python -u scripts/extract_embeddings.py \
    --model-type transformer_vae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${EXP}/baselines/transformer_vae/transformer_vae_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

# ==========================================================================
#  PHASE 4  —  Ablations  (20 epochs each)
# ==========================================================================
echo ""; echo "==== PHASE 4: Ablations ===="
AE=20

# codebook size
for K in 64 128 256 512 1024 2048; do
  D="${EXP}/ablations/codebook_size/K${K}"
  step "P4 K=${K}" \
    python -u scripts/vqvae_train.py \
      --data-path "${DATA}" --output-dir "${D}" \
      --experiment-name "K${K}" \
      --num-codes ${K} --code-dim 64 \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size ${BS} --epochs ${AE} --seed ${SEED} --no-wandb
  CK=$(best "${D}")
  [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
    --model-type vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${D}/ablation_K${K}_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}
done

# code dimension
for DM in 16 32 64 128 256; do
  D="${EXP}/ablations/code_dim/D${DM}"
  step "P4 D=${DM}" \
    python -u scripts/vqvae_train.py \
      --data-path "${DATA}" --output-dir "${D}" \
      --experiment-name "D${DM}" \
      --num-codes 512 --code-dim ${DM} \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size ${BS} --epochs ${AE} --seed ${SEED} --no-wandb
  CK=$(best "${D}")
  [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
    --model-type vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${D}/ablation_D${DM}_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}
done

# masking probability
for MP in 0.05 0.10 0.15 0.20 0.30 0.40; do
  TAG=$(echo ${MP} | tr . p)
  D="${EXP}/ablations/mask_prob/MP${TAG}"
  step "P4 mask=${MP}" \
    python -u scripts/mqvae_train.py \
      --data-path "${DATA}" --output-dir "${D}" \
      --experiment-name "MP${TAG}" \
      --mask-prob ${MP} \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size ${BS} --epochs ${AE} --seed ${SEED} --no-wandb
  CK=$(best "${D}")
  [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
    --model-type masked_vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${D}/ablation_MP${TAG}_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}
done

# contrastive temperature
VQ=$(best "${EXP}/1_standard_vqvae")
if [ -n "${VQ}" ]; then
  for TAU in 0.1 0.3 0.5 0.7 1.0; do
    TAG=$(echo ${TAU} | tr . p)
    D="${EXP}/ablations/temperature/T${TAG}"
    step "P4 tau=${TAU}" \
      python -u scripts/contrastive_finetune.py \
        --data-path "${DATA}" \
        --checkpoint-path "${VQ}" \
        --output-dir "${D}" \
        --experiment-name "T${TAG}" \
        --temperature ${TAU} --proj-dim 64 \
        --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
        --batch-size 64 --epochs 10 --seed ${SEED}
    CK=$(best "${D}")
    [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
      --model-type contrastive --checkpoint "${CK}" \
      --vqvae-checkpoint "${VQ}" \
      --data-path "${DATA}" \
      --output-path "${D}/ablation_T${TAG}_embeddings.npy" \
      --num-samples ${N_EMB} --proj-dim 64 --gpu-id ${GPU0}
  done
fi

# ==========================================================================
#  PHASE 5  —  Multi-Seed  (20 ep VQ/Masked, 10 ep Contrastive)
# ==========================================================================
echo ""; echo "==== PHASE 5: Multi-Seed ===="
MS=20

for S in 42 123 456 789 1024; do
  echo "--- seed ${S} ---"

  D="${EXP}/multi_seed/vqvae_s${S}"
  step "P5 VQ s=${S}" \
    python -u scripts/vqvae_train.py \
      --data-path "${DATA}" --output-dir "${D}" \
      --experiment-name "vqvae_s${S}" \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size ${BS} --epochs ${MS} --seed ${S} --no-wandb
  CK=$(best "${D}")
  [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
    --model-type vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${D}/vqvae_s${S}_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

  D="${EXP}/multi_seed/masked_s${S}"
  step "P5 Masked s=${S}" \
    python -u scripts/mqvae_train.py \
      --data-path "${DATA}" --output-dir "${D}" \
      --experiment-name "masked_s${S}" \
      --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
      --batch-size ${BS} --epochs ${MS} --seed ${S} \
      --mask-prob 0.2 --no-wandb
  CK=$(best "${D}")
  [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
    --model-type masked_vqvae --checkpoint "${CK}" \
    --data-path "${DATA}" \
    --output-path "${D}/masked_s${S}_embeddings.npy" \
    --num-samples ${N_EMB} --gpu-id ${GPU0}

  VQ=$(best "${EXP}/1_standard_vqvae")
  if [ -n "${VQ}" ]; then
    D="${EXP}/multi_seed/contrastive_s${S}"
    step "P5 Contrastive s=${S}" \
      python -u scripts/contrastive_finetune.py \
        --data-path "${DATA}" \
        --checkpoint-path "${VQ}" \
        --output-dir "${D}" \
        --experiment-name "contrastive_s${S}" \
        --n-gpu ${N_GPU} --gpu-ids "${GPUS}" \
        --batch-size 64 --epochs 10 --proj-dim 64 --seed ${S}
    CK=$(best "${D}")
    [ -n "${CK}" ] && python -u scripts/extract_embeddings.py \
      --model-type contrastive --checkpoint "${CK}" \
      --vqvae-checkpoint "${VQ}" \
      --data-path "${DATA}" \
      --output-path "${D}/contrastive_s${S}_embeddings.npy" \
      --num-samples ${N_EMB} --proj-dim 64 --gpu-id ${GPU0}
  fi
done

# ==========================================================================
#  PHASE 6  —  Evaluation
# ==========================================================================
echo ""; echo "==== PHASE 6: Evaluation ===="

step "P6 Evaluate all" \
  python -u scripts/run_experiment.py evaluate-all --results-dir "${EXP}"

CK=$(best "${EXP}/1_standard_vqvae")
[ -n "${CK}" ] && step "P6 VQ-VAE eval" \
  python -u scripts/vqvae_evaluate.py \
    --checkpoint-path "${CK}" --data-path "${DATA}" \
    --output-dir "${EXP}/1_standard_vqvae/eval" \
    --batch-size 64 --num-eval-batches 200 \
    --gpu-id ${GPU0}

CK=$(best "${EXP}/2_masked_vqvae")
[ -n "${CK}" ] && step "P6 Masked eval" \
  python -u scripts/mqvae_evaluate.py \
    --checkpoint-path "${CK}" --data-path "${DATA}" \
    --output-dir "${EXP}/2_masked_vqvae/eval" \
    --batch-size 64 --num-eval-batches 200 --mask-prob 0.2 \
    --gpu-id ${GPU0}

VQ=$(best "${EXP}/1_standard_vqvae")

CK=$(best "${EXP}/3_contrastive_64dim")
[ -n "${CK}" ] && [ -n "${VQ}" ] && step "P6 Contrastive 64d eval" \
  python -u scripts/contrastive_evaluate.py \
    --checkpoint-path "${CK}" --vqvae-checkpoint-path "${VQ}" \
    --data-path "${DATA}" \
    --output-dir "${EXP}/3_contrastive_64dim/eval" \
    --batch-size 64 --num-eval-samples 10000 \
    --gpu-id ${GPU0}

CK=$(best "${EXP}/4_contrastive_128dim")
[ -n "${CK}" ] && [ -n "${VQ}" ] && step "P6 Contrastive 128d eval" \
  python -u scripts/contrastive_evaluate.py \
    --checkpoint-path "${CK}" --vqvae-checkpoint-path "${VQ}" \
    --data-path "${DATA}" \
    --output-dir "${EXP}/4_contrastive_128dim/eval" \
    --batch-size 64 --num-eval-samples 10000 \
    --gpu-id ${GPU0}

# ==========================================================================
#  SUMMARY
# ==========================================================================
SEC=$(( $(date +%s) - T0 ))
echo ""
echo "=========================================="
echo " DONE  |  $(date)  |  $(( SEC/3600 ))h $(( (SEC%3600)/60 ))m"
echo "=========================================="
echo " Passed: ${#PASS[@]}"
for s in "${PASS[@]}"; do echo "   OK   ${s}"; done
echo " Failed: ${#FAIL[@]}"
for s in "${FAIL[@]}"; do echo "   FAIL ${s}"; done
echo ""
echo " Embeddings:"
find "${EXP}" -name '*_embeddings.npy' 2>/dev/null | sort | while read -r f; do
  echo "   ${f#${EXP}/}"
done
echo ""
echo " Log: ${LOG}"
echo "=========================================="
