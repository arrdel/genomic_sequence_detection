#!/bin/bash
# ==============================================================================
# OVERNIGHT TRAINING PIPELINE — Contrastive VQ-VAE for Genomic Surveillance
# ==============================================================================
# This script runs ALL training stages sequentially so they can run overnight
# on GPU without interruption. Each stage depends on the previous one.
#
# Usage:
#   nohup bash scripts/run_all_overnight.sh > training_log.txt 2>&1 &
#
# To monitor:
#   tail -f training_log.txt
#
# Estimated total runtime: ~2-4 hours on 8x A6000
# ==============================================================================

set -e  # Exit on any error

# ---- Configuration ----
export HF_HOME=/shared/achinda1/huggingface_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_DIR="/home/achinda1/projects/genomic_sequence_detection"
DATA_DIR="/home/achinda1/datasets/contrastive"
DATA_PATH="${DATA_DIR}/processed/subset_1M.fastq"  # 1M reads for main training
EXPERIMENT_DIR="${DATA_DIR}/experiments"

N_GPU=8
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE=64  # per-GPU batch size; effective batch = 64 * 8 = 512
SEED=42

# Activate conda
source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_env

cd "$PROJECT_DIR"

echo "============================================================"
echo "  OVERNIGHT TRAINING PIPELINE"
echo "  Started: $(date)"
echo "  Data:    ${DATA_PATH}"
echo "  GPUs:    ${GPU_IDS} (${N_GPU} GPUs)"
echo "  Output:  ${EXPERIMENT_DIR}"
echo "============================================================"

# ==============================================================================
# PHASE 1: Base VQ-VAE Training (50 epochs)
# ==============================================================================
VQVAE_DIR="${EXPERIMENT_DIR}/1_standard_vqvae"
echo ""
echo "============================================================"
echo "  PHASE 1: Base VQ-VAE Training"
echo "  Output: ${VQVAE_DIR}"
echo "  Started: $(date)"
echo "============================================================"

VQVAE_RESUME="${VQVAE_DIR}/vqvae_base/best_model.pt"
RESUME_FLAG=""
if [ -f "${VQVAE_RESUME}" ]; then
    echo "  Resuming from checkpoint: ${VQVAE_RESUME}"
    RESUME_FLAG="--resume ${VQVAE_RESUME}"
fi

python -u scripts/vqvae_train.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${VQVAE_DIR}" \
    --experiment-name "vqvae_base" \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --num-codes 512 \
    --code-dim 64 \
    --embed-dim 128 \
    --hidden-dim 256 \
    --learning-rate 2e-4 \
    --seed ${SEED} \
    --save-freq 5 \
    --no-wandb \
    ${RESUME_FLAG}

echo "  PHASE 1 COMPLETE: $(date)"

# ==============================================================================
# PHASE 2: Masked VQ-VAE Training (50 epochs)
# ==============================================================================
MQVAE_DIR="${EXPERIMENT_DIR}/2_masked_vqvae"
echo ""
echo "============================================================"
echo "  PHASE 2: Masked VQ-VAE Training"
echo "  Output: ${MQVAE_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python -u scripts/mqvae_train.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${MQVAE_DIR}" \
    --experiment-name "mqvae_masked" \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --num-codes 512 \
    --code-dim 128 \
    --embed-dim 128 \
    --hidden-dim 256 \
    --mask-prob 0.2 \
    --learning-rate 1e-4 \
    --seed ${SEED} \
    --save-freq 5 \
    --no-wandb

echo "  PHASE 2 COMPLETE: $(date)"

# ==============================================================================
# PHASE 3a: Contrastive Fine-tuning (64-dim projection, 10 epochs)
# ==============================================================================
CONTRASTIVE_64_DIR="${EXPERIMENT_DIR}/3_contrastive_vqvae"
VQVAE_CHECKPOINT="${VQVAE_DIR}/vqvae_base/best_model.pt"
echo ""
echo "============================================================"
echo "  PHASE 3a: Contrastive Fine-tuning (64-dim)"
echo "  Checkpoint: ${VQVAE_CHECKPOINT}"
echo "  Output: ${CONTRASTIVE_64_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python -u scripts/contrastive_finetune.py \
    --data-path "${DATA_PATH}" \
    --checkpoint-path "${VQVAE_CHECKPOINT}" \
    --output-dir "${CONTRASTIVE_64_DIR}" \
    --experiment-name "contrastive_64dim" \
    --epochs 10 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --proj-dim 64 \
    --temperature 0.5 \
    --mask-prob 0.15 \
    --drop-prob 0.10 \
    --learning-rate 1e-4 \
    --seed ${SEED} \
    --use-canonical

echo "  PHASE 3a COMPLETE: $(date)"

# ==============================================================================
# PHASE 3b: Contrastive Fine-tuning (128-dim projection, 10 epochs)
# ==============================================================================
CONTRASTIVE_128_DIR="${EXPERIMENT_DIR}/3_contrastive_vqvae_128dim"
echo ""
echo "============================================================"
echo "  PHASE 3b: Contrastive Fine-tuning (128-dim)"
echo "  Checkpoint: ${VQVAE_CHECKPOINT}"
echo "  Output: ${CONTRASTIVE_128_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python -u scripts/contrastive_finetune_128dim.py \
    --data-path "${DATA_PATH}" \
    --checkpoint-path "${VQVAE_CHECKPOINT}" \
    --output-dir "${CONTRASTIVE_128_DIR}" \
    --experiment-name "contrastive_128dim" \
    --epochs 10 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --proj-dim 128 \
    --temperature 0.5 \
    --mask-prob 0.15 \
    --drop-prob 0.10 \
    --learning-rate 1e-4 \
    --seed ${SEED} \
    --use-canonical

echo "  PHASE 3b COMPLETE: $(date)"

# ==============================================================================
# PHASE 4: Standard Autoencoder Baseline (50 epochs)
# ==============================================================================
AE_DIR="${EXPERIMENT_DIR}/baselines/autoencoder"
echo ""
echo "============================================================"
echo "  PHASE 4: Standard Autoencoder Baseline"
echo "  Output: ${AE_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python -u scripts/train_autoencoder.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${AE_DIR}" \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --latent-dim 64 \
    --learning-rate 2e-4 \
    --seed ${SEED} \
    --save-freq 10

echo "  PHASE 4 COMPLETE: $(date)"

# ==============================================================================
# PHASE 5: Transformer VAE Baseline (50 epochs)
# ==============================================================================
TVAE_DIR="${EXPERIMENT_DIR}/baselines/transformer_vae"
echo ""
echo "============================================================"
echo "  PHASE 5: Transformer VAE Baseline"
echo "  Output: ${TVAE_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python -u scripts/train_transformer_vae.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${TVAE_DIR}" \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --n-gpu ${N_GPU} \
    --gpu-ids "${GPU_IDS}" \
    --latent-dim 64 \
    --nhead 4 \
    --num-layers 2 \
    --learning-rate 2e-4 \
    --beta 0.1 \
    --seed ${SEED} \
    --save-freq 10

echo "  PHASE 5 COMPLETE: $(date)"

# ==============================================================================
# PHASE 6: DNABERT-2 Baseline (inference only — fast)
# ==============================================================================
DNABERT2_DIR="${EXPERIMENT_DIR}/baselines/dnabert2"
echo ""
echo "============================================================"
echo "  PHASE 6: DNABERT-2 Baseline (embedding extraction)"
echo "  Output: ${DNABERT2_DIR}"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "${DNABERT2_DIR}"

python -u -c "
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '${PROJECT_DIR}')
os.environ['HF_HOME'] = '/shared/achinda1/huggingface_cache'

from src.baselines.dnabert2 import DNABERT2Baseline
from src.data import KmerTokenizer, FastqKmerDataset

print('Loading data...')
tokenizer = KmerTokenizer(k=6)
dataset = FastqKmerDataset('${DATA_PATH}', tokenizer, max_len=150)

# Use 10K samples for DNABERT-2 (it's slow for large datasets)
n = min(len(dataset), 10000)
subset = Subset(dataset, list(range(n)))
loader = DataLoader(subset, batch_size=32, num_workers=4,
    collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])))

print(f'Dataset: {n} samples')

# Extract sequence strings for DNABERT-2
print('Extracting raw sequences...')
sequences = []
for i in range(n):
    seq = str(dataset.records[i].seq)
    sequences.append(seq)

print('Loading DNABERT-2 model...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DNABERT2Baseline(pool_strategy='mean', freeze_backbone=True, device=device)
embeddings = model.encode_sequences(sequences, batch_size=32, max_length=512)

print(f'Embeddings shape: {embeddings.shape}')
np.save('${DNABERT2_DIR}/dnabert2_embeddings.npy', embeddings)
print('DNABERT-2 embeddings saved!')
"

echo "  PHASE 6 COMPLETE: $(date)"

# ==============================================================================
# PHASE 7: K-mer PCA Baseline (CPU only, fast)
# ==============================================================================
KMER_PCA_DIR="${EXPERIMENT_DIR}/baselines/kmer_pca"
echo ""
echo "============================================================"
echo "  PHASE 7: K-mer PCA Baseline"
echo "  Output: ${KMER_PCA_DIR}"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "${KMER_PCA_DIR}"

python -u -c "
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '${PROJECT_DIR}')

from src.baselines.kmer_pca import KmerPCABaseline
from src.data import KmerTokenizer, FastqKmerDataset

print('Loading data...')
tokenizer = KmerTokenizer(k=6)
dataset = FastqKmerDataset('${DATA_PATH}', tokenizer, max_len=150)

n = min(len(dataset), 10000)
subset = Subset(dataset, list(range(n)))
loader = DataLoader(subset, batch_size=64, num_workers=4,
    collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])))

for n_comp in [32, 64, 128]:
    print(f'K-mer PCA with {n_comp} components...')
    baseline = KmerPCABaseline(k=6, n_components=n_comp)
    embeddings = baseline.encode_from_dataloader(loader, tokenizer, fit=True)
    np.save('${KMER_PCA_DIR}/kmer_pca_{}_embeddings.npy'.format(n_comp), embeddings)
    print(f'  Shape: {embeddings.shape}')

print('K-mer PCA baselines complete!')
"

echo "  PHASE 7 COMPLETE: $(date)"

# ==============================================================================
# PHASE 8: Comprehensive Evaluation of All Models
# ==============================================================================
echo ""
echo "============================================================"
echo "  PHASE 8: Comprehensive Evaluation"
echo "  Started: $(date)"
echo "============================================================"

python -u -c "
import os, sys, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
sys.path.insert(0, '${PROJECT_DIR}')

from src.evaluation import evaluate_clustering
from src.data import KmerTokenizer, FastqKmerDataset
from src.models import VQVAE

print('='*80)
print('COMPREHENSIVE EVALUATION')
print('='*80)

# Load data
tokenizer = KmerTokenizer(k=6)
dataset = FastqKmerDataset('${DATA_PATH}', tokenizer, max_len=150)
n = min(len(dataset), 10000)
subset = Subset(dataset, list(range(n)))
loader = DataLoader(subset, batch_size=64, num_workers=4,
    collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}

def extract_vqvae_embeddings(checkpoint_path, loader, device, code_dim=64):
    '''Extract mean-pooled encoder embeddings from a VQ-VAE checkpoint.'''
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get('args', {})
    vocab_size = args_dict.get('vocab_size', len(tokenizer.stoi))
    num_codes = args_dict.get('num_codes', 512)
    cd = args_dict.get('code_dim', code_dim)
    embed_dim = args_dict.get('embed_dim', 128)
    hidden_dim = args_dict.get('hidden_dim', 256)

    model = VQVAE(vocab_size, tokenizer.pad_id,
                  num_codes=num_codes, code_dim=cd,
                  embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    state = checkpoint['model_state_dict']
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for tokens, lengths in loader:
            tokens = tokens.to(device)
            z = model.encoder(tokens)  # (B, L, D)
            emb = z.mean(dim=1)  # (B, D)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)

# --- Evaluate VQ-VAE models ---
model_checkpoints = {
    'vqvae_base': '${VQVAE_DIR}/vqvae_base/best_model.pt',
    'masked_vqvae': '${MQVAE_DIR}/mqvae_masked/best_model.pt',
}

for name, ckpt in model_checkpoints.items():
    if os.path.exists(ckpt):
        print(f'\nEvaluating {name}...')
        cd = 128 if 'masked' in name else 64
        emb = extract_vqvae_embeddings(ckpt, loader, device, code_dim=cd)
        clustering = evaluate_clustering(emb, n_clusters_list=[5, 10, 15, 20])
        results[name] = clustering
        np.save(os.path.join(os.path.dirname(ckpt), f'{name}_embeddings.npy'), emb)
        print(f'  {name}: {len(emb)} embeddings extracted')
    else:
        print(f'  WARNING: {ckpt} not found, skipping {name}')

# --- Evaluate contrastive models ---
# contrastive_finetune.py saves: {output_dir}/{exp_name}/run_{ts}/val_embeddings.npy
# contrastive_finetune_128dim.py saves: {output_dir}/visualizations/final_embeddings.npy
for cdir, cname in [('${CONTRASTIVE_64_DIR}', 'contrastive_64d'),
                     ('${CONTRASTIVE_128_DIR}', 'contrastive_128d')]:
    emb_files = sorted(glob.glob(os.path.join(cdir, '**', '*embeddings.npy'), recursive=True)) if os.path.isdir(cdir) else []
    if emb_files:
        emb = np.load(emb_files[-1])  # use latest
        print(f'\nEvaluating {cname}... ({emb.shape}) from {emb_files[-1]}')
        clustering = evaluate_clustering(emb, n_clusters_list=[5, 10, 15, 20])
        results[cname] = clustering
    else:
        print(f'  WARNING: no embeddings found in {cdir}')

# --- Evaluate baselines ---
baseline_emb_files = {
    'dnabert2': '${DNABERT2_DIR}/dnabert2_embeddings.npy',
    'kmer_pca_64': '${KMER_PCA_DIR}/kmer_pca_64_embeddings.npy',
    'autoencoder': '${AE_DIR}/autoencoder_embeddings.npy',
    'transformer_vae': '${TVAE_DIR}/transformer_vae_embeddings.npy',
}
for name, path in baseline_emb_files.items():
    if os.path.exists(path):
        emb = np.load(path)
        print(f'\nEvaluating {name}... ({emb.shape})')
        clustering = evaluate_clustering(emb, n_clusters_list=[5, 10, 15, 20])
        results[name] = clustering
    else:
        print(f'  WARNING: {path} not found, skipping {name}')

# Save combined results
output_path = '${EXPERIMENT_DIR}/combined_evaluation_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved to {output_path}')

# Print summary table
print('\n' + '='*80)
print('SUMMARY: Silhouette Score (k=10, KMeans)')
print('='*80)
for name, res in results.items():
    k10 = res.get('kmeans_k10', {}).get('silhouette', {})
    mean = k10.get('mean', 'N/A')
    std = k10.get('std', 'N/A')
    if isinstance(mean, float):
        print(f'  {name:25s} → {mean:.4f} ± {std:.4f}')
    else:
        print(f'  {name:25s} → {mean}')
print('='*80)
"

echo "  PHASE 8 COMPLETE: $(date)"

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE!"
echo "  Finished: $(date)"
echo ""
echo "  Results directory: ${EXPERIMENT_DIR}"
echo "  Combined results:  ${EXPERIMENT_DIR}/combined_evaluation_results.json"
echo ""
echo "  Experiments produced:"
ls -la ${EXPERIMENT_DIR}/ 2>/dev/null
echo ""
echo "  To view results:"
echo "    cat ${EXPERIMENT_DIR}/combined_evaluation_results.json | python -m json.tool"
echo "============================================================"
