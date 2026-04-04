#!/usr/bin/env python3
"""
Ablation Study Runner — DDP-Optimised

Trains VQ-VAE variants for each AblationConfig, extracts embeddings,
and evaluates clustering + embedding quality.

Ablation studies (from ablation.py):
  1. Codebook size K  (6 configs)
  2. Code dimension D  (5 configs)
  3. K-mer size k      (5 configs)
  4. Loss components   (5 configs)
  5. Masking prob p     (7 configs)

Each run: 30 epochs, 4-GPU DDP, AMP, TF32, torch.compile.

Usage (single study):
    torchrun --nproc_per_node=4 scripts/run_ablation_study.py \
        --data-path ~/data/contrastive/processed/subset_1M.fastq \
        --output-dir ~/data/contrastive/experiments/ablations \
        --study codebook_size

Usage (all studies):
    torchrun --nproc_per_node=4 scripts/run_ablation_study.py \
        --data-path ~/data/contrastive/processed/subset_1M.fastq \
        --output-dir ~/data/contrastive/experiments/ablations \
        --study all
"""

import os
import sys
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
from src.data import KmerTokenizer, FastqKmerDataset
from src.evaluation.ablation import (
    AblationConfig,
    generate_codebook_size_ablation,
    generate_code_dim_ablation,
    generate_kmer_size_ablation,
    generate_loss_ablation,
    generate_masking_ablation,
)
from src.evaluation import evaluate_clustering, evaluate_embedding_quality


# ═════════════════════════════════════════════════════════════════════════════
# DDP helpers
# ═════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def printf(*a, **kw):
    """Print only on rank 0."""
    if is_main():
        print(*a, **kw, flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


def build_dataloaders(data_path, tokenizer, cfg: AblationConfig, rank, world):
    dataset = FastqKmerDataset(data_path, tokenizer, max_len=cfg.max_seq_length)
    n = len(dataset)
    test_n = int(0.1 * n)
    train_n = n - test_n

    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, test_ds = random_split(dataset, [train_n, test_n], generator=gen)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader, train_sampler


def _clean_state_dict(sd):
    """Strip _orig_mod. prefix (torch.compile) and module. prefix (DDP)."""
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_entropy_bonus(codes, num_codes, weight):
    """Codebook entropy bonus (anti-collapse)."""
    if weight == 0:
        return torch.tensor(0.0, device=codes.device)
    hist = torch.bincount(codes.reshape(-1), minlength=num_codes).float()
    p = hist / (hist.sum() + 1e-9)
    H = -(p * (p + 1e-9).log()).sum()
    return weight * H


def train_one_epoch(model, loader, optimizer, scaler, device, pad_id, cfg: AblationConfig, mask_id=None, amp_dtype=torch.bfloat16):
    model.train()
    total_loss = total_recon = total_vq = 0.0
    n_tok = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        # Optional masking for masking-probability ablation
        if mask_id is not None and cfg.mask_training_prob > 0:
            mask_pos = (
                (torch.rand_like(batch_tokens.float()) < cfg.mask_training_prob)
                & (batch_tokens != pad_id)
            )
            input_tokens = batch_tokens.clone()
            input_tokens[mask_pos] = mask_id
        else:
            input_tokens = batch_tokens

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, loss_vq, codes = model(input_tokens)
            B, L, V = logits.shape

            logits[:, :, pad_id] = -1e9  # safe with bf16/fp32 (bf16 max ~3.4e38)
            valid = (
                (torch.arange(L, device=device)[None, :] < batch_lengths[:, None])
                & (batch_tokens != pad_id)
            )
            if valid.sum() == 0:
                continue

            recon = F.cross_entropy(logits[valid], batch_tokens[valid], reduction="mean")
            if loss_vq.dim() > 0:
                loss_vq = loss_vq.mean()

            ent = compute_entropy_bonus(codes, cfg.num_codes, cfg.entropy_weight)
            loss = recon + loss_vq - ent

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        nv = valid.sum().item()
        total_loss += loss.item() * nv
        total_recon += recon.item() * nv
        total_vq += loss_vq.item() * nv
        n_tok += nv

    return {
        "loss": total_loss / max(n_tok, 1),
        "recon": total_recon / max(n_tok, 1),
        "vq": total_vq / max(n_tok, 1),
    }


@torch.no_grad()
def validate(model, loader, device, pad_id, cfg: AblationConfig, mask_id=None, amp_dtype=torch.bfloat16):
    model.eval()
    total_loss = total_recon = total_vq = 0.0
    n_tok = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        # For masking ablation, apply same masking at eval
        if mask_id is not None and cfg.mask_training_prob > 0:
            mask_pos = (
                (torch.rand_like(batch_tokens.float()) < cfg.mask_training_prob)
                & (batch_tokens != pad_id)
            )
            input_tokens = batch_tokens.clone()
            input_tokens[mask_pos] = mask_id
        else:
            input_tokens = batch_tokens

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, loss_vq, codes = model(input_tokens)
            B, L, V = logits.shape
            logits[:, :, pad_id] = -1e9  # safe with bf16/fp32
            valid = (
                (torch.arange(L, device=device)[None, :] < batch_lengths[:, None])
                & (batch_tokens != pad_id)
            )
            if valid.sum() == 0:
                continue
            recon = F.cross_entropy(logits[valid], batch_tokens[valid], reduction="mean")
            if loss_vq.dim() > 0:
                loss_vq = loss_vq.mean()
            ent = compute_entropy_bonus(codes, cfg.num_codes, cfg.entropy_weight)
            loss = recon + loss_vq - ent

        nv = valid.sum().item()
        total_loss += loss.item() * nv
        total_recon += recon.item() * nv
        total_vq += loss_vq.item() * nv
        n_tok += nv

    return {
        "loss": total_loss / max(n_tok, 1),
        "recon": total_recon / max(n_tok, 1),
        "vq": total_vq / max(n_tok, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=10_000):
    """Extract mean-pooled encoder embeddings (no VQ) for evaluation."""
    model.eval()
    # unwrap DDP
    raw = model.module if hasattr(model, "module") else model
    all_emb = []
    n = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        z_e = raw.encoder(batch_tokens)  # (B, L, D)
        # mean-pool over valid positions
        B, L, D = z_e.shape
        mask = (torch.arange(L, device=device)[None, :] < batch_lengths[:, None]).unsqueeze(-1)  # (B,L,1)
        z_mean = (z_e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        all_emb.append(z_mean.cpu().numpy())
        n += B
        if n >= max_samples:
            break

    emb = np.concatenate(all_emb, axis=0)[:max_samples]
    return emb


# ═════════════════════════════════════════════════════════════════════════════
# Single ablation run
# ═════════════════════════════════════════════════════════════════════════════

def run_single_ablation(
    cfg: AblationConfig,
    data_path: str,
    output_root: str,
    rank: int,
    world: int,
    local_rank: int,
):
    device = torch.device(f"cuda:{local_rank}")
    run_dir = os.path.join(output_root, cfg.name)
    if is_main():
        os.makedirs(run_dir, exist_ok=True)

    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    printf(f"\n{'='*60}")
    printf(f"ABLATION: {cfg.name}")
    printf(f"  {cfg.description}")
    printf(f"  num_codes={cfg.num_codes}, code_dim={cfg.code_dim}, k_mer={cfg.k_mer}")
    printf(f"  commitment_cost={cfg.commitment_cost}, entropy_weight={cfg.entropy_weight}")
    printf(f"  mask_training_prob={cfg.mask_training_prob}, epochs={cfg.epochs}")
    printf(f"{'='*60}")

    # Tokenizer
    tokenizer = KmerTokenizer(k=cfg.k_mer, use_canonical=True)
    VOCAB_SIZE = len(tokenizer.stoi)
    PAD_ID = tokenizer.pad_id
    MASK_ID = tokenizer.mask_id

    # Auto-scale batch size for large vocabularies to avoid OOM
    # vocab 4099 (k=6) works at bs=512; vocab 65539 (k=8) needs ~bs=64
    effective_bs = cfg.batch_size
    if VOCAB_SIZE > 20_000:
        effective_bs = max(32, cfg.batch_size // 8)
        printf(f"  ⚠ Large vocab ({VOCAB_SIZE}) — reducing batch size {cfg.batch_size}→{effective_bs}")
    elif VOCAB_SIZE > 8_000:
        effective_bs = max(64, cfg.batch_size // 4)
        printf(f"  ⚠ Medium vocab ({VOCAB_SIZE}) — reducing batch size {cfg.batch_size}→{effective_bs}")
    cfg_for_data = AblationConfig(**cfg.to_dict())
    cfg_for_data.batch_size = effective_bs

    # Data
    train_loader, test_loader, train_sampler = build_dataloaders(
        data_path, tokenizer, cfg_for_data, rank, world,
    )

    # Model
    model = VQVAE(
        VOCAB_SIZE,
        PAD_ID,
        num_codes=cfg.num_codes,
        code_dim=cfg.code_dim,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        commitment_cost=cfg.commitment_cost,
    ).to(device)

    # TF32 (skip torch.compile — causes LLVM segfault after many sequential runs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)

    # Use bf16 if available (same range as fp32, no overflow), else disable AMP
    use_amp = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # not needed for bf16

    # Whether this ablation uses masking
    uses_masking = cfg.mask_training_prob > 0
    mask_id_arg = MASK_ID if uses_masking else None

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, PAD_ID, cfg, mask_id_arg, amp_dtype=amp_dtype)
        val_stats = validate(model, test_loader, device, PAD_ID, cfg, mask_id_arg, amp_dtype=amp_dtype)

        printf(
            f"  [{cfg.name}] Epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"recon={val_stats['recon']:.4f}"
        )

        # Save best
        if val_stats["loss"] < best_val_loss and is_main():
            best_val_loss = val_stats["loss"]
            raw = model.module if hasattr(model, "module") else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _clean_state_dict(raw.state_dict()),
                    "val_stats": val_stats,
                    "config": cfg.to_dict(),
                },
                os.path.join(run_dir, "best_model.pt"),
            )

    train_time = time.time() - t_start
    printf(f"  Training done in {train_time:.0f}s  best_val_loss={best_val_loss:.4f}")

    # ── Extract embeddings & evaluate (rank 0 only) ─────────────────────
    result = None
    if is_main():
        # Reload best model
        ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device, weights_only=False)
        raw_model = VQVAE(
            VOCAB_SIZE, PAD_ID,
            num_codes=cfg.num_codes, code_dim=cfg.code_dim,
            embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
            commitment_cost=cfg.commitment_cost,
        ).to(device)
        raw_model.load_state_dict(_clean_state_dict(ckpt["model_state_dict"]))

        emb = extract_embeddings(raw_model, test_loader, device, max_samples=10_000)
        np.save(os.path.join(run_dir, "embeddings.npy"), emb)
        printf(f"  Embeddings: {emb.shape}")

        # Clustering evaluation (KMeans only, k=10)
        clust = evaluate_clustering(
            emb,
            n_clusters_list=[5, 10, 15, 20],
            methods=["kmeans"],
            random_seeds=[42, 123, 456],
        )
        eq = evaluate_embedding_quality(emb)

        result = {
            "config": cfg.to_dict(),
            "train_time_s": train_time,
            "best_val_loss": best_val_loss,
            "clustering": clust,
            "embedding_quality": eq,
        }
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        printf(f"  Silhouette(k=10): {clust.get('kmeans_k10', {}).get('silhouette', {}).get('mean', 'N/A')}")

        del raw_model
        torch.cuda.empty_cache()

    # Full cleanup between ablation runs to prevent OOM accumulation
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()

    # Sync before next run
    if dist.is_initialized():
        dist.barrier()

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Study registry
# ═════════════════════════════════════════════════════════════════════════════

STUDY_REGISTRY = {
    "codebook_size": generate_codebook_size_ablation,
    "code_dim": generate_code_dim_ablation,
    "kmer_size": generate_kmer_size_ablation,
    "loss_components": generate_loss_ablation,
    "masking": generate_masking_ablation,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study runner (DDP)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./ablation_results")
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=list(STUDY_REGISTRY.keys()) + ["all"],
        help="Which ablation study to run",
    )
    parser.add_argument(
        "--override-epochs",
        type=int,
        default=None,
        help="Override the epoch count in all configs (for quick testing)",
    )
    parser.add_argument(
        "--override-batch-size",
        type=int,
        default=None,
        help="Override batch size (per GPU)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already have result.json (resume interrupted run)",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def print_ablation_summary(study_name: str, all_results: list):
    """Print a nicely formatted summary for one ablation study."""
    printf(f"\n{'='*70}")
    printf(f"ABLATION SUMMARY: {study_name}")
    printf(f"{'='*70}")

    header = (
        f"{'Config':<30} {'ValLoss':>8} {'Sil@10':>8} {'DB@10':>8} "
        f"{'Uniformity':>11} {'Isotropy':>9} {'EffDim':>7} {'Time':>6}"
    )
    printf(header)
    printf("-" * len(header))

    for r in sorted(all_results, key=lambda x: x.get("best_val_loss", 999)):
        name = r["config"]["name"]
        vl = r["best_val_loss"]
        cl = r.get("clustering", {}).get("kmeans_k10", {})
        sil = cl.get("silhouette", {}).get("mean", float("nan"))
        db = cl.get("davies_bouldin", {}).get("mean", float("nan"))
        eq = r.get("embedding_quality", {})
        uni = eq.get("uniformity", float("nan"))
        iso = eq.get("isotropy", float("nan"))
        ed = eq.get("effective_dimensionality", float("nan"))
        t = r.get("train_time_s", 0)
        printf(
            f"{name:<30} {vl:>8.4f} {sil:>8.4f} {db:>8.4f} "
            f"{uni:>11.4f} {iso:>9.4f} {ed:>7.1f} {t:>5.0f}s"
        )
    printf("=" * len(header))


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    rank, world, local_rank = setup_ddp()

    printf(f"DDP initialized: rank={rank}, world={world}, local_rank={local_rank}")
    printf(f"Data: {args.data_path}")
    printf(f"Output: {args.output_dir}")

    # Determine studies
    if args.study == "all":
        studies_to_run = list(STUDY_REGISTRY.keys())
    else:
        studies_to_run = [args.study]

    grand_results = {}

    for study_name in studies_to_run:
        printf(f"\n{'#'*70}")
        printf(f"# STUDY: {study_name}")
        printf(f"{'#'*70}")

        configs = STUDY_REGISTRY[study_name]()

        # Apply overrides
        for cfg in configs:
            if args.override_epochs is not None:
                cfg.epochs = args.override_epochs
            if args.override_batch_size is not None:
                cfg.batch_size = args.override_batch_size

        study_results = []
        study_dir = os.path.join(args.output_dir, study_name)

        for i, cfg in enumerate(configs):
            printf(f"\n--- [{study_name}] Config {i+1}/{len(configs)}: {cfg.name} ---")

            # Resume: skip configs whose result.json already exists
            result_path = os.path.join(study_dir, cfg.name, "result.json")
            if args.resume and os.path.exists(result_path):
                printf(f"  ⏭ Skipping {cfg.name} (result.json exists)")
                if is_main():
                    with open(result_path) as f:
                        result = json.load(f)
                    study_results.append(result)
                continue

            result = run_single_ablation(cfg, args.data_path, study_dir, rank, world, local_rank)
            if result is not None:
                study_results.append(result)

        if is_main() and study_results:
            print_ablation_summary(study_name, study_results)
            # Save study summary
            summary_path = os.path.join(study_dir, f"{study_name}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(study_results, f, indent=2)
            printf(f"Study summary saved to {summary_path}")

        grand_results[study_name] = study_results

    # Grand summary
    if is_main():
        grand_path = os.path.join(args.output_dir, "all_ablation_results.json")
        with open(grand_path, "w") as f:
            json.dump(grand_results, f, indent=2)
        printf(f"\n✓ All ablation results saved to {grand_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
