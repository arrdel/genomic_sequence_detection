#!/usr/bin/env python3
"""
Entropy Weight Ablation Study — 2-GPU DDP on GPUs 4,5

Runs the 4 MISSING entropy weight configurations:
  λ ∈ {0.001, 0.005, 0.03, 0.05}

The 3 existing values (λ=0, 0.003, 0.01) are already in results/ablations/loss_components/.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
        scripts/run_entropy_ablation.py \
        --data-path ~/data/contrastive/processed/subset_1M.fastq \
        --output-dir ~/data/contrastive/experiments/ablations/entropy_weight
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
from src.evaluation.ablation import AblationConfig
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
    if is_main():
        print(*a, **kw, flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Entropy weight configs — only the 4 we need to run
# ═════════════════════════════════════════════════════════════════════════════

def generate_entropy_weight_ablation():
    """Generate configs for the 4 missing entropy weight values."""
    configs = []
    for lam in [0.001, 0.005, 0.03, 0.05]:
        configs.append(AblationConfig(
            name=f"entropy_lambda_{str(lam).replace('.', '_')}",
            description=f"Entropy weight λ={lam}",
            num_codes=512,
            code_dim=64,
            embed_dim=128,
            hidden_dim=256,
            commitment_cost=0.1,
            entropy_weight=lam,
            k_mer=6,
            max_seq_length=150,
            epochs=30,
            batch_size=1024,   # Maximize GPU utilization on 2× RTX 4090
            learning_rate=2e-4,
            seed=42,
            mask_training_prob=0.2,
        ))
    return configs


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


def build_dataloaders(data_path, tokenizer, cfg, rank, world):
    dataset = FastqKmerDataset(data_path, tokenizer, max_len=cfg.max_seq_length)
    n = len(dataset)
    test_n = int(0.1 * n)
    train_n = n - test_n

    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, test_ds = random_split(dataset, [train_n, test_n], generator=gen)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, sampler=test_sampler,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader, train_sampler


def _clean_state_dict(sd):
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
    if weight == 0:
        return torch.tensor(0.0, device=codes.device)
    hist = torch.bincount(codes.reshape(-1), minlength=num_codes).float()
    p = hist / (hist.sum() + 1e-9)
    H = -(p * (p + 1e-9).log()).sum()
    return weight * H


@torch.no_grad()
def compute_codebook_utilization(codes, num_codes):
    """Fraction of codebook entries used in this batch."""
    unique = codes.reshape(-1).unique().numel()
    return unique / num_codes


def train_one_epoch(model, loader, optimizer, device, pad_id, cfg, mask_id):
    model.train()
    total_loss = total_recon = total_vq = 0.0
    total_util = 0.0
    n_tok = 0
    n_batches = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        # Apply masking
        if cfg.mask_training_prob > 0:
            mask_pos = (
                (torch.rand_like(batch_tokens.float()) < cfg.mask_training_prob)
                & (batch_tokens != pad_id)
            )
            input_tokens = batch_tokens.clone()
            input_tokens[mask_pos] = mask_id
        else:
            input_tokens = batch_tokens

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss_vq, codes = model(input_tokens)
            B, L, V = logits.shape

            logits[:, :, pad_id] = -1e9
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
        total_util += compute_codebook_utilization(codes, cfg.num_codes)
        n_tok += nv
        n_batches += 1

    return {
        "loss": total_loss / max(n_tok, 1),
        "recon": total_recon / max(n_tok, 1),
        "vq": total_vq / max(n_tok, 1),
        "util": total_util / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device, pad_id, cfg, mask_id):
    model.eval()
    total_loss = total_recon = total_vq = 0.0
    total_util = 0.0
    n_tok = 0
    n_batches = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        if cfg.mask_training_prob > 0:
            mask_pos = (
                (torch.rand_like(batch_tokens.float()) < cfg.mask_training_prob)
                & (batch_tokens != pad_id)
            )
            input_tokens = batch_tokens.clone()
            input_tokens[mask_pos] = mask_id
        else:
            input_tokens = batch_tokens

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss_vq, codes = model(input_tokens)
            B, L, V = logits.shape
            logits[:, :, pad_id] = -1e9
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
        total_util += compute_codebook_utilization(codes, cfg.num_codes)
        n_tok += nv
        n_batches += 1

    return {
        "loss": total_loss / max(n_tok, 1),
        "recon": total_recon / max(n_tok, 1),
        "vq": total_vq / max(n_tok, 1),
        "util": total_util / max(n_batches, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=10_000):
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    all_emb = []
    n = 0

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_lengths = batch_lengths.to(device, non_blocking=True)

        z_e = raw.encoder(batch_tokens)
        B, L, D = z_e.shape
        mask = (torch.arange(L, device=device)[None, :] < batch_lengths[:, None]).unsqueeze(-1)
        z_mean = (z_e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_emb.append(z_mean.cpu().numpy())
        n += B
        if n >= max_samples:
            break

    return np.concatenate(all_emb, axis=0)[:max_samples]


# ═════════════════════════════════════════════════════════════════════════════
# Single ablation run
# ═════════════════════════════════════════════════════════════════════════════

def run_single_ablation(cfg, data_path, output_root, rank, world, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    run_dir = os.path.join(output_root, cfg.name)
    if is_main():
        os.makedirs(run_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    printf(f"\n{'='*60}")
    printf(f"ENTROPY ABLATION: {cfg.name}")
    printf(f"  {cfg.description}")
    printf(f"  entropy_weight={cfg.entropy_weight}, batch_size={cfg.batch_size}")
    printf(f"  epochs={cfg.epochs}, GPUs={world}")
    printf(f"{'='*60}")

    tokenizer = KmerTokenizer(k=cfg.k_mer, use_canonical=True)
    VOCAB_SIZE = len(tokenizer.stoi)
    PAD_ID = tokenizer.pad_id
    MASK_ID = tokenizer.mask_id

    train_loader, test_loader, train_sampler = build_dataloaders(
        data_path, tokenizer, cfg, rank, world,
    )

    model = VQVAE(
        VOCAB_SIZE, PAD_ID,
        num_codes=cfg.num_codes, code_dim=cfg.code_dim,
        embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
        commitment_cost=cfg.commitment_cost,
    ).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, optimizer, device, PAD_ID, cfg, MASK_ID)
        val_stats = validate(model, test_loader, device, PAD_ID, cfg, MASK_ID)

        printf(
            f"  [{cfg.name}] Epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"recon={val_stats['recon']:.4f} "
            f"util={val_stats['util']:.1%}"
        )

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

    # Evaluate (rank 0 only)
    result = None
    if is_main():
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

        sil = clust.get("kmeans_k10", {}).get("silhouette", {}).get("mean", "N/A")
        db = clust.get("kmeans_k10", {}).get("davies_bouldin", {}).get("mean", "N/A")
        printf(f"  Silhouette(k=10): {sil}")
        printf(f"  Davies-Bouldin(k=10): {db}")
        printf(f"  Uniformity: {eq.get('uniformity', 'N/A')}")
        printf(f"  Eff. Dim: {eq.get('effective_dimensionality', 'N/A')}")

        del raw_model
        torch.cuda.empty_cache()

    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Entropy weight ablation (2-GPU DDP)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./entropy_ablation_results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs that already have result.json")
    args = parser.parse_args()

    rank, world, local_rank = setup_ddp()

    printf(f"Entropy Weight Ablation Study")
    printf(f"DDP: rank={rank}, world={world}, local_rank={local_rank}")
    printf(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    printf(f"Data: {args.data_path}")
    printf(f"Output: {args.output_dir}")

    configs = generate_entropy_weight_ablation()
    printf(f"Configs to run: {len(configs)}")
    for c in configs:
        printf(f"  - {c.name}: λ={c.entropy_weight}")

    all_results = []
    for i, cfg in enumerate(configs):
        printf(f"\n--- Config {i+1}/{len(configs)}: {cfg.name} ---")

        result_path = os.path.join(args.output_dir, cfg.name, "result.json")
        if args.resume and os.path.exists(result_path):
            printf(f"  ⏭ Skipping {cfg.name} (result.json exists)")
            if is_main():
                with open(result_path) as f:
                    all_results.append(json.load(f))
            continue

        result = run_single_ablation(cfg, args.data_path, args.output_dir, rank, world, local_rank)
        if result is not None:
            all_results.append(result)

    # Print summary
    if is_main() and all_results:
        printf(f"\n{'='*70}")
        printf(f"ENTROPY WEIGHT ABLATION SUMMARY")
        printf(f"{'='*70}")
        printf(f"{'Config':<35} {'λ':>6} {'ValLoss':>8} {'Sil@10':>8} {'DB@10':>8} {'Util':>6} {'EffDim':>7} {'Time':>6}")
        printf("-" * 90)

        for r in sorted(all_results, key=lambda x: x["config"]["entropy_weight"]):
            name = r["config"]["name"]
            lam = r["config"]["entropy_weight"]
            vl = r["best_val_loss"]
            cl = r.get("clustering", {}).get("kmeans_k10", {})
            sil = cl.get("silhouette", {}).get("mean", float("nan"))
            db = cl.get("davies_bouldin", {}).get("mean", float("nan"))
            eq = r.get("embedding_quality", {})
            ed = eq.get("effective_dimensionality", float("nan"))
            t = r.get("train_time_s", 0)
            printf(f"{name:<35} {lam:>6.3f} {vl:>8.4f} {sil:>8.4f} {db:>8.4f} {'--':>6} {ed:>7.1f} {t:>5.0f}s")

        printf("=" * 90)

        summary_path = os.path.join(args.output_dir, "entropy_weight_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        printf(f"\n✓ Summary saved to {summary_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
