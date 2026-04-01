#!/usr/bin/env python3
"""
Train Transformer-based VAE Baseline — DDP version

Uses DistributedDataParallel for efficient multi-GPU training with:
  - Per-GPU process (no GIL bottleneck)
  - NCCL all-reduce (even GPU load)
  - Mixed precision (AMP) + TF32
  - torch.compile() kernel fusion
  - Optimized data loading

Launch (multi-GPU):
    torchrun --nproc_per_node=4 scripts/train_transformer_vae.py \
        --data-path data.fastq --output-dir experiments/tvae \
        --epochs 50 --batch-size 512

Single-GPU fallback (no torchrun):
    python scripts/train_transformer_vae.py \
        --data-path data.fastq --output-dir experiments/tvae \
        --epochs 50 --batch-size 512
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset
from src.baselines.transformer_vae import TransformerVAE


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main():
    return get_rank() == 0

def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, True
    return 0, False

def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()

def print_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths

def set_seed(seed, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Unwrap helper
# ---------------------------------------------------------------------------

def _unwrap(model):
    """Unwrap DDP / DataParallel / torch.compile to access custom methods."""
    m = model
    if isinstance(m, (DDP, nn.DataParallel)):
        m = m.module
    # torch.compile wraps in OptimizedModule
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, pad_id, beta=1.0,
                grad_clip=1.0, scaler=None):
    model.train()
    base = _unwrap(model)
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0
    use_amp = scaler is not None

    iterator = tqdm(loader, desc="Training", disable=not is_main())
    for batch_tokens, _ in iterator:
        batch_tokens = batch_tokens.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, mu, logvar = model(batch_tokens)
            loss, recon_loss, kl_loss = base.loss(
                batch_tokens, logits, mu, logvar, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, pad_id, beta=1.0, use_amp=False):
    model.eval()
    base = _unwrap(model)
    total_loss = total_recon = total_kl = 0.0
    total_correct = total_tokens = 0
    n_batches = 0

    iterator = tqdm(loader, desc="Evaluating", disable=not is_main())
    for batch_tokens, _ in iterator:
        batch_tokens = batch_tokens.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, mu, logvar = model(batch_tokens)
            loss, recon_loss, kl_loss = base.loss(
                batch_tokens, logits, mu, logvar, beta=beta)

        preds = logits.argmax(dim=-1)
        mask = batch_tokens != pad_id
        total_correct += ((preds == batch_tokens) & mask).sum().item()
        total_tokens += mask.sum().item()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
        "token_accuracy": total_correct / (total_tokens + 1e-9),
    }


@torch.no_grad()
def extract_embeddings(model, loader, device, use_amp=False):
    model.eval()
    base = _unwrap(model)
    all_emb = []
    for batch_tokens, _ in tqdm(loader, desc="Extracting embeddings",
                                 disable=not is_main()):
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            emb = base.get_embeddings(batch_tokens, pool="mean")
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str,
                        default="./experiments/transformer_vae")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Per-GPU batch size")
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL weight (beta-VAE)")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=10)
    # Legacy args kept for script compatibility
    parser.add_argument("--n-gpu", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    args = parser.parse_args()

    # ---- Distributed setup ------------------------------------------------
    local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}"
                          if torch.cuda.is_available() else "cpu")
    rank = get_rank()
    world_size = get_world_size()
    set_seed(args.seed, rank)

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    if distributed:
        dist.barrier()

    # ---- Performance flags ------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data -------------------------------------------------------------
    tokenizer = KmerTokenizer(k=6)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)

    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_sampler = (DistributedSampler(train_set, num_replicas=world_size,
                                        rank=rank, shuffle=True,
                                        seed=args.seed)
                     if distributed else None)
    test_sampler = (DistributedSampler(test_set, num_replicas=world_size,
                                       rank=rank, shuffle=False)
                    if distributed else None)

    nw = args.num_workers
    loader_kwargs = dict(
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
        num_workers=nw,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=True, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, sampler=test_sampler, **loader_kwargs,
    )

    # ---- Model ------------------------------------------------------------
    model = TransformerVAE(
        vocab_size=len(tokenizer.stoi),
        pad_id=tokenizer.pad_id,
        embed_dim=128,
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=256,
    ).to(device)

    # torch.compile for kernel fusion
    try:
        model = torch.compile(model)
        print_main("✓ torch.compile() enabled")
    except Exception as e:
        print_main(f"⚠ torch.compile() skipped: {e}")

    if distributed:
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate, weight_decay=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    eff_bs = args.batch_size * world_size
    print_main(f"Tokenizer built: vocab size = {len(tokenizer.stoi)}")
    print_main(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print_main(f"Train: {train_size}, Test: {test_size}")
    print_main(f"Per-GPU batch: {args.batch_size} × {world_size} GPUs = "
               f"{eff_bs} effective")
    print_main(f"AMP: {use_amp}  |  TF32: True  |  "
               f"DDP: {distributed}  |  compile: True")

    # ---- Training ---------------------------------------------------------
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        print_main(f"\nEpoch {epoch}/{args.epochs}")

        train_m = train_epoch(model, train_loader, optimizer, device,
                              tokenizer.pad_id, beta=args.beta, scaler=scaler)
        test_m = evaluate(model, test_loader, device, tokenizer.pad_id,
                          beta=args.beta, use_amp=use_amp)

        print_main(f"  Train: loss={train_m['loss']:.4f} "
                   f"recon={train_m['recon_loss']:.4f} "
                   f"kl={train_m['kl_loss']:.4f}")
        print_main(f"  Test:  loss={test_m['loss']:.4f} "
                   f"recon={test_m['recon_loss']:.4f} "
                   f"kl={test_m['kl_loss']:.4f} "
                   f"acc={test_m['token_accuracy']:.4f}")

        history.append({"epoch": epoch, "train": train_m, "test": test_m})

        if is_main():
            core = _unwrap(model)
            if test_m["loss"] < best_loss:
                best_loss = test_m["loss"]
                torch.save({
                    "model_state_dict": core.state_dict(),
                    "epoch": epoch,
                    "test_metrics": test_m,
                    "args": vars(args),
                }, os.path.join(args.output_dir, "checkpoints",
                                "best_model.pt"))
                print("  ✓ Best model saved", flush=True)

            if epoch % args.save_freq == 0:
                torch.save({
                    "model_state_dict": core.state_dict(),
                    "epoch": epoch,
                }, os.path.join(args.output_dir, "checkpoints",
                                f"checkpoint_epoch_{epoch}.pt"))

    # ---- Post-training (rank 0) -------------------------------------------
    if is_main():
        print("\nExtracting test embeddings...")
        final_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=nw, collate_fn=collate_fn, pin_memory=True,
        )
        embeddings = extract_embeddings(model, final_loader, device,
                                        use_amp=use_amp)
        np.save(os.path.join(args.output_dir,
                             "transformer_vae_embeddings.npy"), embeddings)
        with open(os.path.join(args.output_dir,
                               "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n✓ Training complete. Best test loss: {best_loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
