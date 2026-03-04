#!/usr/bin/env python3
"""
Train Transformer-based VAE Baseline

Usage:
    python scripts/train_transformer_vae.py \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
        --output-dir /media/scratch/adele/contrastive/experiments/transformer_vae \
        --epochs 50 --batch-size 32 --n-gpu 2
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
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset
from src.baselines.transformer_vae import TransformerVAE


def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, pad_id, beta=1.0, grad_clip=1.0):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    n_batches = 0
    
    for batch_tokens, batch_lengths in tqdm(loader, desc="Training"):
        batch_tokens = batch_tokens.to(device)
        
        logits, mu, logvar = model(batch_tokens)
        loss, recon_loss, kl_loss = model.loss(batch_tokens, logits, mu, logvar, beta=beta)
        
        optimizer.zero_grad()
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
def evaluate(model, loader, device, pad_id, beta=1.0):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    total_correct, total_tokens = 0, 0
    n_batches = 0
    
    for batch_tokens, batch_lengths in tqdm(loader, desc="Evaluating"):
        batch_tokens = batch_tokens.to(device)
        
        logits, mu, logvar = model(batch_tokens)
        loss, recon_loss, kl_loss = model.loss(batch_tokens, logits, mu, logvar, beta=beta)
        
        # Token accuracy
        preds = logits.argmax(dim=-1)
        mask = (batch_tokens != pad_id)
        correct = ((preds == batch_tokens) & mask).sum().item()
        tokens = mask.sum().item()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_correct += correct
        total_tokens += tokens
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
        "token_accuracy": total_correct / (total_tokens + 1e-9),
    }


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    
    for batch_tokens, batch_lengths in tqdm(loader, desc="Extracting embeddings"):
        batch_tokens = batch_tokens.to(device)
        emb = model.get_embeddings(batch_tokens, pool="mean")
        all_embeddings.append(emb.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./experiments/transformer_vae")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight (beta-VAE)")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-gpu", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=10)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Device
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    elif args.n_gpu > 0:
        gpu_ids = list(range(args.n_gpu))
    else:
        gpu_ids = []
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available() else "cpu")
    
    # Data
    tokenizer = KmerTokenizer(k=6)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)
    
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    
    # Model
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
    
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {train_size}, Test: {test_size}")
    
    # Training loop
    best_loss = float("inf")
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    tokenizer.pad_id, beta=args.beta)
        test_metrics = evaluate(model, test_loader, device, tokenizer.pad_id, beta=args.beta)
        
        print(f"  Train: loss={train_metrics['loss']:.4f} "
              f"recon={train_metrics['recon_loss']:.4f} kl={train_metrics['kl_loss']:.4f}")
        print(f"  Test:  loss={test_metrics['loss']:.4f} "
              f"recon={test_metrics['recon_loss']:.4f} kl={test_metrics['kl_loss']:.4f} "
              f"acc={test_metrics['token_accuracy']:.4f}")
        
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})
        
        # Save best
        core_model = model.module if hasattr(model, "module") else model
        if test_metrics["loss"] < best_loss:
            best_loss = test_metrics["loss"]
            torch.save({
                "model_state_dict": core_model.state_dict(),
                "epoch": epoch,
                "test_metrics": test_metrics,
                "args": vars(args),
            }, os.path.join(args.output_dir, "checkpoints", "best_model.pt"))
            print("  ✓ Best model saved")
        
        if epoch % args.save_freq == 0:
            torch.save({
                "model_state_dict": core_model.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"))
    
    # Extract and save test embeddings
    print("\nExtracting test embeddings...")
    core_model = model.module if hasattr(model, "module") else model
    embeddings = extract_embeddings(core_model, test_loader, device)
    np.save(os.path.join(args.output_dir, "transformer_vae_embeddings.npy"), embeddings)
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete. Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
