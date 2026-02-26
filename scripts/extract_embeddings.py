#!/usr/bin/env python3
"""
Extract embeddings from any trained model and save as .npy files.

Supports: VQ-VAE, Masked VQ-VAE, Contrastive VQ-VAE, Autoencoder, Transformer VAE

Usage:
    python scripts/extract_embeddings.py \
        --model-type vqvae \
        --checkpoint /path/to/best_model.pt \
        --data-path /path/to/cleaned_reads.fastq \
        --output-path /path/to/vqvae_embeddings.npy \
        --num-samples 10000

    python scripts/extract_embeddings.py \
        --model-type contrastive \
        --checkpoint /path/to/best_model.pt \
        --vqvae-checkpoint /path/to/vqvae_best_model.pt \
        --data-path /path/to/cleaned_reads.fastq \
        --output-path /path/to/contrastive_embeddings.npy
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset
from src.models import VQVAE


def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


class ContrastiveHead(nn.Module):
    """Projection head matching the training architecture."""
    def __init__(self, encoder, embed_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, tokens):
        z_e = self.encoder(tokens)
        z_mean = z_e.mean(dim=1)
        z_proj = self.proj(z_mean)
        z_norm = F.normalize(z_proj, dim=-1)
        return z_norm


@torch.no_grad()
def extract_vqvae_embeddings(model, loader, device, pool="mean"):
    """Extract encoder embeddings from VQ-VAE (pre-quantization)."""
    model.eval()
    all_emb = []
    for batch_tokens, batch_lengths in tqdm(loader, desc="Extracting VQ-VAE embeddings"):
        batch_tokens = batch_tokens.to(device)
        z_e = model.encoder(batch_tokens)  # [B, L, D]
        if pool == "mean":
            emb = z_e.mean(dim=1)  # [B, D]
        elif pool == "max":
            emb = z_e.max(dim=1).values
        else:
            emb = z_e[:, 0, :]  # CLS-like: first token
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_contrastive_embeddings(model, loader, device):
    """Extract projected+normalized embeddings from contrastive model."""
    model.eval()
    all_emb = []
    for batch_tokens, _ in tqdm(loader, desc="Extracting contrastive embeddings"):
        batch_tokens = batch_tokens.to(device)
        z = model(batch_tokens)
        all_emb.append(z.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_autoencoder_embeddings(model, loader, device):
    """Extract latent embeddings from standard autoencoder."""
    model.eval()
    all_emb = []
    for batch_tokens, _ in tqdm(loader, desc="Extracting AE embeddings"):
        batch_tokens = batch_tokens.to(device)
        emb = model.get_embeddings(batch_tokens, pool="mean")
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_transformer_vae_embeddings(model, loader, device):
    """Extract latent embeddings from Transformer VAE."""
    model.eval()
    all_emb = []
    for batch_tokens, _ in tqdm(loader, desc="Extracting TransformerVAE embeddings"):
        batch_tokens = batch_tokens.to(device)
        emb = model.get_embeddings(batch_tokens, pool="mean")
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def load_vqvae(checkpoint_path, device, tokenizer):
    """Load a VQ-VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model args from checkpoint
    model_args = checkpoint.get("args", {})
    vocab_size = model_args.get("vocab_size", len(tokenizer.stoi))
    num_codes = model_args.get("num_codes", 512)
    code_dim = model_args.get("code_dim", 64)
    embed_dim = model_args.get("embed_dim", 128)
    hidden_dim = model_args.get("hidden_dim", 256)
    
    model = VQVAE(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_codes=num_codes,
        code_dim=code_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    
    state_dict = checkpoint["model_state_dict"]
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"  Loaded VQ-VAE: vocab={vocab_size}, K={num_codes}, D={code_dim}")
    return model, code_dim


def load_contrastive(checkpoint_path, vqvae_checkpoint_path, device, tokenizer, proj_dim=64):
    """Load a contrastive model from checkpoint."""
    # First load the base VQ-VAE
    base_model, code_dim = load_vqvae(vqvae_checkpoint_path, device, tokenizer)
    
    # Build contrastive head
    contrastive_model = ContrastiveHead(
        encoder=base_model.encoder,
        embed_dim=code_dim,
        proj_dim=proj_dim,
    ).to(device)
    
    # Load contrastive weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    contrastive_model.load_state_dict(state_dict)
    print(f"  Loaded Contrastive head: proj_dim={proj_dim}")
    return contrastive_model


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from trained models")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["vqvae", "masked_vqvae", "contrastive", "autoencoder", "transformer_vae"],
                        help="Type of model to extract from")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vqvae-checkpoint", type=str, default=None,
                        help="Path to base VQ-VAE checkpoint (for contrastive models)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to FASTQ data file")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save embeddings (.npy)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Max number of samples to extract")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=64,
                        help="Projection dimension (for contrastive models)")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    tokenizer = KmerTokenizer(k=6)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)
    n = min(len(dataset), args.num_samples)
    subset = Subset(dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} sequences, using {n}")

    # Extract based on model type
    if args.model_type in ["vqvae", "masked_vqvae"]:
        model, _ = load_vqvae(args.checkpoint, device, tokenizer)
        embeddings = extract_vqvae_embeddings(model, loader, device)

    elif args.model_type == "contrastive":
        if not args.vqvae_checkpoint:
            print("ERROR: --vqvae-checkpoint required for contrastive models")
            sys.exit(1)
        model = load_contrastive(
            args.checkpoint, args.vqvae_checkpoint, device, tokenizer,
            proj_dim=args.proj_dim,
        )
        embeddings = extract_contrastive_embeddings(model, loader, device)

    elif args.model_type == "autoencoder":
        from src.baselines.autoencoder import StandardAutoencoder
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_args = checkpoint.get("args", {})
        model = StandardAutoencoder(
            vocab_size=len(tokenizer.stoi),
            pad_id=tokenizer.pad_id,
            latent_dim=model_args.get("latent_dim", 64),
        ).to(device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        embeddings = extract_autoencoder_embeddings(model, loader, device)

    elif args.model_type == "transformer_vae":
        from src.baselines.transformer_vae import TransformerVAE
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_args = checkpoint.get("args", {})
        model = TransformerVAE(
            vocab_size=len(tokenizer.stoi),
            pad_id=tokenizer.pad_id,
            embed_dim=128,
            latent_dim=model_args.get("latent_dim", 64),
            nhead=model_args.get("nhead", 4),
            num_encoder_layers=model_args.get("num_layers", 2),
            num_decoder_layers=model_args.get("num_layers", 2),
            dim_feedforward=256,
        ).to(device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        embeddings = extract_transformer_vae_embeddings(model, loader, device)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.save(args.output_path, embeddings)
    print(f"\n✓ Saved embeddings: {args.output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")


if __name__ == "__main__":
    main()
