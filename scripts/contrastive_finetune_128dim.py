#!/usr/bin/env python3
"""
Contrastive Learning Fine-tuning for VQ-VAE with 128-dim Projection

This script fine-tunes a pretrained VQ-VAE encoder using contrastive learning
with InfoNCE loss and data augmentation. Uses 128-dim projection (matching 
masked VQ-VAE dimensionality) for better clustering compared to 64-dim.

Key Difference from Standard Contrastive:
- proj_dim=128 (instead of 64) to match masked VQ-VAE encoder dimensionality
- Preserves more information for downstream clustering tasks
- Saves to separate output directory: experiments/3_contrastive_vqvae_128dim/

Usage:
      
python -u scripts/contrastive_finetune_128dim.py \
  --data-path cleaned_reads.fastq \
  --checkpoint-path experiments/1_standard_vqvae/checkpoints/best_model.pt \
  --output-dir experiments/3_contrastive_vqvae_128dim \
  --experiment-name contrastive_128dim_run_1 \
  --n-gpu 2 \
  --epochs 50 \
  --proj-dim 128
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports using src structure
from src.data import KmerTokenizer, FastqKmerDataset
from src.models import VQVAE

# Optional UMAP (will skip plotting if not installed)
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Contrastive fine-tuning for VQ-VAE encoder with 128-dim projection"
    )
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to pretrained VQ-VAE checkpoint')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    parser.add_argument('--use-canonical', action='store_true',
                        help='Use canonical k-mers')
    parser.add_argument('--proj-dim', type=int, default=128,
                        help='Projection head output dimension (default: 128 to match masked VQ-VAE)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--mask-prob', type=float, default=0.15,
                        help='Probability of masking tokens for augmentation')
    parser.add_argument('--drop-prob', type=float, default=0.10,
                        help='Probability of dropping tokens for augmentation')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value (None to disable)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='experiments/3_contrastive_vqvae_128dim',
                        help='Directory to save outputs')
    parser.add_argument('--experiment-name', type=str, default='contrastive_128dim_1',
                        help='Experiment name for output directory')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Logging arguments
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='genomic-vqvae-contrastive-128dim',
                        help='Weights & Biases project name')
    
    # GPU arguments
    parser.add_argument('--n-gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu-ids', type=str, default="2,3",
                        help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_devices(args):
    """Setup CUDA devices for training"""
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        use_data_parallel = False
        gpu_ids = []
        print("Using device: CPU")
    else:
        if args.gpu_ids:
            gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(args.n_gpu))
        
        device = torch.device(f'cuda:{gpu_ids[0]}')
        use_data_parallel = len(gpu_ids) > 1
        
        if use_data_parallel:
            print(f"Using DataParallel with GPUs: {gpu_ids}")
        else:
            print(f"Using device: GPU {gpu_ids[0]}")
    
    return device, use_data_parallel, gpu_ids


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([b[1] for b in batch])
    return tokens, lengths


def augment_batch(batch_tokens, pad_id, vocab_size, mask_prob=0.15, drop_prob=0.10):
    """
    Create two augmented views of the same batch for contrastive learning:
      - view1: random token replacement (with random valid tokens)
      - view2: random dropout (replace with PAD)
    
    Args:
        batch_tokens: (B, L) tensor of token IDs
        pad_id: ID of PAD token
        vocab_size: Size of vocabulary (for random token replacement)
        mask_prob: Probability of replacing each token with random token
        drop_prob: Probability of dropping each token (replace with PAD)
    
    Returns:
        view1: (B, L) tensor with random token replacement
        view2: (B, L) tensor with random dropout
    """
    def random_replace(toks):
        # Replace non-PAD tokens with random valid tokens
        mask = (torch.rand_like(toks.float()) < mask_prob) & (toks != pad_id)
        out = toks.clone()
        # Random tokens from [0, vocab_size - 1), excluding pad_id
        random_tokens = torch.randint(0, vocab_size - 1, toks.shape, device=toks.device)
        out[mask] = random_tokens[mask]
        return out

    def random_drop(toks):
        # Drop non-PAD tokens by replacing with PAD
        drop = (torch.rand_like(toks.float()) < drop_prob) & (toks != pad_id)
        out = toks.clone()
        out[drop] = pad_id
        return out

    return random_replace(batch_tokens), random_drop(batch_tokens)


class ContrastiveHead128(nn.Module):
    """
    Wraps a pretrained VQ-VAE encoder with a projection head for contrastive learning.
    Uses 128-dim projection (matching masked VQ-VAE) for better clustering.
    
    Architecture:
      encoder(tokens) -> [B, L, D=128]
      mean pool -> [B, 128]
      projection -> [B, 128]
      normalize -> [B, 128] with ||z|| = 1
    """
    def __init__(self, encoder, embed_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, tokens):
        z_e = self.encoder(tokens)           # [B, L, D=128]
        z_mean = z_e.mean(dim=1)             # [B, 128] - mean pooling
        z_proj = self.proj(z_mean)           # [B, 128] - projection
        z_norm = F.normalize(z_proj, dim=-1) # [B, 128] unit norm for cosine similarity
        return z_norm


def info_nce_loss(z1, z2, temperature=0.5):
    """
    InfoNCE (NT-Xent) Loss for contrastive learning.
    
    Args:
        z1, z2: [B, D] normalized embeddings, two views of the same batch
        temperature: Temperature scaling parameter
    
    Returns:
        loss: Scalar InfoNCE loss
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                          # [2B, D]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # [2B, 2B]
    sim = sim / temperature

    # mask self-similarity (diagonal)
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -9e15)

    # positive pairs indices: (i <-> i+B)
    pos_idx = torch.arange(B, device=z.device)
    pos_sim_1 = sim[0:B, B:2*B][pos_idx, pos_idx]           # z1 vs z2
    pos_sim_2 = sim[B:2*B, 0:B][pos_idx, pos_idx]           # z2 vs z1
    positives = torch.cat([pos_sim_1, pos_sim_2], dim=0)    # [2B]

    # denominator over all pairs in each row
    denom = torch.logsumexp(sim, dim=-1)                    # [2B]
    loss = -positives + denom
    return loss.mean()


def train_one_epoch(model, dataloader, optimizer, device, pad_id, vocab_size,
                    mask_prob=0.15, drop_prob=0.10, temperature=0.5,
                    grad_clip=None, epoch=1, use_wandb=False):
    """Train contrastive model for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=True)
    for batch_tokens, _ in pbar:
        batch_tokens = batch_tokens.to(device, non_blocking=True)

        # Create two augmented views
        v1, v2 = augment_batch(batch_tokens, pad_id, vocab_size,
                               mask_prob=mask_prob, drop_prob=drop_prob)

        z1 = model(v1)
        z2 = model(v2)

        loss = info_nce_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if use_wandb:
            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch
            })

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def validate(model, dataloader, device, pad_id, vocab_size,
             mask_prob=0.15, drop_prob=0.10, temperature=0.5):
    """Validate contrastive model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_tokens, _ in tqdm(dataloader, desc="Validation", leave=False):
            batch_tokens = batch_tokens.to(device, non_blocking=True)

            v1, v2 = augment_batch(batch_tokens, pad_id, vocab_size,
                                   mask_prob=mask_prob, drop_prob=drop_prob)

            z1 = model(v1)
            z2 = model(v2)

            loss = info_nce_loss(z1, z2, temperature=temperature)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def extract_embeddings_for_viz(model, dataloader, device, max_samples=2000):
    """Extract embeddings for visualization"""
    model.eval()
    embeddings = []
    num_samples = 0
    
    for batch_tokens, _ in dataloader:
        if num_samples >= max_samples:
            break
        
        batch_tokens = batch_tokens.to(device)
        z = model(batch_tokens)  # [B, proj_dim]
        embeddings.append(z.cpu().numpy())
        num_samples += z.size(0)
    
    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    return embeddings


def plot_embedding_space(embeddings, out_path):
    """Plot UMAP projection of embedding space"""
    if not HAS_UMAP:
        print("[WARN] umap-learn not installed; skipping visualization.")
        return
    
    print("Computing UMAP projection for visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=5, alpha=0.5)
    ax.set_title('Contrastive Embeddings (128-dim) - UMAP Projection', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Saved embedding visualization: {out_path}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup devices
    device, use_data_parallel, gpu_ids = setup_devices(args)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    viz_dir = output_dir / 'visualizations'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CONTRASTIVE VQ-VAE TRAINING (128-DIM PROJECTION)")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Projection dimension: {args.proj_dim} (matching masked VQ-VAE)")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Augmentation - Mask prob: {args.mask_prob}, Drop prob: {args.drop_prob}")
    print("="*80 + "\n")
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Load pretrained VQ-VAE checkpoint
    print(f"Loading pretrained VQ-VAE from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        num_codes = model_args.get('num_codes', 512)
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        commitment_cost = model_args.get('commitment_cost', 0.1)
        vocab_size = model_args.get('vocab_size', 4097)
        pad_id = model_args.get('pad_id', 4096)
    else:
        # Default values
        num_codes = 512
        code_dim = 128
        embed_dim = 128
        hidden_dim = 256
        commitment_cost = 0.1
        vocab_size = 4097
        pad_id = 4096
    
    print(f"  Model config: vocab={vocab_size}, codes={num_codes}, code_dim={code_dim}")
    print(f"  Encoder output dim: {code_dim}")
    
    # Create base VQ-VAE model
    base_model = VQVAE(
        vocab_size,
        pad_id,
        num_codes=num_codes,
        code_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        commitment_cost=commitment_cost
    ).to(device)
    
    # Load VQ-VAE weights
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)
    
    print("✓ VQ-VAE checkpoint loaded successfully")
    
    # Create contrastive model with projection head
    model = ContrastiveHead128(
        encoder=base_model.encoder,
        embed_dim=code_dim,
        proj_dim=args.proj_dim
    ).to(device)
    
    # Freeze encoder initially (optional - can be unfrozen later)
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    # Use DataParallel if multiple GPUs
    if use_data_parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    print(f"✓ Contrastive model created with {args.proj_dim}-dim projection head")
    
    # Create tokenizer and dataset
    print(f"\nLoading dataset from: {args.data_path}")
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=args.use_canonical)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    print(f"  Total sequences: {len(dataset)}")
    
    # Split into train/val
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"  Train sequences: {train_size}")
    print(f"  Val sequences: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, pad_id, vocab_size,
            mask_prob=args.mask_prob,
            drop_prob=args.drop_prob,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            epoch=epoch,
            use_wandb=args.use_wandb
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, pad_id, vocab_size,
            mask_prob=args.mask_prob,
            drop_prob=args.drop_prob,
            temperature=args.temperature
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            
            # Get model state (handle DataParallel)
            model_state = model.module.state_dict() if use_data_parallel else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            
            print(f"  → Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / 'best_model.pt'
            
            model_state = model.module.state_dict() if use_data_parallel else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, best_model_path)
            
            print(f"  ★ New best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
    
    # Plot training curves
    print("\nPlotting training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    ax.plot(epochs_range, val_losses, label='Val Loss', marker='s')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('InfoNCE Loss', fontsize=12)
    ax.set_title('Contrastive Training (128-dim Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    curve_path = viz_dir / 'training_curves.png'
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"✓ Saved training curves: {curve_path}")
    
    # Extract and visualize final embeddings
    print("\nExtracting embeddings for visualization...")
    embeddings = extract_embeddings_for_viz(model, val_loader, device, max_samples=2000)
    print(f"  Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Save embeddings
    emb_path = viz_dir / 'final_embeddings.npy'
    np.save(emb_path, embeddings)
    print(f"✓ Saved embeddings: {emb_path}")
    
    # Plot UMAP
    umap_path = viz_dir / 'embedding_space_umap.png'
    plot_embedding_space(embeddings, umap_path)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Output directory: {output_dir}")
    print(f"Projection dimension: {args.proj_dim}")
    print("\nGenerated files:")
    print(f"  • checkpoints/best_model.pt - Best model weights")
    print(f"  • checkpoints/checkpoint_epoch_*.pt - Periodic checkpoints")
    print(f"  • visualizations/training_curves.png - Loss curves")
    print(f"  • visualizations/embedding_space_umap.png - UMAP visualization")
    print(f"  • visualizations/final_embeddings.npy - Extracted embeddings")
    print("="*80 + "\n")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
