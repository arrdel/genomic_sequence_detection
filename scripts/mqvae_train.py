#!/usr/bin/env python3
"""
Masked VQ-VAE Training Script for Viral Genome Sequences

This script implements masked language modeling for VQ-VAE, where random tokens
are masked during training and the model learns to reconstruct them.

Usage:
    python -u scripts/mqvae_train.py \
      --data-path /path/to/cleaned_reads.fastq \
      --output-dir ./outputs \
      --experiment-name mqvae_train1 \
      --n-gpu 4 \
      --batch-size 64 \
      --epochs 50 \
      --mask-prob 0.2
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from Bio import SeqIO
import wandb
from datetime import datetime
import tqdm
import random
import json
import numpy as np

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
from src.data import KmerTokenizer, FastqKmerDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Masked VQ-VAE on viral genome sequences')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test set ratio (default: 0.1 = 10%% test, 90%% train)')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=4097,
                        help='Vocabulary size (4^k + 1 for PAD)')
    parser.add_argument('--num-codes', type=int, default=512,
                        help='Number of codebook vectors')
    parser.add_argument('--code-dim', type=int, default=128,
                        help='Dimension of codebook vectors')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension in encoder/decoder')
    parser.add_argument('--commitment-cost', type=float, default=0.1,
                        help='Commitment cost for VQ loss')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    
    # Masking arguments
    parser.add_argument('--mask-prob', type=float, default=0.2,
                        help='Probability of masking each token')
    parser.add_argument('--mask-token', type=str, default='<MASK>',
                        help='Special mask token')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--recon-loss-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU arguments
    parser.add_argument('--n-gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Checkpoint arguments
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--wandb-project', type=str, default='mqvae-genomics',
                        help='Weights & Biases project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Log metrics every N batches')
    
    # Evaluation arguments
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--eval-batches', type=int, default=50,
                        help='Number of batches to evaluate')
    parser.add_argument('--visualize-examples', type=int, default=5,
                        help='Number of examples to visualize during evaluation')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_devices(args):
    """Setup GPU/CPU devices"""
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
        return device, []
    
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.n_gpu))
    
    device = torch.device(f'cuda:{gpu_ids[0]}')
    print(f"Using GPUs: {gpu_ids}")
    return device, gpu_ids


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


def apply_random_mask(batch_tokens, pad_id, mask_id, mask_prob=0.2):
    """
    Randomly replaces ~mask_prob fraction of non-PAD tokens with <MASK>.
    
    Args:
        batch_tokens: (B, L) tensor of token IDs
        pad_id: ID of PAD token
        mask_id: ID of MASK token
        mask_prob: Probability of masking each token
    
    Returns:
        masked_tokens: (B, L) tensor with some tokens replaced by mask_id
        mask_positions: (B, L) boolean tensor indicating masked positions
    """
    B, L = batch_tokens.shape
    mask_positions = (torch.rand(B, L, device=batch_tokens.device) < mask_prob) & (batch_tokens != pad_id)
    masked_tokens = batch_tokens.clone()
    masked_tokens[mask_positions] = mask_id
    return masked_tokens, mask_positions


def train_one_epoch(model, dataloader, optimizer, device, pad_id, mask_id, args, epoch):
    """
    Train for one epoch with masked language modeling
    
    Args:
        model: VQ-VAE model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        pad_id: PAD token ID
        mask_id: MASK token ID
        args: Arguments
        epoch: Current epoch number
    
    Returns:
        dict: Training statistics
    """
    model.train()
    total_loss, total_recon, total_vq, n_masked = 0, 0, 0, 0
    
    for batch_idx, (batch_tokens, batch_lengths) in enumerate(dataloader):
        batch_tokens = batch_tokens.to(device)
        batch_lengths = batch_lengths.to(device)
        
        # Apply random masking
        masked_inputs, mask_positions = apply_random_mask(
            batch_tokens, pad_id, mask_id, mask_prob=args.mask_prob
        )
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss_vq, codes = model(masked_inputs)
        B, L, V = logits.shape
        
        # Prevent model from predicting PAD tokens
        logits[:, :, pad_id] = -1e9
        
        # Compute loss only on masked positions
        valid_mask = mask_positions & (batch_tokens != pad_id)
        
        if valid_mask.sum() == 0:
            continue  # Skip batch if no tokens were masked
        
        logits_masked = logits[valid_mask]      # (N_masked, V)
        targets_masked = batch_tokens[valid_mask]  # (N_masked,)
        
        # Reconstruction loss
        recon_loss = F.cross_entropy(logits_masked, targets_masked)
        
        # Handle DataParallel: loss_vq may be a vector if using multiple GPUs
        if loss_vq.dim() > 0:
            loss_vq = loss_vq.mean()
        
        # Total loss
        loss = args.recon_loss_weight * recon_loss + loss_vq
        
        loss.backward()
        optimizer.step()
        
        # Update statistics
        n_valid = targets_masked.size(0)
        total_loss += loss.item() * n_valid
        total_recon += recon_loss.item() * n_valid
        total_vq += loss_vq.item() * n_valid
        n_masked += n_valid
        
        # Log to wandb periodically
        if not args.no_wandb and batch_idx % args.log_freq == 0:
            batch_loss = loss.item() if loss.dim() == 0 else loss.mean().item()
            batch_recon = recon_loss.item() if recon_loss.dim() == 0 else recon_loss.mean().item()
            batch_vq = loss_vq.item() if loss_vq.dim() == 0 else loss_vq.mean().item()
            
            wandb.log({
                "train/batch_loss": batch_loss,
                "train/batch_recon_loss": batch_recon,
                "train/batch_vq_loss": batch_vq,
                "train/masked_tokens": n_valid,
                "global_step": epoch * len(dataloader) + batch_idx
            })
        
        # Free up memory
        del logits, loss_vq, codes, loss
        torch.cuda.empty_cache()
    
    return {
        "loss": total_loss / n_masked,
        "recon": total_recon / n_masked,
        "vq": total_vq / n_masked,
        "n_masked": n_masked
    }


def validate(model, dataloader, device, pad_id, mask_id, args):
    """
    Validate the model on test set
    
    Args:
        model: VQ-VAE model
        dataloader: Test dataloader
        device: Device
        pad_id: PAD token ID
        mask_id: MASK token ID
        args: Arguments
    
    Returns:
        dict: Test statistics
    """
    model.eval()
    total_loss, total_recon, total_vq, n_masked = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_tokens, batch_lengths in dataloader:
            batch_tokens = batch_tokens.to(device)
            batch_lengths = batch_lengths.to(device)
            
            # Apply random masking
            masked_inputs, mask_positions = apply_random_mask(
                batch_tokens, pad_id, mask_id, mask_prob=args.mask_prob
            )
            
            # Forward pass
            logits, loss_vq, _ = model(masked_inputs)
            B, L, V = logits.shape
            
            # Prevent PAD prediction
            logits[:, :, pad_id] = -1e9
            
            # Compute loss only on masked positions
            valid_mask = mask_positions & (batch_tokens != pad_id)
            
            if valid_mask.sum() == 0:
                continue
            
            logits_masked = logits[valid_mask]
            targets_masked = batch_tokens[valid_mask]
            
            recon_loss = F.cross_entropy(logits_masked, targets_masked)
            
            if loss_vq.dim() > 0:
                loss_vq = loss_vq.mean()
            
            loss = args.recon_loss_weight * recon_loss + loss_vq
            
            n_valid = targets_masked.size(0)
            total_loss += loss.item() * n_valid
            total_recon += recon_loss.item() * n_valid
            total_vq += loss_vq.item() * n_valid
            n_masked += n_valid
    
    return {
        "loss": total_loss / n_masked if n_masked > 0 else 0.0,
        "recon": total_recon / n_masked if n_masked > 0 else 0.0,
        "vq": total_vq / n_masked if n_masked > 0 else 0.0,
        "n_masked": n_masked
    }


def visualize_masked_reconstruction(model, batch_tokens, batch_lengths, tokenizer, 
                                   device, pad_id, mask_id, mask_prob, num_examples=5):
    """
    Visualize masked reconstruction examples
    
    Args:
        model: VQ-VAE model
        batch_tokens: Batch of token sequences
        batch_lengths: Sequence lengths
        tokenizer: Tokenizer
        device: Device
        pad_id: PAD token ID
        mask_id: MASK token ID
        mask_prob: Masking probability
        num_examples: Number of examples to show
    
    Returns:
        str: Formatted visualization string
    """
    model.eval()
    batch_tokens = batch_tokens.to(device)
    batch_lengths = batch_lengths.to(device)
    
    # Apply random masking
    masked_inputs, mask_positions = apply_random_mask(batch_tokens, pad_id, mask_id, mask_prob)
    
    with torch.no_grad():
        logits, _, _ = model(masked_inputs)
        preds = logits.argmax(dim=-1).cpu()
    
    output_lines = []
    output_lines.append("\n" + "="*80)
    output_lines.append("MASKED RECONSTRUCTION EXAMPLES")
    output_lines.append("="*80)
    
    for i in range(min(num_examples, batch_tokens.size(0))):
        L_true = int(batch_lengths[i].item())
        
        # Get sequences
        true_ids = batch_tokens[i, :L_true].cpu().tolist()
        mask_ids = masked_inputs[i, :L_true].cpu().tolist()
        pred_ids = preds[i, :L_true].tolist()
        
        # Decode to k-mers
        true_kmers = tokenizer.decode_tokens(true_ids, remove_pad=True, reconstruct=False)
        mask_kmers = tokenizer.decode_tokens(mask_ids, remove_pad=True, reconstruct=False)
        pred_kmers = tokenizer.decode_tokens(pred_ids, remove_pad=True, reconstruct=False)
        
        # Find masked positions
        masked_pos = [j for j, m in enumerate(mask_positions[i, :L_true].cpu().tolist()) if m]
        
        # Calculate accuracy on masked positions
        if masked_pos:
            correct = sum(1 for j in masked_pos if pred_ids[j] == true_ids[j])
            accuracy = correct / len(masked_pos) * 100
        else:
            accuracy = 0.0
        
        output_lines.append(f"\n--- Example {i+1} ---")
        output_lines.append(f"Original  (first 15): {true_kmers[:15]}")
        output_lines.append(f"Masked    (first 15): {mask_kmers[:15]}")
        output_lines.append(f"Predicted (first 15): {pred_kmers[:15]}")
        output_lines.append(f"Masked positions (first 10): {masked_pos[:10]}")
        output_lines.append(f"Accuracy on masked tokens: {accuracy:.1f}% ({len(masked_pos)} tokens)")
    
    output_lines.append("="*80 + "\n")
    
    return "\n".join(output_lines)


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f"mqvae_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Setup devices
    device, gpu_ids = setup_devices(args)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args)
        )
    
    # Create tokenizer and add mask token
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=True)
    tokenizer.add_special_token(args.mask_token)
    mask_id = tokenizer.mask_id
    
    print(f"Vocabulary size: {len(tokenizer.stoi)}")
    print(f"PAD ID: {tokenizer.pad_id}")
    print(f"MASK ID: {mask_id}")
    
    # Initialize model
    VOCAB_SIZE = len(tokenizer.stoi)
    PAD_ID = tokenizer.pad_id
    
    model = VQVAE(
        VOCAB_SIZE,
        PAD_ID,
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        commitment_cost=args.commitment_cost
    )
    
    # Multi-GPU support
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    if not args.no_wandb:
        wandb.watch(model, log="all")
    
    # Create dataset
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Split into train/test
    dataset_size = len(dataset)
    test_size = int(args.test_split * dataset_size)
    train_size = dataset_size - test_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    print(f"\nDataset split:")
    print(f"  Total sequences: {dataset_size}")
    print(f"  Training: {train_size} ({(1-args.test_split)*100:.1f}%)")
    print(f"  Test: {test_size} ({args.test_split*100:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel prefix
        if len(gpu_ids) > 1 and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif len(gpu_ids) <= 1 and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Masking probability: {args.mask_prob}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Track best model
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            # Training
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, PAD_ID, mask_id, args, epoch
            )
            
            # Test evaluation
            if epoch % args.eval_freq == 0:
                test_stats = validate(model, test_loader, device, PAD_ID, mask_id, args)
                
                # Print training and test stats
                print(f"Epoch {epoch}/{args.epochs}:")
                print(f"  [TRAIN] loss={train_stats['loss']:.4f} "
                      f"recon={train_stats['recon']:.4f} "
                      f"vq={train_stats['vq']:.4f} "
                      f"(n_masked={train_stats['n_masked']})")
                print(f"  [TEST]  loss={test_stats['loss']:.4f} "
                      f"recon={test_stats['recon']:.4f} "
                      f"vq={test_stats['vq']:.4f} "
                      f"(n_masked={test_stats['n_masked']})")
                
                # Log metrics
                if not args.no_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_stats['loss'],
                        "train/recon_loss": train_stats['recon'],
                        "train/vq_loss": train_stats['vq'],
                        "train/n_masked_tokens": train_stats['n_masked'],
                        "test/loss": test_stats['loss'],
                        "test/recon_loss": test_stats['recon'],
                        "test/vq_loss": test_stats['vq'],
                        "test/n_masked_tokens": test_stats['n_masked']
                    })
                
                # Visualize examples from test set
                batch_tokens, batch_lengths = next(iter(test_loader))
                viz_text = visualize_masked_reconstruction(
                    model, batch_tokens, batch_lengths, tokenizer,
                    device, PAD_ID, mask_id, args.mask_prob,
                    num_examples=args.visualize_examples
                )
                print(viz_text)
                
                # Save visualization to file
                viz_file = os.path.join(output_dir, f'reconstruction_epoch_{epoch}.txt')
                with open(viz_file, 'w') as f:
                    f.write(viz_text)
                
                # Save best model based on test loss
                if test_stats['loss'] < best_test_loss:
                    best_test_loss = test_stats['loss']
                    best_path = os.path.join(output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_stats': train_stats,
                        'test_stats': test_stats,
                        'args': vars(args)
                    }, best_path)
                    print(f"  ✓ New best model saved! (test_loss: {test_stats['loss']:.4f})")
            
            else:
                print(f"Epoch {epoch}/{args.epochs}: "
                      f"train_loss={train_stats['loss']:.4f} "
                      f"recon={train_stats['recon']:.4f} "
                      f"vq={train_stats['vq']:.4f}")
            
            # Save checkpoint
            if epoch % args.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_stats': train_stats,
                    'args': vars(args)
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Checkpoint saved to: {checkpoint_path}")
                
                if not args.no_wandb:
                    artifact = wandb.Artifact(f'mqvae-checkpoint-{epoch}', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
        
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            raise
    
    # Clean up wandb
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
