#!/usr/bin/env python3
"""
VQ-VAE Training Script for Viral Genome Sequences

Usage:
    python -u vqvae_train.py.py --data-path /path/to/data --output-dir ./outputs --n-gpu 2 --batch-size 64 --epochs 100
    
    python -u scripts/train.py \
  --data-path /home/adelechinda/home/semester_projects/fall_25/deep_learning/project/cleaned_reads.fastq \
  --output-dir ./outputs \
  --experiment-name vqvae_train_run_4 \
  --n-gpu 4 \
  --batch-size 128 \
  --epochs 10
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


# from src.models import VQVAE, VectorQuantizerEMA as VectorQuantizer, Encoder, Decoder
from src.models import VQVAE, Encoder, Decoder

from src.data import KmerTokenizer, FastqKmerDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train VQ-VAE on viral genome sequences')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=4097,
                        help='Vocabulary size (4^k + 1 for PAD)')
    parser.add_argument('--num-codes', type=int, default=512,
                        help='Number of codebook vectors')
    parser.add_argument('--code-dim', type=int, default=64,
                        help='Dimension of codebook vectors')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension in encoder/decoder')
    parser.add_argument('--commitment-cost', type=float, default=0.1,
                        help='Commitment cost for VQ loss')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--recon-loss-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    
    # GPU arguments
    parser.add_argument('--n-gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Checkpoint arguments
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--wandb-project', type=str, default='vqvae-genomics',
                        help='Weights & Biases project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Log metrics every N batches')
    
    # Evaluation arguments
    parser.add_argument('--eval-batches', type=int, default=100,
                        help='Number of batches to evaluate for reconstruction')
    parser.add_argument('--eval-samples-per-batch', type=int, default=3,
                        help='Number of samples per batch for reconstruction eval')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test set ratio (default: 0.1 = 10%% test, 90%% train)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Dead code refresh arguments
    parser.add_argument('--refresh-codes', action='store_true',
                        help='Enable dead code refresh mechanism')
    parser.add_argument('--refresh-interval', type=int, default=500,
                        help='Refresh dead codes every N batches')
    parser.add_argument('--refresh-buffer-size', type=int, default=10000,
                        help='Size of encoder output buffer for code refresh')
    parser.add_argument('--refresh-min-count', type=int, default=5,
                        help='Minimum EMA count threshold for dead codes')
    
    args = parser.parse_args()
    return args


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


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


def refresh_dead_codes(vq, z_buffer, min_count=5):
    """
    Reactivate dead codebook entries that are not being used.
    
    Args:
        vq: VectorQuantizerEMA instance (model.vq)
        z_buffer: torch.Tensor of recent encoder outputs (N, D)
        min_count: codes with EMA count below this threshold get refreshed
    """
    with torch.no_grad():
        dead = (vq.ema_cluster_size < min_count)
        if dead.any():
            k = dead.nonzero(as_tuple=False).flatten()
            z_samples = z_buffer[torch.randint(0, z_buffer.size(0), (k.numel(),))]
            vq.embedding.weight.data[k] = z_samples
            vq.ema_w[k] = z_samples
            vq.ema_cluster_size[k] = min_count
            print(f"🔄 Refreshed {k.numel()} dead codes")




def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    pad_id,
    args,
    epoch,
    refresh_codes=False,
    refresh_interval=500,
    buffer_size=10000
):
    """
    TRAIN ONE EPOCH — COLLAPSE-RESISTANT VERSION
    """

    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    total_entropy = 0
    n_tokens = 0

    # Buffer for encoder outputs (optional)
    z_buffer = [] if refresh_codes else None
    max_buffer_size = buffer_size

    for batch_idx, (batch_tokens, batch_lengths) in enumerate(dataloader):
        batch_tokens = batch_tokens.to(device)
        batch_lengths = batch_lengths.to(device)

        optimizer.zero_grad()

        # -----------------------------
        # Forward pass
        # -----------------------------
        logits, loss_vq, codes = model(batch_tokens)
        B, L, V = logits.shape

        # Never predict PAD
        logits[:, :, pad_id] = -1e9

        # Valid mask
        valid_mask = (
            (torch.arange(L, device=device)[None, :] < batch_lengths[:, None])
            & (batch_tokens != pad_id)
        )

        if valid_mask.sum() == 0:
            continue

        # -----------------------------
        # Reconstruction (Cross-Entropy)
        # -----------------------------
        logits_flat = logits[valid_mask]          # (N_valid, V)
        targets_flat = batch_tokens[valid_mask]   # (N_valid,)

        ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
        n_valid = valid_mask.sum().clamp_min(1)
        recon_loss = ce / n_valid

        # VQ loss fix for DataParallel
        if loss_vq.dim() > 0:
            loss_vq = loss_vq.mean()

        # -----------------------------
        # ENTROPY BONUS (Anti-collapse)
        # -----------------------------
        with torch.no_grad():
            if hasattr(model, "module"):
                K = model.module.vq.num_codes
            else:
                K = model.vq.num_codes

            hist = torch.bincount(
                codes.reshape(-1),
                minlength=K
            ).float()

            p = hist / (hist.sum() + 1e-9)
            H = -(p * (p + 1e-9).log()).sum()    # entropy

        entropy_bonus = 3e-3 * H   # ⬅ Key fix (increase usage to 60–90%)

        # -----------------------------
        # Total Loss
        # -----------------------------
        loss = (
            args.recon_loss_weight * recon_loss
            + loss_vq
            - entropy_bonus       # ⬅ CRITICAL FIX
        )

        # Backprop
        loss.backward()
        optimizer.step()

        # -----------------------------
        # Logging
        # -----------------------------
        n_valid_num = n_valid.item()
        total_loss += loss.item() * n_valid_num
        total_recon += recon_loss.item() * n_valid_num
        total_vq += loss_vq.item() * n_valid_num
        # Note: entropy_bonus is SUBTRACTED in the loss, so we track it separately
        # and won't include it in the total_loss verification
        total_entropy += entropy_bonus.item() * n_valid_num
        n_tokens += n_valid_num

        # -----------------------------
        # Optional dead-code refresh
        # -----------------------------
        if refresh_codes and batch_idx % refresh_interval == 0:
            with torch.no_grad():
                if hasattr(model, "module"):
                    z_e = model.module.encoder(batch_tokens)
                else:
                    z_e = model.encoder(batch_tokens)

                # add to buffer
                z_buffer.append(z_e.reshape(-1, z_e.size(-1)).cpu())
                if len(z_buffer) > max_buffer_size // B:
                    z_buffer.pop(0)

                # concatenate
                z_concat = torch.cat(z_buffer, dim=0).to(device)

                if hasattr(model, "module"):
                    refresh_dead_codes(model.module.vq, z_concat, min_count=args.refresh_min_count)
                else:
                    refresh_dead_codes(model.vq, z_concat, min_count=args.refresh_min_count)

        # Free memory
        del logits, loss_vq, codes, loss
        torch.cuda.empty_cache()

    # -----------------------------
    # Return epoch statistics
    # -----------------------------
    return {
        "loss": total_loss / n_tokens,
        "recon": total_recon / n_tokens,
        "vq": total_vq / n_tokens,
        "entropy": total_entropy / n_tokens,
    }


def validate_one_epoch(model, dataloader, device, pad_id, args):
    """
    Validate on test/validation set
    """
    model.eval()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    total_entropy = 0
    n_tokens = 0

    with torch.no_grad():
        for batch_tokens, batch_lengths in dataloader:
            batch_tokens = batch_tokens.to(device)
            batch_lengths = batch_lengths.to(device)

            # Forward pass
            logits, loss_vq, codes = model(batch_tokens)
            B, L, V = logits.shape

            # Prevent PAD output
            logits[:, :, pad_id] = -1e9

            # Valid positions
            valid_mask = (
                (torch.arange(L, device=device)[None, :] < batch_lengths[:, None])
                & (batch_tokens != pad_id)
            )

            if valid_mask.sum() == 0:
                continue

            # Reconstruction loss
            logits_flat = logits[valid_mask]
            targets_flat = batch_tokens[valid_mask]
            ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
            n_valid = valid_mask.sum().clamp_min(1)
            recon_loss = ce / n_valid

            # VQ loss fix for DataParallel
            if loss_vq.dim() > 0:
                loss_vq = loss_vq.mean()

            # Entropy calculation
            if hasattr(model, "module"):
                K = model.module.vq.num_codes
            else:
                K = model.vq.num_codes

            hist = torch.bincount(codes.reshape(-1), minlength=K).float()
            p = hist / (hist.sum() + 1e-9)
            H = -(p * (p + 1e-9).log()).sum()
            entropy_bonus = 3e-3 * H

            # Total loss
            loss = args.recon_loss_weight * recon_loss + loss_vq - entropy_bonus

            # Accumulate statistics
            n_valid_num = n_valid.item()
            total_loss += loss.item() * n_valid_num
            total_recon += recon_loss.item() * n_valid_num
            total_vq += loss_vq.item() * n_valid_num
            total_entropy += entropy_bonus.item() * n_valid_num
            n_tokens += n_valid_num

    return {
        "loss": total_loss / n_tokens,
        "recon": total_recon / n_tokens,
        "vq": total_vq / n_tokens,
        "entropy": total_entropy / n_tokens,
    }


def reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device):
    """Reconstruct sequences from model"""
    model.eval()
    batch_tokens = batch_tokens.to(device)
    batch_lengths = batch_lengths.to(device)

    with torch.no_grad():
        logits, _, _ = model(batch_tokens)
        preds = logits.argmax(dim=-1).cpu()

    results = []
    for i in range(batch_tokens.size(0)):
        L_true = int(batch_lengths[i].item())
        input_ids = batch_tokens[i, :L_true].cpu().numpy()
        pred_ids = preds[i, :L_true].numpy()

        input_seq = tokenizer.decode_tokens(input_ids, remove_pad=True, reconstruct=True)
        recon_seq = tokenizer.decode_tokens(pred_ids, remove_pad=True, reconstruct=True)

        results.append((input_seq, recon_seq))

    return results


def evaluate_reconstruction(model, dataloader, tokenizer, device, output_dir, args):
    """Evaluate reconstruction quality"""
    print(f"\nGenerating reconstruction examples from {args.eval_batches} random batches...")
    reconstruction_file = os.path.join(output_dir, 'sequence_reconstructions.txt')

    with open(reconstruction_file, 'w') as f:
        all_batches = list(dataloader)
        num_batches = min(args.eval_batches, len(all_batches))
        random_batches = random.sample(all_batches, num_batches)

        for batch_idx, (batch_tokens, batch_lengths) in enumerate(tqdm.tqdm(random_batches)):
            examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device)

            batch_size = len(examples)
            sample_indices = random.sample(range(batch_size), min(args.eval_samples_per_batch, batch_size))

            batch_header = f"\n{'='*20} Batch {batch_idx + 1} {'='*20}\n"
            f.write(batch_header)

            for sample_idx, i in enumerate(sample_indices):
                in_seq, pred_seq = examples[i]

                example_text = f"---- Example {sample_idx + 1} ----\n"
                example_text += f"Input seq (first 120 bp):  {in_seq[:120]}\n"
                example_text += f"Recon seq (first 120 bp):  {pred_seq[:120]}\n"
                match_rate = sum(a==b for a,b in zip(in_seq[:120], pred_seq[:120]))/max(1, len(in_seq[:120]))
                example_text += f"Match rate: {match_rate:.2%}\n\n"

                f.write(example_text)

    print(f"Reconstruction examples saved to: {reconstruction_file}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f"vqvae_{timestamp}"
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
    
    # Initialize model
    PAD_ID = args.vocab_size - 1
    model = VQVAE(
        args.vocab_size, 
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
    
    # Create dataset and split into train/test
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=True)
    full_dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Split dataset into train and test
    dataset_size = len(full_dataset)
    test_size = int(args.test_split * dataset_size)
    train_size = dataset_size - test_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
    
    print(f"\nDataset split:")
    print(f"  Total sequences: {dataset_size}")
    print(f"  Training: {train_size} ({(1-args.test_split)*100:.1f}%)")
    print(f"  Test: {test_size} ({args.test_split*100:.1f}%)")
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle state dict loading
        state_dict = checkpoint['model_state_dict']
        
        # If model is DataParallel wrapped but checkpoint has 'module.' prefix
        if len(gpu_ids) > 1 and not list(state_dict.keys())[0].startswith('module.'):
            # Add 'module.' prefix
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        # If model is NOT DataParallel but checkpoint has 'module.' prefix
        elif len(gpu_ids) <= 1 and list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Total dataset size: {dataset_size}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Track best model
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            # Train on training set
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, PAD_ID, args, epoch,
                refresh_codes=args.refresh_codes,
                refresh_interval=args.refresh_interval,
                buffer_size=args.refresh_buffer_size
            )
            
            # Validate on test set
            test_stats = validate_one_epoch(model, test_loader, device, PAD_ID, args)
            
            # Print training stats with all components for verification
            # Formula: loss = recon + vq - entropy
            train_expected = train_stats['recon'] + train_stats['vq'] - train_stats['entropy']
            test_expected = test_stats['recon'] + test_stats['vq'] - test_stats['entropy']
            
            print(f"Epoch {epoch}/{args.epochs}:")
            print(f"  [TRAIN] loss={train_stats['loss']:.6f} "
                  f"recon={train_stats['recon']:.6f} "
                  f"vq={train_stats['vq']:.6f} "
                  f"entropy={train_stats['entropy']:.6f} "
                  f"(expected={train_expected:.6f})")
            print(f"  [TEST]  loss={test_stats['loss']:.6f} "
                  f"recon={test_stats['recon']:.6f} "
                  f"vq={test_stats['vq']:.6f} "
                  f"entropy={test_stats['entropy']:.6f} "
                  f"(expected={test_expected:.6f})")
            
            # Log metrics to wandb
            if not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_stats['loss'],
                    "train/recon_loss": train_stats['recon'],
                    "train/vq_loss": train_stats['vq'],
                    "train/entropy": train_stats['entropy'],
                    "test/loss": test_stats['loss'],
                    "test/recon_loss": test_stats['recon'],
                    "test/vq_loss": test_stats['vq'],
                    "test/entropy": test_stats['entropy']
                })
            
            # Save checkpoint
            if epoch % args.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'args': vars(args)
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Checkpoint saved to: {checkpoint_path}")
                
                # Log checkpoint as artifact
                if not args.no_wandb:
                    artifact = wandb.Artifact(f'model-checkpoint-{epoch}', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
            
            # Save best model based on test loss
            if test_stats['loss'] < best_test_loss:
                best_test_loss = test_stats['loss']
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'args': vars(args)
                }
                torch.save(checkpoint, best_model_path)
                print(f"  ✓ New best model saved! (test_loss: {test_stats['loss']:.6f})")
        
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            raise
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Training complete! Running final evaluation on TEST SET...")
    print("="*50)
    evaluate_reconstruction(model, test_loader, tokenizer, device, output_dir, args)
    
    # Clean up wandb
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


""""
- Implement evaluation VQVAE
- Implement masked VQVAE, run evaluation on it as well
- Implement latent space visualization for various grouping in latent space.
- Run contrastive on covid dataset
- Compare results on three methods
- Literature writing
- Poster design


- Evaluate all three models on the same test set
- Compare metrics across all approaches:
    - Reconstruction Accuracy
        -Token Accuracy (%) 
        - Sequence Accuracy (%)     
    - Codebook Utilization
        - Codebook Perplexity
        - Alignment (↓)
        - Uniformity (↓)
    - Latent Space Clustering
        - Silhouette Score
        - Davies-Bouldin Index
    
- Visualize embeddings (UMAP/t-SNE) for each model
- Downstream task: Use learned representations for:
  - Viral family classification
  - Anomaly detection (novel viruses)
  - Sequence clustering by virus type
  
  
  
  
  # TODO: Project website
  #* TODO: Wait for training completion
  #* TODO: Poster design
  #? TODO: Literature writing excluding results
  # TODO: Literature writing including results
  # TODO: Visualize embeddings (UMAP/t-SNE) for each model
  
  #? TODO: Train contrastive with 128 dim embedding size
  # TODO: Evaluating all 4 models
  # TODO: Generate t-SNE and umap for all 4 models (10000 samples, 10 clusters) showing mixed tsne, but composition graph shows separation
  # TODO: Understand reason for low codebook utilization and try to fix
  # TODO: Populate report with results
  # TODO: Start google slide
  # TODO: 
  

"""
