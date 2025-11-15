#!/usr/bin/env python3
"""
VQ-VAE Training Script for Viral Genome Sequences

Usage:
    python -u train.py --data-path /path/to/data --output-dir ./outputs --n-gpu 2 --batch-size 64 --epochs 100
    
    python -u scripts/train.py \
  --data-path /home/adelechinda/home/semester_projects/fall_25/deep_learning/project/cleaned_reads.fastq \
  --output-dir ./outputs \
  --experiment-name vqvae_train2 \
  --n-gpu 4 \
  --batch-size 64 \
  --epochs 10
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Bio import SeqIO
import wandb
from datetime import datetime
import tqdm
import random
import json

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
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
    parser.add_argument('--commitment-cost', type=float, default=0.25,
                        help='Commitment cost for VQ loss')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
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


def train_one_epoch(model, dataloader, optimizer, device, pad_id, args, epoch, 
                    refresh_codes=False, refresh_interval=500, buffer_size=10000):
    """
    Train for one epoch
    
    Args:
        refresh_codes: Enable dead code refresh mechanism
        refresh_interval: Refresh dead codes every N batches
        buffer_size: Size of encoder output buffer for code refresh
    """
    model.train()
    total_loss, total_recon, total_vq, n_tokens = 0, 0, 0, 0
    
    # Buffer for collecting encoder outputs (for dead code refresh)
    z_buffer = [] if refresh_codes else None
    max_buffer_size = buffer_size

    for batch_idx, (batch_tokens, batch_lengths) in enumerate(dataloader):
        batch_tokens = batch_tokens.to(device)
        batch_lengths = batch_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, loss_vq, codes = model(batch_tokens)
        B, L, V = logits.shape
        
        # Collect encoder outputs for dead code refresh
        if refresh_codes and z_buffer is not None:
            # Get encoder outputs (before quantization)
            with torch.no_grad():
                # Access encoder through DataParallel if needed
                if hasattr(model, 'module'):
                    z_e = model.module.encoder(batch_tokens)
                else:
                    z_e = model.encoder(batch_tokens)
                # Flatten to (B*L, D) and store
                z_buffer.append(z_e.reshape(-1, z_e.size(-1)).cpu())
                # Keep buffer size manageable
                if len(z_buffer) > max_buffer_size // B:
                    z_buffer.pop(0)

        # Prevent model from predicting PAD tokens
        logits[:, :, pad_id] = -1e9

        # Create mask based on sequence lengths
        valid_mask = torch.arange(L, device=device)[None, :] < batch_lengths[:, None]
        valid_mask = valid_mask & (batch_tokens != pad_id)

        if valid_mask.sum() == 0:
            continue

        # ---- Step 3: Balanced CE loss over valid tokens ----
        logits_flat = logits[valid_mask]           # (N_valid, V)
        targets_flat = batch_tokens[valid_mask]     # (N_valid,)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
        n_valid = valid_mask.sum().clamp_min(1)
        recon_loss = ce / n_valid                   # normalize by token count
        
        # Handle DataParallel: loss_vq may be a vector if using multiple GPUs
        if loss_vq.dim() > 0:
            loss_vq = loss_vq.mean()
        
        loss = args.recon_loss_weight * recon_loss + loss_vq

        loss.backward()
        optimizer.step()

        # Update statistics
        n_valid_count = n_valid.item()
        total_loss += loss.item() * n_valid_count
        total_recon += recon_loss.item() * n_valid_count
        total_vq += loss_vq.item() * n_valid_count
        n_tokens += n_valid_count

        # Log to wandb periodically
        if not args.no_wandb and batch_idx % args.log_freq == 0:
            # Handle potential tensor values for logging
            batch_loss = loss.item() if loss.dim() == 0 else loss.mean().item()
            batch_recon = recon_loss.item() if recon_loss.dim() == 0 else recon_loss.mean().item()
            batch_vq = loss_vq.item() if loss_vq.dim() == 0 else loss_vq.mean().item()
            
            wandb.log({
                "batch_loss": batch_loss,
                "batch_recon_loss": batch_recon,
                "batch_vq_loss": batch_vq,
                "global_step": epoch * len(dataloader) + batch_idx
            })
        
        # Refresh dead codes periodically
        if refresh_codes and batch_idx % refresh_interval == 0 and len(z_buffer) > 0:
            z_concat = torch.cat(z_buffer, dim=0).to(device)
            # Access VQ layer through DataParallel if needed
            if hasattr(model, 'module'):
                refresh_dead_codes(model.module.vq, z_concat, min_count=args.refresh_min_count)
            else:
                refresh_dead_codes(model.vq, z_concat, min_count=args.refresh_min_count)

        # Free up memory
        del logits, loss_vq, codes, loss
        torch.cuda.empty_cache()

    return {
        "loss": total_loss / n_tokens,
        "recon": total_recon / n_tokens,
        "vq": total_vq / n_tokens
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
    
    # Create dataset and dataloader
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=True)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        try:
            stats = train_one_epoch(
                model, dataloader, optimizer, device, PAD_ID, args, epoch,
                refresh_codes=args.refresh_codes,
                refresh_interval=args.refresh_interval,
                buffer_size=args.refresh_buffer_size
            )
            
            # Log metrics
            if not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": stats['loss'],
                    "reconstruction_loss": stats['recon'],
                    "vq_loss": stats['vq']
                })
            
            print(f"Epoch {epoch}/{args.epochs}: loss {stats['loss']:.6f} recon {stats['recon']:.6f} vq {stats['vq']:.6f}")
            
            # Save checkpoint
            if epoch % args.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stats': stats,
                    'args': vars(args)
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                
                # Log checkpoint as artifact
                if not args.no_wandb:
                    artifact = wandb.Artifact(f'model-checkpoint-{epoch}', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
        
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            raise
    
    # Final evaluation
    print("\n" + "="*50)
    print("Training complete! Running final evaluation...")
    print("="*50)
    evaluate_reconstruction(model, dataloader, tokenizer, device, output_dir, args)
    
    # Clean up wandb
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()






# #* TODO: implement sequence shuffling on original dataset, 
# #* TODO:  run original sequences- cleaned_reads on k2minusb, 
# # TODO:  do contrastive learning to reconstruct shuffled sequences, 
# # TODO:  test contrastive sequences on kraken


#* 4 models: VQVAE(base Vector Quantizer), VQVAE(EMA Vector Quantizer), Masked VQVAE, Contrastive VQVAE.

# TODO: Accuracy on 4 approaches
# TODO: Latent space cluster analysis



""""
- Implement evaluation VQVAE
- Implement masked VQVAE, run evaluation on it as well
- Implement latent space visualization for various grouping in latent space.
- Run contrastive on covid dataset
- Compare results on three methods
- Literature writing
- Poster design
"""
