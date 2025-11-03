#!/usr/bin/env python3
"""
VQ-VAE Evaluation Script for Viral Genome Sequences

Usage:
    python evaluate.py --checkpoint-path ./checkpoints/checkpoint_epoch_100.pt --data-path /path/to/data
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
import random
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
from src.data import KmerTokenizer, FastqKmerDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE on viral genome sequences')
    
    # Data arguments
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num-eval-batches', type=int, default=100,
                        help='Number of batches to evaluate')
    parser.add_argument('--num-samples-per-batch', type=int, default=3,
                        help='Number of samples per batch to save')
    
    # GPU arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    return args


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


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


def compute_reconstruction_metrics(model, dataloader, tokenizer, device, num_batches=None):
    """Compute reconstruction accuracy metrics"""
    model.eval()
    
    total_exact_matches = 0
    total_sequences = 0
    match_rates = []
    
    batches_to_process = list(dataloader)[:num_batches] if num_batches else dataloader
    
    print(f"Computing reconstruction metrics on {len(batches_to_process)} batches...")
    
    for batch_tokens, batch_lengths in tqdm(batches_to_process):
        batch_tokens = batch_tokens.to(device)
        batch_lengths = batch_lengths.to(device)
        
        with torch.no_grad():
            logits, _, _ = model(batch_tokens)
            preds = logits.argmax(dim=-1)
        
        for i in range(batch_tokens.size(0)):
            L_true = int(batch_lengths[i].item())
            input_ids = batch_tokens[i, :L_true].cpu().numpy()
            pred_ids = preds[i, :L_true].cpu().numpy()
            
            # Compute exact match
            if np.array_equal(input_ids, pred_ids):
                total_exact_matches += 1
            
            # Compute per-token accuracy
            match_rate = (input_ids == pred_ids).mean()
            match_rates.append(match_rate)
            
            total_sequences += 1
    
    metrics = {
        'exact_match_rate': total_exact_matches / total_sequences,
        'mean_token_accuracy': np.mean(match_rates),
        'median_token_accuracy': np.median(match_rates),
        'std_token_accuracy': np.std(match_rates),
        'total_sequences': total_sequences
    }
    
    return metrics


def evaluate_codebook_usage(model, dataloader, device, num_batches=None):
    """Evaluate codebook usage statistics"""
    model.eval()
    
    num_codes = model.vq.num_codes if not isinstance(model, torch.nn.DataParallel) else model.module.vq.num_codes
    code_counts = torch.zeros(num_codes, dtype=torch.long)
    
    batches_to_process = list(dataloader)[:num_batches] if num_batches else dataloader
    
    print(f"Computing codebook usage on {len(batches_to_process)} batches...")
    
    for batch_tokens, batch_lengths in tqdm(batches_to_process):
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            _, _, codes = model(batch_tokens)
            # Count code usage
            unique_codes, counts = torch.unique(codes.cpu().flatten(), return_counts=True)
            for code, count in zip(unique_codes, counts):
                code_counts[code] += count
    
    # Calculate statistics
    active_codes = (code_counts > 0).sum().item()
    utilization = active_codes / num_codes
    
    usage_stats = {
        'num_codes': num_codes,
        'active_codes': active_codes,
        'codebook_utilization': utilization,
        'mean_usage': code_counts.float().mean().item(),
        'std_usage': code_counts.float().std().item(),
        'min_usage': code_counts.min().item(),
        'max_usage': code_counts.max().item()
    }
    
    return usage_stats, code_counts


def main():
    args = parse_args()
    
    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU {args.gpu_id}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        vocab_size = model_args['vocab_size']
        num_codes = model_args['num_codes']
        code_dim = model_args['code_dim']
        embed_dim = model_args['embed_dim']
        hidden_dim = model_args['hidden_dim']
        commitment_cost = model_args['commitment_cost']
    else:
        # Default values
        vocab_size = 4097
        num_codes = 512
        code_dim = 64
        embed_dim = 128
        hidden_dim = 256
        commitment_cost = 0.25
    
    # Initialize model
    PAD_ID = vocab_size - 1
    model = VQVAE(
        vocab_size,
        PAD_ID,
        num_codes=num_codes,
        code_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        commitment_cost=commitment_cost
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel models: remove 'module.' prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dataset and dataloader
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=True)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 1. Compute reconstruction metrics
    print("\n" + "="*50)
    print("Computing reconstruction metrics...")
    print("="*50)
    recon_metrics = compute_reconstruction_metrics(
        model, dataloader, tokenizer, device, num_batches=args.num_eval_batches
    )
    
    print("\nReconstruction Metrics:")
    for key, value in recon_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 2. Evaluate codebook usage
    print("\n" + "="*50)
    print("Evaluating codebook usage...")
    print("="*50)
    usage_stats, code_counts = evaluate_codebook_usage(
        model, dataloader, device, num_batches=args.num_eval_batches
    )
    
    print("\nCodebook Usage Statistics:")
    for key, value in usage_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 3. Generate reconstruction examples
    print("\n" + "="*50)
    print("Generating reconstruction examples...")
    print("="*50)
    reconstruction_file = os.path.join(args.output_dir, 'sequence_reconstructions.txt')
    
    all_batches = list(dataloader)
    num_batches = min(args.num_eval_batches, len(all_batches))
    random_batches = random.sample(all_batches, num_batches)
    
    with open(reconstruction_file, 'w') as f:
        for batch_idx, (batch_tokens, batch_lengths) in enumerate(tqdm(random_batches)):
            examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device)
            
            batch_size = len(examples)
            sample_indices = random.sample(range(batch_size), min(args.num_samples_per_batch, batch_size))
            
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
    
    # 4. Save all results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    results = {
        'checkpoint_path': args.checkpoint_path,
        'data_path': args.data_path,
        'reconstruction_metrics': recon_metrics,
        'codebook_usage': usage_stats,
        'model_config': {
            'vocab_size': vocab_size,
            'num_codes': num_codes,
            'code_dim': code_dim,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'commitment_cost': commitment_cost
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
