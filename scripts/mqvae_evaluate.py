#!/usr/bin/env python3
"""
Masked VQ-VAE Evaluation Script for Viral Genome Sequences

This script evaluates a Masked VQ-VAE model by:
1. Computing masked token prediction accuracy
2. Evaluating perplexity on masked tokens
3. Analyzing codebook usage
4. Generating masked reconstruction examples

Usage:
    python scripts/mqvae_evaluate.py \
      --checkpoint-path outputs/mqvae_train1/checkpoint_epoch_50.pt \
      --data-path cleaned_reads.fastq \
      --output-dir ./mqvae_evaluation_results
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from Bio import SeqIO
import numpy as np
import random
from tqdm import tqdm
import json
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
from src.data import KmerTokenizer, FastqKmerDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Masked VQ-VAE on viral genome sequences')
    
    # Data arguments
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./mqvae_evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    parser.add_argument('--mask-token', type=str, default='<MASK>',
                        help='Special mask token')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num-eval-batches', type=int, default=100,
                        help='Number of batches to evaluate')
    parser.add_argument('--num-samples-per-batch', type=int, default=3,
                        help='Number of samples per batch to save')
    parser.add_argument('--mask-prob', type=float, default=0.2,
                        help='Probability of masking each token')
    parser.add_argument('--use-validation-split', action='store_true',
                        help='Evaluate on validation split instead of full dataset')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio if using validation split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_masked_reconstruction_metrics(model, dataloader, tokenizer, device, 
                                          pad_id, mask_id, mask_prob, num_batches=None):
    """
    Compute reconstruction accuracy metrics on masked tokens
    
    Returns:
        dict: Metrics including accuracy, perplexity, confusion stats
    """
    model.eval()
    
    total_correct = 0
    total_masked = 0
    all_losses = []
    prediction_confidences = []
    top5_accuracies = []
    
    # Track prediction errors
    error_counts = Counter()
    correct_counts = Counter()
    
    batches_to_process = list(dataloader)[:num_batches] if num_batches else dataloader
    
    print(f"Computing masked reconstruction metrics on {len(batches_to_process)} batches...")
    
    with torch.no_grad():
        for batch_tokens, batch_lengths in tqdm(batches_to_process):
            batch_tokens = batch_tokens.to(device)
            batch_lengths = batch_lengths.to(device)
            
            # Apply random masking
            masked_inputs, mask_positions = apply_random_mask(
                batch_tokens, pad_id, mask_id, mask_prob
            )
            
            # Forward pass
            logits, _, _ = model(masked_inputs)
            B, L, V = logits.shape
            
            # Prevent PAD prediction
            logits[:, :, pad_id] = -1e9
            
            # Get predictions
            preds = logits.argmax(dim=-1)
            
            # Compute metrics only on masked positions
            valid_mask = mask_positions & (batch_tokens != pad_id)
            
            if valid_mask.sum() == 0:
                continue
            
            # Extract masked tokens and predictions
            logits_masked = logits[valid_mask]      # (N_masked, V)
            targets_masked = batch_tokens[valid_mask]  # (N_masked,)
            preds_masked = preds[valid_mask]        # (N_masked,)
            
            # Accuracy
            correct = (preds_masked == targets_masked)
            total_correct += correct.sum().item()
            total_masked += targets_masked.size(0)
            
            # Loss (for perplexity calculation)
            loss = F.cross_entropy(logits_masked, targets_masked, reduction='none')
            all_losses.extend(loss.cpu().numpy().tolist())
            
            # Prediction confidence (probability assigned to predicted token)
            probs = F.softmax(logits_masked, dim=-1)
            pred_probs = probs[torch.arange(preds_masked.size(0)), preds_masked]
            prediction_confidences.extend(pred_probs.cpu().numpy().tolist())
            
            # Top-5 accuracy
            top5_preds = logits_masked.topk(5, dim=-1)[1]
            top5_correct = (top5_preds == targets_masked.unsqueeze(-1)).any(dim=-1)
            top5_accuracies.extend(top5_correct.cpu().numpy().tolist())
            
            # Track error patterns
            for pred, target in zip(preds_masked.cpu().numpy(), targets_masked.cpu().numpy()):
                if pred == target:
                    correct_counts[target] += 1
                else:
                    error_counts[(target, pred)] += 1
    
    # Calculate metrics
    accuracy = total_correct / total_masked if total_masked > 0 else 0.0
    perplexity = np.exp(np.mean(all_losses)) if all_losses else float('inf')
    top5_accuracy = np.mean(top5_accuracies) if top5_accuracies else 0.0
    mean_confidence = np.mean(prediction_confidences) if prediction_confidences else 0.0
    
    # Get most common errors
    most_common_errors = error_counts.most_common(10)
    
    metrics = {
        'accuracy': accuracy,
        'perplexity': perplexity,
        'top5_accuracy': top5_accuracy,
        'mean_prediction_confidence': mean_confidence,
        'std_prediction_confidence': np.std(prediction_confidences) if prediction_confidences else 0.0,
        'total_masked_tokens': total_masked,
        'total_correct_predictions': total_correct,
        'most_common_errors': [
            {'true_token': int(true_tok), 'predicted_token': int(pred_tok), 'count': count}
            for (true_tok, pred_tok), count in most_common_errors
        ]
    }
    
    return metrics


def evaluate_codebook_usage(model, dataloader, device, pad_id, mask_id, 
                            mask_prob, num_batches=None):
    """
    Evaluate codebook usage statistics on masked inputs
    """
    model.eval()
    
    # Get number of codes
    if isinstance(model, torch.nn.DataParallel):
        num_codes = model.module.vq.num_codes
    else:
        num_codes = model.vq.num_codes
    
    code_counts = torch.zeros(num_codes, dtype=torch.long)
    
    batches_to_process = list(dataloader)[:num_batches] if num_batches else dataloader
    
    print(f"Computing codebook usage on {len(batches_to_process)} batches...")
    
    with torch.no_grad():
        for batch_tokens, batch_lengths in tqdm(batches_to_process):
            batch_tokens = batch_tokens.to(device)
            
            # Apply random masking
            masked_inputs, _ = apply_random_mask(batch_tokens, pad_id, mask_id, mask_prob)
            
            # Get codes
            _, _, codes = model(masked_inputs)
            
            # Count code usage
            unique_codes, counts = torch.unique(codes.cpu().flatten(), return_counts=True)
            for code, count in zip(unique_codes, counts):
                code_counts[code] += count
    
    # Calculate statistics
    active_codes = (code_counts > 0).sum().item()
    utilization = active_codes / num_codes
    
    # Entropy of code distribution (measure of uniformity)
    code_probs = code_counts.float() / code_counts.sum()
    code_probs = code_probs[code_probs > 0]  # Remove zeros for log
    entropy = -(code_probs * torch.log(code_probs)).sum().item()
    max_entropy = np.log(num_codes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    usage_stats = {
        'num_codes': num_codes,
        'active_codes': active_codes,
        'codebook_utilization': utilization,
        'code_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'mean_usage': code_counts.float().mean().item(),
        'std_usage': code_counts.float().std().item(),
        'min_usage': code_counts.min().item(),
        'max_usage': code_counts.max().item()
    }
    
    return usage_stats, code_counts


def visualize_masked_reconstruction(model, batch_tokens, batch_lengths, tokenizer, 
                                   device, pad_id, mask_id, mask_prob, num_examples=5):
    """
    Generate visualization of masked reconstruction examples
    
    Returns:
        str: Formatted text with examples
        list: List of example dictionaries for JSON output
    """
    model.eval()
    batch_tokens = batch_tokens.to(device)
    batch_lengths = batch_lengths.to(device)
    
    # Apply random masking
    masked_inputs, mask_positions = apply_random_mask(batch_tokens, pad_id, mask_id, mask_prob)
    
    with torch.no_grad():
        logits, _, _ = model(masked_inputs)
        preds = logits.argmax(dim=-1).cpu()
        
        # Get prediction probabilities for confidence scores
        probs = F.softmax(logits, dim=-1)
    
    output_lines = []
    output_lines.append("\n" + "="*80)
    output_lines.append("MASKED RECONSTRUCTION EXAMPLES")
    output_lines.append("="*80)
    
    examples_data = []
    
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
        
        # Decode to sequences
        true_seq = tokenizer.decode_tokens(true_ids, remove_pad=True, reconstruct=True)
        pred_seq = tokenizer.decode_tokens(pred_ids, remove_pad=True, reconstruct=True)
        
        # Find masked positions
        masked_pos = [j for j, m in enumerate(mask_positions[i, :L_true].cpu().tolist()) if m]
        
        # Calculate accuracy on masked positions
        if masked_pos:
            correct = sum(1 for j in masked_pos if pred_ids[j] == true_ids[j])
            accuracy = correct / len(masked_pos) * 100
            
            # Get average confidence on masked positions
            masked_confidences = []
            for j in masked_pos:
                pred_token = pred_ids[j]
                confidence = probs[i, j, pred_token].item()
                masked_confidences.append(confidence)
            avg_confidence = np.mean(masked_confidences) * 100
        else:
            accuracy = 0.0
            avg_confidence = 0.0
        
        # Calculate sequence-level match rate
        seq_match_rate = sum(a == b for a, b in zip(true_seq[:120], pred_seq[:120])) / min(len(true_seq[:120]), 120) * 100
        
        output_lines.append(f"\n{'─'*80}")
        output_lines.append(f"Example {i+1}")
        output_lines.append(f"{'─'*80}")
        output_lines.append(f"Original sequence (first 120 bp):")
        output_lines.append(f"  {true_seq[:120]}")
        output_lines.append(f"Reconstructed sequence (first 120 bp):")
        output_lines.append(f"  {pred_seq[:120]}")
        output_lines.append(f"")
        output_lines.append(f"Original k-mers (first 15):  {true_kmers[:15]}")
        output_lines.append(f"Masked k-mers (first 15):    {mask_kmers[:15]}")
        output_lines.append(f"Predicted k-mers (first 15): {pred_kmers[:15]}")
        output_lines.append(f"")
        output_lines.append(f"Masked positions (first 20): {masked_pos[:20]}")
        output_lines.append(f"Total masked tokens: {len(masked_pos)}")
        output_lines.append(f"Accuracy on masked tokens: {accuracy:.1f}%")
        output_lines.append(f"Avg confidence on masked tokens: {avg_confidence:.1f}%")
        output_lines.append(f"Sequence match rate (first 120 bp): {seq_match_rate:.1f}%")
        
        # Store for JSON
        examples_data.append({
            'example_id': i + 1,
            'original_sequence': true_seq[:120],
            'reconstructed_sequence': pred_seq[:120],
            'masked_positions': masked_pos[:20],
            'num_masked_tokens': len(masked_pos),
            'masked_accuracy': float(accuracy),
            'avg_confidence': float(avg_confidence),
            'sequence_match_rate': float(seq_match_rate)
        })
    
    output_lines.append("="*80 + "\n")
    
    return "\n".join(output_lines), examples_data


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
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
        vocab_size = model_args.get('vocab_size', 4097)
        num_codes = model_args.get('num_codes', 512)
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        commitment_cost = model_args.get('commitment_cost', 0.1)
        mask_prob = model_args.get('mask_prob', args.mask_prob)
    else:
        # Default values
        vocab_size = 4098  # Including MASK token
        num_codes = 512
        code_dim = 128
        embed_dim = 128
        hidden_dim = 256
        commitment_cost = 0.1
        mask_prob = args.mask_prob
    
    print(f"Model configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num codes: {num_codes}")
    print(f"  Code dim: {code_dim}")
    print(f"  Mask probability: {mask_prob}")
    
    # Create tokenizer and add mask token
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=True)
    tokenizer.add_special_token(args.mask_token)
    mask_id = tokenizer.mask_id
    PAD_ID = tokenizer.pad_id
    
    print(f"Tokenizer:")
    print(f"  Vocabulary size: {len(tokenizer.stoi)}")
    print(f"  PAD ID: {PAD_ID}")
    print(f"  MASK ID: {mask_id}")
    
    # Initialize model
    model = VQVAE(
        len(tokenizer.stoi),
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
    
    print("✓ Model loaded successfully")
    
    # Create dataset
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Optionally split into validation set
    if args.use_validation_split:
        dataset_size = len(dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(args.seed)
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"Using validation split: {val_size} sequences")
    else:
        eval_dataset = dataset
        print(f"Using full dataset: {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 1. Compute masked reconstruction metrics
    print("\n" + "="*80)
    print("COMPUTING MASKED RECONSTRUCTION METRICS")
    print("="*80)
    
    recon_metrics = compute_masked_reconstruction_metrics(
        model, dataloader, tokenizer, device, PAD_ID, mask_id, 
        mask_prob, num_batches=args.num_eval_batches
    )
    
    print("\nMasked Reconstruction Metrics:")
    print(f"  Accuracy on masked tokens: {recon_metrics['accuracy']:.4f} ({recon_metrics['accuracy']*100:.2f}%)")
    print(f"  Top-5 accuracy: {recon_metrics['top5_accuracy']:.4f} ({recon_metrics['top5_accuracy']*100:.2f}%)")
    print(f"  Perplexity: {recon_metrics['perplexity']:.4f}")
    print(f"  Mean prediction confidence: {recon_metrics['mean_prediction_confidence']:.4f}")
    print(f"  Total masked tokens evaluated: {recon_metrics['total_masked_tokens']}")
    print(f"  Total correct predictions: {recon_metrics['total_correct_predictions']}")
    
    # 2. Evaluate codebook usage
    print("\n" + "="*80)
    print("EVALUATING CODEBOOK USAGE")
    print("="*80)
    
    usage_stats, code_counts = evaluate_codebook_usage(
        model, dataloader, device, PAD_ID, mask_id, mask_prob,
        num_batches=args.num_eval_batches
    )
    
    print("\nCodebook Usage Statistics:")
    print(f"  Total codes: {usage_stats['num_codes']}")
    print(f"  Active codes: {usage_stats['active_codes']}")
    print(f"  Codebook utilization: {usage_stats['codebook_utilization']:.4f} ({usage_stats['codebook_utilization']*100:.2f}%)")
    print(f"  Code entropy: {usage_stats['code_entropy']:.4f}")
    print(f"  Normalized entropy: {usage_stats['normalized_entropy']:.4f}")
    print(f"  Mean usage: {usage_stats['mean_usage']:.2f}")
    print(f"  Std usage: {usage_stats['std_usage']:.2f}")
    
    # 3. Generate reconstruction examples
    print("\n" + "="*80)
    print("GENERATING RECONSTRUCTION EXAMPLES")
    print("="*80)
    
    reconstruction_file = os.path.join(args.output_dir, 'masked_reconstruction_examples.txt')
    
    all_batches = list(dataloader)
    num_batches = min(args.num_eval_batches, len(all_batches))
    random_batches = random.sample(all_batches, num_batches)
    
    all_examples = []
    
    with open(reconstruction_file, 'w') as f:
        f.write(f"Masked VQ-VAE Reconstruction Examples\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Mask probability: {mask_prob}\n")
        f.write(f"{'='*80}\n\n")
        
        for batch_idx, (batch_tokens, batch_lengths) in enumerate(tqdm(random_batches, desc="Generating examples")):
            viz_text, examples_data = visualize_masked_reconstruction(
                model, batch_tokens, batch_lengths, tokenizer,
                device, PAD_ID, mask_id, mask_prob,
                num_examples=args.num_samples_per_batch
            )
            
            f.write(f"\nBatch {batch_idx + 1}\n")
            f.write(viz_text)
            f.write("\n")
            
            all_examples.extend(examples_data)
            
            # Print first few examples to console
            if batch_idx == 0:
                print(viz_text)
    
    print(f"\n✓ Reconstruction examples saved to: {reconstruction_file}")
    
    # 4. Save all results
    results_file = os.path.join(args.output_dir, 'mqvae_evaluation_results.json')
    results = {
        'checkpoint_path': args.checkpoint_path,
        'data_path': args.data_path,
        'evaluation_config': {
            'mask_probability': mask_prob,
            'num_batches_evaluated': args.num_eval_batches,
            'batch_size': args.batch_size,
            'use_validation_split': args.use_validation_split,
            'seed': args.seed
        },
        'masked_reconstruction_metrics': recon_metrics,
        'codebook_usage': usage_stats,
        'model_config': {
            'vocab_size': len(tokenizer.stoi),
            'num_codes': num_codes,
            'code_dim': code_dim,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'commitment_cost': commitment_cost
        },
        'example_reconstructions': all_examples[:20]  # Save first 20 examples
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Evaluation results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Masked Token Accuracy: {recon_metrics['accuracy']*100:.2f}%")
    print(f"Perplexity: {recon_metrics['perplexity']:.4f}")
    print(f"Codebook Utilization: {usage_stats['codebook_utilization']*100:.2f}%")
    print(f"Results directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
