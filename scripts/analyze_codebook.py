#!/usr/bin/env python3
"""
Codebook Analysis Visualization

This script creates detailed visualizations of VQ-VAE codebook usage:
1. Usage frequency distribution (log scale)
2. Heatmap showing code activation patterns
3. Example k-mers from top-5 most used codes

Usage:
    python scripts/analyze_codebook.py \
        --checkpoint-path experiments/1_standard_vqvae/checkpoints/best_model.pt \
        --data-path /home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq\
        --output-dir experiments/1_standard_vqvae/codebook_analysis \
        --num-samples 5000
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE
from src.data import KmerTokenizer, FastqKmerDataset


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[1] for item in batch])
    return tokens, lengths


def extract_code_usage(model, dataloader, device, num_codes=512):
    """
    Extract codebook usage statistics
    
    Returns:
        code_counts: Array of usage counts per code
        code_examples: Dict mapping code_id -> list of (sequence_idx, position, kmer)
        code_embeddings: Codebook embeddings [num_codes, code_dim]
    """
    model.eval()
    
    code_counts = np.zeros(num_codes, dtype=np.int64)
    code_examples = defaultdict(list)
    
    print("Extracting code usage from data...")
    
    with torch.no_grad():
        for batch_idx, (batch_tokens, batch_lengths) in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx * len(batch_tokens)} sequences...")
            
            batch_tokens = batch_tokens.to(device)
            
            # Forward pass to get codes
            _, _, codes = model(batch_tokens)  # codes: [B, L]
            codes = codes.cpu().numpy()
            batch_tokens_np = batch_tokens.cpu().numpy()
            
            # Count code usage
            for code_id in range(num_codes):
                count = np.sum(codes == code_id)
                code_counts[code_id] += count
            
            # Store examples for top codes (do this for first few batches)
            if batch_idx < 10:  # Limit to avoid memory issues
                for seq_idx in range(len(codes)):
                    for pos in range(codes.shape[1]):
                        code_id = codes[seq_idx, pos]
                        token_id = batch_tokens_np[seq_idx, pos]
                        
                        # Store up to 20 examples per code
                        if len(code_examples[code_id]) < 20:
                            code_examples[code_id].append({
                                'seq_idx': batch_idx * len(batch_tokens) + seq_idx,
                                'position': pos,
                                'token_id': token_id
                            })
    
    print(f"✓ Analyzed {len(dataloader) * dataloader.batch_size} sequences")
    
    # Get codebook embeddings
    code_embeddings = model.vq.embedding.weight.data.cpu().numpy()
    
    return code_counts, code_examples, code_embeddings


def plot_usage_frequency(code_counts, output_path, num_codes=512):
    """Plot usage frequency distribution on log scale"""
    print("\nCreating usage frequency plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort codes by usage
    sorted_counts = np.sort(code_counts)[::-1]
    used_codes = np.sum(code_counts > 0)
    utilization = (used_codes / num_codes) * 100
    
    # Top plot: Linear scale
    ax = axes[0]
    ax.bar(range(num_codes), sorted_counts, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax.set_xlabel('Code Rank (sorted by usage)', fontsize=12)
    ax.set_ylabel('Usage Count', fontsize=12)
    ax.set_title(f'Codebook Usage Frequency Distribution (Linear Scale)\n'
                 f'Utilization: {utilization:.2f}% ({used_codes}/{num_codes} codes used)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = (
        f'Total usages: {code_counts.sum():,}\n'
        f'Most used code: {code_counts.max():,} times\n'
        f'Mean usage: {code_counts.mean():.1f}\n'
        f'Median usage: {np.median(code_counts):.1f}\n'
        f'Unused codes: {num_codes - used_codes}'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Bottom plot: Log scale
    ax = axes[1]
    # Add small value to avoid log(0)
    log_counts = np.log10(sorted_counts + 1)
    ax.bar(range(num_codes), log_counts, color='darkgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
    ax.set_xlabel('Code Rank (sorted by usage)', fontsize=12)
    ax.set_ylabel('Log10(Usage Count + 1)', fontsize=12)
    ax.set_title('Codebook Usage Frequency Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mark frequently used vs rarely used
    threshold_idx = int(num_codes * 0.2)  # Top 20%
    ax.axvline(threshold_idx, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Top 20% codes ({threshold_idx} codes)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return used_codes, utilization


def plot_code_activation_heatmap(code_counts, code_embeddings, output_path, top_k=100):
    """Plot heatmap showing code activation patterns"""
    print("\nCreating code activation heatmap...")
    
    # Get top-k most used codes
    top_indices = np.argsort(code_counts)[::-1][:top_k]
    top_counts = code_counts[top_indices]
    top_embeddings = code_embeddings[top_indices]
    
    # Normalize embeddings for visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    top_embeddings_normalized = scaler.fit_transform(top_embeddings)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                          hspace=0.05, wspace=0.05)
    
    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 0])
    im = ax_main.imshow(top_embeddings_normalized, aspect='auto', cmap='RdBu_r', 
                        vmin=-3, vmax=3, interpolation='nearest')
    ax_main.set_xlabel('Embedding Dimension', fontsize=12)
    ax_main.set_ylabel('Code ID (Top {} by usage)'.format(top_k), fontsize=12)
    ax_main.set_yticks(range(0, top_k, max(1, top_k // 10)))
    ax_main.set_yticklabels([f'{top_indices[i]}' for i in range(0, top_k, max(1, top_k // 10))])
    
    # Top bar chart (usage counts)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.bar(range(top_embeddings.shape[1]), 
               np.abs(top_embeddings_normalized).mean(axis=0),
               color='steelblue', alpha=0.7)
    ax_top.set_ylabel('Mean |Activity|', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis='y')
    ax_top.set_title(f'Code Activation Patterns (Top {top_k} Most Used Codes)\n'
                     'Heatmap shows normalized embedding values',
                     fontsize=13, fontweight='bold', pad=10)
    
    # Right bar chart (usage frequency)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.barh(range(top_k), top_counts, color='darkgreen', alpha=0.7)
    ax_right.set_xlabel('Usage\nCount', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.invert_xaxis()
    ax_right.grid(True, alpha=0.3, axis='x')
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized Embedding Value', fontsize=10)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def decode_token_to_kmer(token_id, tokenizer):
    """Convert token ID back to k-mer string"""
    # Get the k-mer from tokenizer's reverse mapping
    if token_id == tokenizer.pad_id:
        return '<PAD>'
    elif token_id == tokenizer.unk_id:
        return '<UNK>'
    elif token_id == tokenizer.mask_id:
        return '<MASK>'
    
    # Find the k-mer for this token
    for kmer, tid in tokenizer.stoi.items():
        if tid == token_id:
            return kmer
    
    return '<UNKNOWN>'


def plot_top_codes_examples(code_counts, code_examples, tokenizer, output_path, top_n=5):
    """Show example k-mers from top-N most used codes"""
    print(f"\nCreating top-{top_n} codes examples...")
    
    # Get top-N codes
    top_indices = np.argsort(code_counts)[::-1][:top_n]
    
    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n))
    if top_n == 1:
        axes = [axes]
    
    for idx, (ax, code_id) in enumerate(zip(axes, top_indices)):
        examples = code_examples[code_id]
        usage_count = code_counts[code_id]
        
        # Get unique k-mers from examples
        kmers = []
        for ex in examples[:15]:  # Show up to 15 examples
            token_id = ex['token_id']
            kmer = decode_token_to_kmer(token_id, tokenizer)
            kmers.append(kmer)
        
        # Count k-mer frequencies
        from collections import Counter
        kmer_counts = Counter(kmers)
        most_common = kmer_counts.most_common(10)
        
        # Plot
        if most_common:
            labels, counts = zip(*most_common)
            y_pos = np.arange(len(labels))
            
            bars = ax.barh(y_pos, counts, color='coral', alpha=0.7, edgecolor='darkred')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=11, family='monospace')
            ax.set_xlabel('Frequency in Sample', fontsize=11)
            ax.set_title(f'Code {code_id} (Rank #{idx+1}) - Total Usage: {usage_count:,}\n'
                        f'Top K-mers from {len(examples)} sampled positions',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f' {count}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No examples available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def save_codebook_statistics(code_counts, code_embeddings, output_path, num_codes=512):
    """Save detailed statistics to text file"""
    print("\nSaving codebook statistics...")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VQ-VAE CODEBOOK ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total codes: {num_codes}\n")
        f.write(f"Used codes: {np.sum(code_counts > 0)} ({np.sum(code_counts > 0)/num_codes*100:.2f}%)\n")
        f.write(f"Unused codes: {np.sum(code_counts == 0)} ({np.sum(code_counts == 0)/num_codes*100:.2f}%)\n")
        f.write(f"Total usages: {code_counts.sum():,}\n")
        f.write(f"Mean usage per code: {code_counts.mean():.2f}\n")
        f.write(f"Median usage: {np.median(code_counts):.2f}\n")
        f.write(f"Std deviation: {code_counts.std():.2f}\n")
        f.write(f"Max usage (single code): {code_counts.max():,}\n")
        f.write(f"Min usage (non-zero): {code_counts[code_counts > 0].min() if np.any(code_counts > 0) else 0}\n")
        
        # Perplexity
        p = code_counts / (code_counts.sum() + 1e-9)
        entropy = -(p * np.log(p + 1e-9)).sum()
        perplexity = np.exp(entropy)
        f.write(f"\nPerplexity: {perplexity:.2f} (max possible: {num_codes})\n")
        
        # Distribution analysis
        f.write("\n" + "="*80 + "\n")
        f.write("USAGE DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        f.write("Usage percentiles:\n")
        for p in percentiles:
            val = np.percentile(code_counts, p)
            f.write(f"  {p}th percentile: {val:.1f}\n")
        
        # Top codes
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 MOST USED CODES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Code ID':<10} {'Usage Count':<15} {'Percentage':<12}\n")
        f.write("-" * 80 + "\n")
        
        top_indices = np.argsort(code_counts)[::-1][:20]
        total_usage = code_counts.sum()
        for rank, code_id in enumerate(top_indices, 1):
            count = code_counts[code_id]
            pct = (count / total_usage) * 100
            f.write(f"{rank:<6} {code_id:<10} {count:<15,} {pct:<12.4f}%\n")
        
        # Concentration analysis
        top_10_pct = code_counts[top_indices[:int(num_codes*0.1)]].sum() / total_usage * 100
        top_20_pct = code_counts[top_indices[:int(num_codes*0.2)]].sum() / total_usage * 100
        top_50_pct = code_counts[top_indices[:int(num_codes*0.5)]].sum() / total_usage * 100
        
        f.write("\n" + "="*80 + "\n")
        f.write("USAGE CONCENTRATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Top 10% codes account for: {top_10_pct:.2f}% of total usage\n")
        f.write(f"Top 20% codes account for: {top_20_pct:.2f}% of total usage\n")
        f.write(f"Top 50% codes account for: {top_50_pct:.2f}% of total usage\n")
        
        # Embedding statistics
        f.write("\n" + "="*80 + "\n")
        f.write("EMBEDDING STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Embedding dimension: {code_embeddings.shape[1]}\n")
        f.write(f"Mean embedding norm: {np.linalg.norm(code_embeddings, axis=1).mean():.4f}\n")
        f.write(f"Std embedding norm: {np.linalg.norm(code_embeddings, axis=1).std():.4f}\n")
        
        # Pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(code_embeddings[:100], metric='euclidean')  # Sample for speed
        f.write(f"Mean pairwise distance (sample): {distances.mean():.4f}\n")
        f.write(f"Min pairwise distance (sample): {distances.min():.4f}\n")
        f.write(f"Max pairwise distance (sample): {distances.max():.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize VQ-VAE codebook usage'
    )
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to FASTQ data file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size (default: 6)')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length (default: 150)')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of sequences to analyze (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--top-codes', type=int, default=5,
                        help='Number of top codes to show examples for (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VQ-VAE CODEBOOK ANALYSIS")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Data: {args.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Samples to analyze: {args.num_samples}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create tokenizer
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=False)
    PAD_ID = tokenizer.pad_id
    VOCAB_SIZE = len(tokenizer.stoi)
    
    print(f"\nTokenizer:")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  PAD ID: {PAD_ID}")
    
    # Load checkpoint
    print(f"\nLoading VQ-VAE checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        num_codes = model_args.get('num_codes', 512)
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        checkpoint_vocab_size = model_args.get('vocab_size', 4097)
        checkpoint_pad_id = model_args.get('pad_id', 4096)
    else:
        num_codes = 512
        code_dim = 128
        embed_dim = 128
        hidden_dim = 256
        checkpoint_vocab_size = 4097
        checkpoint_pad_id = 4096
    
    print(f"  Number of codes: {num_codes}")
    print(f"  Code dimension: {code_dim}")
    
    # Create model
    model = VQVAE(
        checkpoint_vocab_size, checkpoint_pad_id,
        num_codes=num_codes, code_dim=code_dim,
        embed_dim=embed_dim, hidden_dim=hidden_dim
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Sample subset
    if len(dataset) > args.num_samples:
        indices = torch.randperm(len(dataset))[:args.num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"  Analyzing {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Extract code usage
    code_counts, code_examples, code_embeddings = extract_code_usage(
        model, dataloader, device, num_codes
    )
    
    print(f"\n✓ Extracted codebook statistics")
    print(f"  Used codes: {np.sum(code_counts > 0)}/{num_codes}")
    print(f"  Total usages: {code_counts.sum():,}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Usage frequency distribution
    used_codes, utilization = plot_usage_frequency(
        code_counts,
        output_dir / 'usage_frequency_distribution.png',
        num_codes
    )
    
    # 2. Code activation heatmap
    plot_code_activation_heatmap(
        code_counts,
        code_embeddings,
        output_dir / 'code_activation_heatmap.png',
        top_k=min(100, num_codes)
    )
    
    # 3. Top codes examples
    plot_top_codes_examples(
        code_counts,
        code_examples,
        tokenizer,
        output_dir / 'top_codes_kmers.png',
        top_n=args.top_codes
    )
    
    # 4. Save statistics
    save_codebook_statistics(
        code_counts,
        code_embeddings,
        output_dir / 'codebook_statistics.txt',
        num_codes
    )
    
    print("\n" + "="*80)
    print("✓ CODEBOOK ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  • usage_frequency_distribution.png  - Usage counts (linear & log scale)")
    print(f"  • code_activation_heatmap.png       - Embedding activation patterns")
    print(f"  • top_codes_kmers.png               - K-mer examples from top codes")
    print(f"  • codebook_statistics.txt           - Detailed statistics")
    print(f"\nKey Findings:")
    print(f"  • Codebook utilization: {utilization:.2f}%")
    print(f"  • Used codes: {used_codes}/{num_codes}")
    print(f"  • Total usages: {code_counts.sum():,}")
    print(f"  • Most used code: {code_counts.max():,} times (Code {np.argmax(code_counts)})")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
