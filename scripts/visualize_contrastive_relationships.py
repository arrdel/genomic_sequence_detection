#!/usr/bin/env python3
"""
Contrastive Learning Visualization: Embedding Space with Relationships

This script creates a detailed visualization of contrastive learning showing:
1. 2D embedding space with colored clusters
2. Positive pairs connected with green arrows
3. Negative relationships shown with red dashes
4. Legend showing sequence groups

Usage:
    python scripts/visualize_contrastive_relationships.py \
        --checkpoint-path experiments/3_contrastive_vqvae/checkpoints/best_model.pt \
        --vqvae-checkpoint-path experiments/1_standard_vqvae/checkpoints/best_model.pt \
        --data-path cleaned_reads.fastq \
        --output-dir experiments/3_contrastive_vqvae/visualizations \
        --num-samples 500 \
        --num-pairs 50
"""

import os
import sys
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VQVAE, Encoder
from src.data import KmerTokenizer, FastqKmerDataset
from torch.utils.data import DataLoader

# Optional UMAP
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. Will use t-SNE only.")


class ContrastiveHead(nn.Module):
    """Contrastive learning head (64-dim projection)"""
    def __init__(self, encoder, embed_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, tokens):
        z_e = self.encoder(tokens)           # [B, L, D]
        z_mean = z_e.mean(dim=1)             # [B, D] - mean pooling
        z_proj = self.proj(z_mean)           # [B, proj_dim]
        z_norm = F.normalize(z_proj, dim=-1) # unit norm
        return z_norm


class ContrastiveHead128(nn.Module):
    """Contrastive learning head (128-dim projection)"""
    def __init__(self, encoder, embed_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, tokens):
        z_e = self.encoder(tokens)
        z_mean = z_e.mean(dim=1)
        z_proj = self.proj(z_mean)
        z_norm = F.normalize(z_proj, dim=-1)
        return z_norm


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[1] for item in batch])
    return tokens, lengths


def augment_sequence(tokens, pad_id, vocab_size, mask_prob=0.15, drop_prob=0.10):
    """Apply augmentation to create different views"""
    tokens = tokens.clone()
    seq_len = tokens.size(0)
    device = tokens.device
    
    # Masking
    mask = torch.rand(seq_len, device=device) < mask_prob
    mask &= (tokens != pad_id)
    if mask.any():
        tokens[mask] = torch.randint(0, vocab_size, (mask.sum(),), device=device)
    
    # Dropout
    drop_mask = torch.rand(seq_len, device=device) < drop_prob
    drop_mask &= (tokens != pad_id)
    if drop_mask.any():
        tokens[drop_mask] = pad_id
    
    return tokens


def extract_embeddings_and_pairs(model, dataset, device, num_samples=500, num_pairs=50,
                                  pad_id=4096, vocab_size=4097):
    """
    Extract embeddings and create positive/negative pairs
    
    Returns:
        embeddings: [N, D] array of embeddings
        sequences: List of original sequences
        positive_pairs: List of (idx1, idx2) tuples for positive pairs
        negative_pairs: List of (idx1, idx2) tuples for negative pairs
    """
    model.eval()
    
    # Sample sequences
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    embeddings = []
    sequences = []
    
    print(f"Extracting embeddings for {len(indices)} sequences...")
    
    with torch.no_grad():
        for idx in indices:
            tokens, length = dataset[idx]
            tokens = tokens.unsqueeze(0).to(device)  # [1, L]
            
            # Get embedding
            z = model(tokens)  # [1, D]
            embeddings.append(z.cpu().numpy())
            sequences.append(tokens.squeeze(0).cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)  # [N, D]
    
    print(f"Generating {num_pairs} positive pairs with augmentation...")
    
    # Generate positive pairs (same sequence, different augmentations)
    positive_pairs = []
    positive_emb1 = []
    positive_emb2 = []
    
    with torch.no_grad():
        for _ in range(num_pairs):
            # Random sequence
            idx = random.randint(0, len(indices) - 1)
            tokens, _ = dataset[indices[idx]]
            tokens = tokens.to(device)
            
            # Create two augmented views
            aug1 = augment_sequence(tokens, pad_id, vocab_size).unsqueeze(0)
            aug2 = augment_sequence(tokens, pad_id, vocab_size).unsqueeze(0)
            
            z1 = model(aug1)
            z2 = model(aug2)
            
            positive_emb1.append(z1.cpu().numpy())
            positive_emb2.append(z2.cpu().numpy())
            positive_pairs.append((idx, idx))  # Same sequence index
    
    positive_emb1 = np.concatenate(positive_emb1, axis=0)
    positive_emb2 = np.concatenate(positive_emb2, axis=0)
    
    print(f"Selecting {num_pairs} negative pairs (different sequences)...")
    
    # Generate negative pairs (different sequences)
    negative_pairs = []
    for _ in range(num_pairs):
        idx1, idx2 = random.sample(range(len(indices)), 2)
        negative_pairs.append((idx1, idx2))
    
    return embeddings, sequences, positive_pairs, negative_pairs, positive_emb1, positive_emb2


def plot_contrastive_relationships_2d(embeddings, cluster_labels, positive_pairs, 
                                       negative_pairs, positive_emb1, positive_emb2,
                                       output_path, method='tsne', num_display_pairs=30):
    """
    Create visualization with:
    - 2D embedding space with colored clusters
    - Green arrows for positive pairs
    - Red dashes for negative pairs
    - Legend
    """
    print(f"\nCreating {method.upper()} visualization with relationships...")
    
    # Reduce to 2D
    if method == 'umap' and HAS_UMAP:
        print("  Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Also reduce positive pair embeddings
        all_emb = np.vstack([embeddings, positive_emb1, positive_emb2])
        all_2d = reducer.fit_transform(all_emb)
        positive_2d_1 = all_2d[len(embeddings):len(embeddings)+len(positive_emb1)]
        positive_2d_2 = all_2d[len(embeddings)+len(positive_emb1):]
    else:
        print("  Running t-SNE...")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Reduce positive pairs
        all_emb = np.vstack([embeddings, positive_emb1, positive_emb2])
        all_2d = reducer.fit_transform(all_emb)
        positive_2d_1 = all_2d[len(embeddings):len(embeddings)+len(positive_emb1)]
        positive_2d_2 = all_2d[len(embeddings)+len(positive_emb1):]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot clusters as background
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], s=40, alpha=0.5, edgecolors='none',
                  label=f'Cluster {i}')
    
    # Draw negative pairs (red dashed lines) - sample subset
    print(f"  Drawing negative pairs...")
    neg_sample = random.sample(negative_pairs, min(num_display_pairs, len(negative_pairs)))
    for idx1, idx2 in neg_sample:
        x1, y1 = embeddings_2d[idx1]
        x2, y2 = embeddings_2d[idx2]
        ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.2, linewidth=0.8, zorder=1)
    
    # Draw positive pairs (green arrows) - sample subset
    print(f"  Drawing positive pairs...")
    pos_sample = random.sample(range(len(positive_pairs)), min(num_display_pairs, len(positive_pairs)))
    for i in pos_sample:
        x1, y1 = positive_2d_1[i]
        x2, y2 = positive_2d_2[i]
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15,
            color='green', linewidth=1.5, alpha=0.6, zorder=3
        )
        ax.add_patch(arrow)
        
        # Mark the points
        ax.scatter([x1], [y1], c='green', s=60, marker='o', 
                  edgecolors='darkgreen', linewidth=1.5, alpha=0.8, zorder=4)
        ax.scatter([x2], [y2], c='green', s=60, marker='s',
                  edgecolors='darkgreen', linewidth=1.5, alpha=0.8, zorder=4)
    
    # Create legend
    cluster_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}', alpha=0.5) 
                      for i in range(n_clusters)]
    
    relationship_patches = [
        mpatches.Patch(color='green', label='Positive Pairs (same sequence, different augmentations)'),
        mpatches.Patch(color='red', label='Negative Pairs (different sequences)', alpha=0.3)
    ]
    
    marker_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markeredgecolor='darkgreen', markersize=8, label='Augmentation 1'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                  markeredgecolor='darkgreen', markersize=8, label='Augmentation 2')
    ]
    
    # Combine legends
    first_legend = ax.legend(handles=cluster_patches, loc='upper left', 
                            title='Sequence Clusters', fontsize=9, framealpha=0.9)
    ax.add_artist(first_legend)
    
    second_legend = ax.legend(handles=relationship_patches + marker_patches,
                             loc='upper right', title='Contrastive Relationships',
                             fontsize=9, framealpha=0.9)
    
    # Labels and title
    method_name = 'UMAP' if method == 'umap' else 't-SNE'
    ax.set_xlabel(f'{method_name}-1', fontsize=12)
    ax.set_ylabel(f'{method_name}-2', fontsize=12)
    ax.set_title(
        f'Contrastive Learning: {method_name} Embedding Space with Positive/Negative Relationships\n'
        f'Green arrows = Positive pairs (same sequence, different augmentations)\n'
        f'Red dashes = Negative pairs (different sequences)',
        fontsize=13, fontweight='bold', pad=20
    )
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_similarity_heatmap(embeddings, cluster_labels, output_path, max_samples=100):
    """Plot cosine similarity heatmap grouped by clusters"""
    print("\nCreating similarity heatmap...")
    
    # Sample for visualization
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        cluster_labels = cluster_labels[indices]
    
    # Sort by cluster
    sort_idx = np.argsort(cluster_labels)
    embeddings = embeddings[sort_idx]
    cluster_labels = cluster_labels[sort_idx]
    
    # Compute cosine similarity
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarity = embeddings_norm @ embeddings_norm.T
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Add cluster boundaries
    cluster_changes = np.where(np.diff(cluster_labels))[0] + 0.5
    for change in cluster_changes:
        ax.axhline(change, color='black', linewidth=2)
        ax.axvline(change, color='black', linewidth=2)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # Labels
    ax.set_xlabel('Sequence Index (sorted by cluster)', fontsize=12)
    ax.set_ylabel('Sequence Index (sorted by cluster)', fontsize=12)
    ax.set_title('Cosine Similarity Matrix (Grouped by Clusters)\n'
                 'High similarity within clusters, low between clusters',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add cluster labels
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]
        if len(indices) > 0:
            center = (indices[0] + indices[-1]) / 2
            ax.text(-5, center, f'C{cluster_id}', ha='right', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(center, -5, f'C{cluster_id}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize contrastive learning relationships in embedding space'
    )
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to contrastive model checkpoint')
    parser.add_argument('--vqvae-checkpoint-path', type=str, required=True,
                        help='Path to base VQ-VAE checkpoint (for encoder architecture)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to FASTQ data file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size (default: 6)')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length (default: 150)')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of sequences to visualize (default: 500)')
    parser.add_argument('--num-pairs', type=int, default=50,
                        help='Number of positive/negative pairs to generate (default: 50)')
    parser.add_argument('--num-display-pairs', type=int, default=30,
                        help='Number of pairs to display in plot (default: 30)')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of clusters for coloring (default: 10)')
    parser.add_argument('--proj-dim', type=int, default=64,
                        help='Projection dimension (64 or 128, default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CONTRASTIVE LEARNING: RELATIONSHIP VISUALIZATION")
    print("="*80)
    print(f"\nContrastive checkpoint: {args.checkpoint_path}")
    print(f"VQ-VAE checkpoint: {args.vqvae_checkpoint_path}")
    print(f"Data: {args.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Projection dimension: {args.proj_dim}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Positive/negative pairs: {args.num_pairs}")
    print(f"Display pairs: {args.num_display_pairs}")
    print(f"Clusters: {args.num_clusters}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create tokenizer
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=False)
    PAD_ID = tokenizer.pad_id
    VOCAB_SIZE = len(tokenizer.stoi)
    
    print(f"\nTokenizer:")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  PAD ID: {PAD_ID}")
    
    # Load VQ-VAE checkpoint to get config
    print(f"\nLoading VQ-VAE checkpoint for architecture...")
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint_path, map_location=device)
    
    if 'args' in vqvae_checkpoint:
        model_args = vqvae_checkpoint['args']
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        checkpoint_vocab_size = model_args.get('vocab_size', 4097)
        checkpoint_pad_id = model_args.get('pad_id', 4096)
    else:
        code_dim = 128
        embed_dim = 128
        hidden_dim = 256
        checkpoint_vocab_size = 4097
        checkpoint_pad_id = 4096
    
    print(f"  Code dimension: {code_dim}")
    print(f"  Embed dimension: {embed_dim}")
    
    # Create base model and load encoder
    base_model = VQVAE(
        checkpoint_vocab_size, checkpoint_pad_id,
        num_codes=512, code_dim=code_dim,
        embed_dim=embed_dim, hidden_dim=hidden_dim
    ).to(device)
    
    state_dict = vqvae_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)
    
    # Create contrastive model
    print(f"\nLoading contrastive model (proj_dim={args.proj_dim})...")
    if args.proj_dim == 128:
        contrastive_model = ContrastiveHead128(
            encoder=base_model.encoder,
            embed_dim=code_dim,
            proj_dim=args.proj_dim
        ).to(device)
    else:
        contrastive_model = ContrastiveHead(
            encoder=base_model.encoder,
            embed_dim=code_dim,
            proj_dim=args.proj_dim
        ).to(device)
    
    # Load contrastive checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    contrastive_model.load_state_dict(state_dict)
    contrastive_model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    print(f"  Total sequences: {len(dataset)}")
    
    # Extract embeddings and pairs
    (embeddings, sequences, positive_pairs, negative_pairs, 
     positive_emb1, positive_emb2) = extract_embeddings_and_pairs(
        contrastive_model, dataset, device,
        num_samples=args.num_samples,
        num_pairs=args.num_pairs,
        pad_id=PAD_ID,
        vocab_size=VOCAB_SIZE
    )
    
    print(f"\n✓ Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
    print(f"✓ Generated {len(positive_pairs)} positive pairs")
    print(f"✓ Generated {len(negative_pairs)} negative pairs")
    
    # Cluster embeddings
    print(f"\nClustering embeddings (k={args.num_clusters})...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"✓ Cluster distribution: {np.bincount(cluster_labels)}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # t-SNE visualization
    plot_contrastive_relationships_2d(
        embeddings, cluster_labels,
        positive_pairs, negative_pairs,
        positive_emb1, positive_emb2,
        output_dir / 'contrastive_relationships_tsne.png',
        method='tsne',
        num_display_pairs=args.num_display_pairs
    )
    
    # UMAP visualization (if available)
    if HAS_UMAP:
        plot_contrastive_relationships_2d(
            embeddings, cluster_labels,
            positive_pairs, negative_pairs,
            positive_emb1, positive_emb2,
            output_dir / 'contrastive_relationships_umap.png',
            method='umap',
            num_display_pairs=args.num_display_pairs
        )
    
    # Similarity heatmap
    plot_similarity_heatmap(
        embeddings, cluster_labels,
        output_dir / 'similarity_heatmap.png',
        max_samples=100
    )
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  • contrastive_relationships_tsne.png  - t-SNE with relationships")
    if HAS_UMAP:
        print(f"  • contrastive_relationships_umap.png  - UMAP with relationships")
    print(f"  • similarity_heatmap.png              - Cosine similarity matrix")
    print("\nVisualization Legend:")
    print("  • Colored points = Sequences grouped by cluster")
    print("  • Green arrows = Positive pairs (same sequence, different augmentations)")
    print("  • Red dashes = Negative pairs (different sequences)")
    print("  • Circle/Square markers = Two augmented views of same sequence")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
