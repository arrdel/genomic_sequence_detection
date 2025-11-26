#!/usr/bin/env python3
"""
Contrastive VQ-VAE Evaluation Script

This script evaluates a contrastive VQ-VAE model by:
1. Computing embedding quality metrics (alignment & uniformity)
2. Evaluating clustering quality
3. Generating UMAP/t-SNE visualizations
4. (Optional) Linear probing for downstream tasks

Usage:
    python scripts/contrastive_evaluate.py \
      --checkpoint-path outputs/contrastive_training/best_model.pt \
      --vqvae-checkpoint-path outputs/vqvae_training_1/checkpoint_epoch_50.pt \
      --data-path cleaned_reads.fastq \
      --output-dir ./contrastive_evaluation_results
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import json
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset
from src.models import VQVAE

# Optional dependencies
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except:
    HAS_TSNE = False


class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning"""
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
        z_mean = z_e.mean(dim=1)             # [B, D]
        z_proj = self.proj(z_mean)           # [B, proj_dim]
        z_norm = F.normalize(z_proj, dim=-1) # unit norm
        return z_norm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Contrastive VQ-VAE')
    
    # Data arguments
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to contrastive model checkpoint')
    parser.add_argument('--vqvae-checkpoint-path', type=str, required=True,
                        help='Path to base VQ-VAE checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./contrastive_evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size for tokenization')
    parser.add_argument('--use-canonical', action='store_true',
                        help='Use canonical k-mers')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num-eval-samples', type=int, default=5000,
                        help='Number of samples to evaluate (for efficiency)')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of clusters for K-means')
    parser.add_argument('--use-validation-split', action='store_true',
                        help='Evaluate on validation split instead of full dataset')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio if using validation split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization arguments
    parser.add_argument('--skip-umap', action='store_true',
                        help='Skip UMAP visualization')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE visualization')
    
    # GPU arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([b[1] for b in batch])
    return tokens, lengths


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=None):
    """Extract embeddings from the contrastive model"""
    model.eval()
    all_emb = []
    num_samples = 0
    
    for batch_tokens, _ in tqdm(dataloader, desc="Extracting embeddings"):
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        z = model(batch_tokens)             # normalized [B, D]
        all_emb.append(z.cpu().numpy())
        
        num_samples += z.size(0)
        if max_samples and num_samples >= max_samples:
            break
    
    embeddings = np.concatenate(all_emb, axis=0)
    if max_samples:
        embeddings = embeddings[:max_samples]
    
    return embeddings


def compute_alignment_uniformity(embeddings, temperature=0.5):
    """
    Compute alignment and uniformity metrics for contrastive learning.
    
    Alignment: How well positive pairs are aligned (lower is better)
    Uniformity: How uniformly embeddings are distributed (lower is better)
    
    Reference: "Understanding Contrastive Representation Learning through Alignment 
               and Uniformity on the Hypersphere" (Wang & Isola, 2020)
    
    Args:
        embeddings: (N, D) array of normalized embeddings
        temperature: Temperature parameter
    
    Returns:
        alignment: Alignment metric (simulated with consecutive pairs)
        uniformity: Uniformity metric
    """
    N = embeddings.shape[0]
    
    # Uniformity: measure distribution uniformity on hypersphere
    # uniformity = log E[exp(-||z_i - z_j||^2 / t)]
    # For efficiency, sample random pairs
    num_pairs = min(10000, N * (N - 1) // 2)
    idx1 = np.random.randint(0, N, num_pairs)
    idx2 = np.random.randint(0, N, num_pairs)
    
    # Remove self-pairs
    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]
    
    diffs = embeddings[idx1] - embeddings[idx2]
    sq_dists = np.sum(diffs ** 2, axis=1)
    uniformity = np.log(np.mean(np.exp(-sq_dists / temperature)))
    
    # Alignment: we don't have explicit positive pairs in this evaluation
    # We approximate by treating consecutive sequences as positive pairs
    # (assuming similar sequences are near each other in the dataset)
    num_pos_pairs = min(1000, N - 1)
    pos_idx = np.random.randint(0, N - 1, num_pos_pairs)
    
    pos_diffs = embeddings[pos_idx] - embeddings[pos_idx + 1]
    pos_sq_dists = np.sum(pos_diffs ** 2, axis=1)
    alignment = np.mean(pos_sq_dists)
    
    return alignment, uniformity


def evaluate_clustering(embeddings, num_clusters=10):
    """
    Evaluate clustering quality using K-means and DBSCAN.
    
    Returns:
        dict: Clustering metrics
    """
    print(f"\nEvaluating clustering quality...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score (higher is better, range [-1, 1])
    silhouette = silhouette_score(embeddings, kmeans_labels, sample_size=min(5000, len(embeddings)))
    
    # Davies-Bouldin index (lower is better, >= 0)
    davies_bouldin = davies_bouldin_score(embeddings, kmeans_labels)
    
    # Calinski-Harabasz index (higher is better, >= 0)
    calinski_harabasz = calinski_harabasz_score(embeddings, kmeans_labels)
    
    # Cluster sizes
    cluster_sizes = Counter(kmeans_labels)
    
    # DBSCAN clustering (density-based)
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(embeddings)
    
    num_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    num_noise_points = np.sum(dbscan_labels == -1)
    
    metrics = {
        'kmeans': {
            'num_clusters': num_clusters,
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'calinski_harabasz_index': float(calinski_harabasz),
            'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
            'min_cluster_size': int(min(cluster_sizes.values())),
            'max_cluster_size': int(max(cluster_sizes.values())),
            'avg_cluster_size': float(np.mean(list(cluster_sizes.values())))
        },
        'dbscan': {
            'num_clusters': int(num_dbscan_clusters),
            'num_noise_points': int(num_noise_points),
            'noise_ratio': float(num_noise_points / len(embeddings))
        }
    }
    
    return metrics, kmeans_labels


def plot_umap(embeddings, labels, out_png, title="UMAP of Contrastive Embeddings"):
    """Plot UMAP visualization with cluster colors"""
    if not HAS_UMAP:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")
        return
    
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Get unique clusters and colors
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster separately with proper labels
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_size = np.sum(mask)
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], 
                  c=[colors[i]], s=20, alpha=0.7, 
                  label=f'Cluster {cluster_id} (n={cluster_size})')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP plot: {out_png}")


def plot_tsne(embeddings, labels, out_png, title="t-SNE of Contrastive Embeddings"):
    """Plot t-SNE visualization with cluster colors"""
    if not HAS_TSNE:
        print("[WARN] scikit-learn not installed; skipping t-SNE plot.")
        return
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Get unique clusters and colors
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster separately with proper labels
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_size = np.sum(mask)
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                  c=[colors[i]], s=20, alpha=0.7,
                  label=f'Cluster {cluster_id} (n={cluster_size})')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE-1", fontsize=12)
    ax.set_ylabel("t-SNE-2", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE plot: {out_png}")


def compute_embedding_statistics(embeddings):
    """Compute basic statistics about the embeddings"""
    
    # Norms (should be ~1 for normalized embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Pairwise cosine similarities (sample for efficiency)
    num_pairs = min(10000, len(embeddings) * (len(embeddings) - 1) // 2)
    idx1 = np.random.randint(0, len(embeddings), num_pairs)
    idx2 = np.random.randint(0, len(embeddings), num_pairs)
    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]
    
    cos_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)
    
    stats = {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'norm_mean': float(np.mean(norms)),
        'norm_std': float(np.std(norms)),
        'norm_min': float(np.min(norms)),
        'norm_max': float(np.max(norms)),
        'cosine_sim_mean': float(np.mean(cos_sims)),
        'cosine_sim_std': float(np.std(cos_sims)),
        'cosine_sim_min': float(np.min(cos_sims)),
        'cosine_sim_max': float(np.max(cos_sims))
    }
    
    return stats


def plot_embedding_distribution(embeddings, out_png):
    """Plot distribution of embedding norms and cosine similarities"""
    
    # Compute norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Sample pairwise cosine similarities
    num_pairs = min(20000, len(embeddings) * (len(embeddings) - 1) // 2)
    idx1 = np.random.randint(0, len(embeddings), num_pairs)
    idx2 = np.random.randint(0, len(embeddings), num_pairs)
    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]
    
    cos_sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Norm distribution
    axes[0].hist(norms, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(norms), color='red', linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
    axes[0].set_xlabel('L2 Norm')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Embedding Norms')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Cosine similarity distribution
    axes[1].hist(cos_sims, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(cos_sims), color='red', linestyle='--', label=f'Mean: {np.mean(cos_sims):.3f}')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Pairwise Cosine Similarities')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"✓ Saved embedding distribution plot: {out_png}")


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
    
    print("\n" + "="*80)
    print("CONTRASTIVE VQ-VAE EVALUATION")
    print("="*80)
    
    # Load VQ-VAE checkpoint
    print(f"\nLoading base VQ-VAE from: {args.vqvae_checkpoint_path}")
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint_path, map_location=device)
    
    # Get model config
    if 'args' in vqvae_checkpoint:
        model_args = vqvae_checkpoint['args']
        num_codes = model_args.get('num_codes', 512)
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        commitment_cost = model_args.get('commitment_cost', 0.1)
        # Get original vocab size from checkpoint
        checkpoint_vocab_size = model_args.get('vocab_size', 4097)
        checkpoint_pad_id = model_args.get('pad_id', 4096)
    else:
        num_codes = 512
        code_dim = 128
        embed_dim = 128
        hidden_dim = 256
        commitment_cost = 0.1
        checkpoint_vocab_size = 4097
        checkpoint_pad_id = 4096
    
    # Create tokenizer (no special tokens needed - must match checkpoint)
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=args.use_canonical)
    
    PAD_ID = tokenizer.pad_id
    VOCAB_SIZE = len(tokenizer.stoi)
    
    print(f"\nTokenizer:")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  PAD ID: {PAD_ID}")
    print(f"  Checkpoint vocab size: {checkpoint_vocab_size}")
    
    # Initialize base VQ-VAE with CHECKPOINT vocab size to load weights
    base_model = VQVAE(
        checkpoint_vocab_size,
        checkpoint_pad_id,
        num_codes=num_codes,
        code_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        commitment_cost=commitment_cost
    ).to(device)
    
    # Load VQ-VAE weights
    state_dict = vqvae_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    print("✓ VQ-VAE loaded successfully")
    print(f"  Note: Using checkpoint vocab size {checkpoint_vocab_size}")
    
    # Load contrastive checkpoint
    print(f"\nLoading contrastive model from: {args.checkpoint_path}")
    contrastive_checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get projection dim from checkpoint
    if 'args' in contrastive_checkpoint:
        proj_dim = contrastive_checkpoint['args'].get('proj_dim', 64)
    else:
        proj_dim = 64
    
    print(f"  Projection dim: {proj_dim}")
    print(f"  Encoder output dim: {code_dim} (code_dim)")
    
    # Build contrastive model
    # NOTE: Encoder outputs code_dim, not embed_dim!
    contrastive_model = ContrastiveHead(
        encoder=base_model.encoder,
        embed_dim=code_dim,  # Use code_dim, not embed_dim!
        proj_dim=proj_dim
    ).to(device)
    
    # Load contrastive weights
    state_dict = contrastive_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    contrastive_model.load_state_dict(state_dict)
    contrastive_model.eval()
    
    print("✓ Contrastive model loaded successfully")
    
    # Create dataset
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Optionally split into validation set
    if args.use_validation_split:
        dataset_size = len(dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(args.seed)
        _, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"\nUsing validation split: {val_size} sequences")
    else:
        eval_dataset = dataset
        print(f"\nUsing full dataset: {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Extract embeddings
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    
    embeddings = extract_embeddings(
        contrastive_model,
        dataloader,
        device,
        max_samples=args.num_eval_samples
    )
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    emb_path = os.path.join(args.output_dir, 'embeddings.npy')
    np.save(emb_path, embeddings)
    print(f"✓ Saved embeddings: {emb_path}")
    
    # Save cluster labels (will be generated later)
    cluster_labels_path = os.path.join(args.output_dir, 'cluster_labels.npy')
    
    # Compute embedding statistics
    print("\n" + "="*80)
    print("EMBEDDING STATISTICS")
    print("="*80)
    
    emb_stats = compute_embedding_statistics(embeddings)
    
    print(f"\nEmbedding Statistics:")
    print(f"  Number of embeddings: {emb_stats['num_embeddings']}")
    print(f"  Embedding dimension: {emb_stats['embedding_dim']}")
    print(f"  Norm: {emb_stats['norm_mean']:.4f} ± {emb_stats['norm_std']:.4f}")
    print(f"  Norm range: [{emb_stats['norm_min']:.4f}, {emb_stats['norm_max']:.4f}]")
    print(f"  Cosine similarity: {emb_stats['cosine_sim_mean']:.4f} ± {emb_stats['cosine_sim_std']:.4f}")
    print(f"  Cosine similarity range: [{emb_stats['cosine_sim_min']:.4f}, {emb_stats['cosine_sim_max']:.4f}]")
    
    # Plot embedding distributions
    dist_plot_path = os.path.join(args.output_dir, 'embedding_distributions.png')
    plot_embedding_distribution(embeddings, dist_plot_path)
    
    # Compute alignment and uniformity
    print("\n" + "="*80)
    print("ALIGNMENT & UNIFORMITY METRICS")
    print("="*80)
    
    alignment, uniformity = compute_alignment_uniformity(embeddings)
    
    print(f"\nAlignment & Uniformity:")
    print(f"  Alignment: {alignment:.4f} (lower is better)")
    print(f"  Uniformity: {uniformity:.4f} (lower is better)")
    
    # Evaluate clustering
    print("\n" + "="*80)
    print("CLUSTERING EVALUATION")
    print("="*80)
    
    clustering_metrics, cluster_labels = evaluate_clustering(embeddings, num_clusters=args.num_clusters)
    
    # Save cluster labels for later use with taxonomy
    np.save(cluster_labels_path, cluster_labels)
    print(f"✓ Saved cluster labels: {cluster_labels_path}")
    
    print(f"\nK-Means Clustering:")
    print(f"  Silhouette score: {clustering_metrics['kmeans']['silhouette_score']:.4f} (higher is better)")
    print(f"  Davies-Bouldin index: {clustering_metrics['kmeans']['davies_bouldin_index']:.4f} (lower is better)")
    print(f"  Calinski-Harabasz index: {clustering_metrics['kmeans']['calinski_harabasz_index']:.2f} (higher is better)")
    print(f"  Cluster size range: [{clustering_metrics['kmeans']['min_cluster_size']}, {clustering_metrics['kmeans']['max_cluster_size']}]")
    
    print(f"\nDBSCAN Clustering:")
    print(f"  Number of clusters: {clustering_metrics['dbscan']['num_clusters']}")
    print(f"  Noise points: {clustering_metrics['dbscan']['num_noise_points']} ({clustering_metrics['dbscan']['noise_ratio']*100:.2f}%)")
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    if not args.skip_umap:
        umap_path = os.path.join(args.output_dir, 'embeddings_umap.png')
        plot_umap(embeddings, cluster_labels, umap_path)
    
    if not args.skip_tsne:
        tsne_path = os.path.join(args.output_dir, 'embeddings_tsne.png')
        plot_tsne(embeddings, cluster_labels, tsne_path)
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results = {
        'checkpoint_path': args.checkpoint_path,
        'vqvae_checkpoint_path': args.vqvae_checkpoint_path,
        'data_path': args.data_path,
        'evaluation_config': {
            'num_eval_samples': args.num_eval_samples,
            'batch_size': args.batch_size,
            'num_clusters': args.num_clusters,
            'use_validation_split': args.use_validation_split,
            'seed': args.seed
        },
        'embedding_statistics': emb_stats,
        'alignment_uniformity': {
            'alignment': float(alignment),
            'uniformity': float(uniformity)
        },
        'clustering_metrics': clustering_metrics,
        'model_config': {
            'vocab_size': VOCAB_SIZE,
            'num_codes': num_codes,
            'code_dim': code_dim,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'proj_dim': proj_dim
        }
    }
    
    results_file = os.path.join(args.output_dir, 'contrastive_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved evaluation results: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Number of embeddings evaluated: {emb_stats['num_embeddings']}")
    print(f"Embedding dimension: {emb_stats['embedding_dim']}")
    print(f"Alignment: {alignment:.4f}")
    print(f"Uniformity: {uniformity:.4f}")
    print(f"Silhouette score: {clustering_metrics['kmeans']['silhouette_score']:.4f}")
    print(f"Results directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
