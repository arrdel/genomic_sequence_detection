#!/usr/bin/env python3
"""
Unified Model Comparison Script

Compares three VQ-VAE approaches on the same test set:
1. Standard VQ-VAE (full reconstruction)
2. Masked VQ-VAE (BERT-style masked language modeling)
3. Contrastive VQ-VAE (SimCLR-style contrastive learning)

Usage:
    python scripts/evaluation.py \
      --standard-checkpoint outputs/vqvae_train_run_5/best_model.pt \
      --masked-checkpoint outputs/mqvae_train_run_5/best_model.pt \
      --contrastive-checkpoint outputs/contrastive_training/contrastive_finetune_run_1/best_model.pt \
      --data-path data/cleaned_reads.fastq \
      --output-dir evaluation_results \
      --num-samples 10000
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sklearn metrics
from sklearn.metrics import ( 
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except:
    HAS_TSNE = False


# ============================================================================
# Model Definitions
# ============================================================================

class ContrastiveHead(nn.Module):
    """Contrastive learning projection head"""
    def __init__(self, encoder, embed_dim=64, proj_dim=64):
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
        z_norm = F.normalize(z_proj, dim=-1)
        return z_norm


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Compare all VQ-VAE models')
    
    # Model checkpoints
    parser.add_argument('--standard-checkpoint', type=str, required=True,
                        help='Path to standard VQ-VAE checkpoint')
    parser.add_argument('--masked-checkpoint', type=str, required=True,
                        help='Path to masked VQ-VAE checkpoint')
    parser.add_argument('--contrastive-checkpoint', type=str, required=True,
                        help='Path to contrastive VQ-VAE checkpoint')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to FASTQ file')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model config
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size')
    parser.add_argument('--use-canonical', action='store_true',
                        help='Use canonical k-mers')
    
    # Evaluation settings
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of clusters for K-means')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip UMAP/t-SNE visualizations')
    
    # GPU
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_standard_vqvae(checkpoint_path, device):
    """Load standard VQ-VAE model"""
    print(f"\nLoading Standard VQ-VAE from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'args' in checkpoint:
        args = checkpoint['args']
        vocab_size = args.get('vocab_size', 4097)
        pad_id = args.get('pad_id', 4096)
        num_codes = args.get('num_codes', 512)
        code_dim = args.get('code_dim', 64)
        embed_dim = args.get('embed_dim', 128)
        hidden_dim = args.get('hidden_dim', 256)
        commitment_cost = args.get('commitment_cost', 0.1)
    else:
        vocab_size, pad_id = 4097, 4096
        num_codes, code_dim = 512, 64
        embed_dim, hidden_dim = 128, 256
        commitment_cost = 0.1
    
    # Create model
    model = VQVAE(
        vocab_size, pad_id,
        num_codes=num_codes,
        code_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        commitment_cost=commitment_cost
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"  ✓ Loaded (vocab={vocab_size}, codes={num_codes}, dim={code_dim})")
    return model, code_dim


def load_masked_vqvae(checkpoint_path, device):
    """Load masked VQ-VAE model"""
    print(f"\nLoading Masked VQ-VAE from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if 'args' in checkpoint:
        args = checkpoint['args']
        vocab_size = args.get('vocab_size', 4099)
        pad_id = args.get('pad_id', 4096)
        num_codes = args.get('num_codes', 512)
        code_dim = args.get('code_dim', 64)
        embed_dim = args.get('embed_dim', 128)
        hidden_dim = args.get('hidden_dim', 256)
        commitment_cost = args.get('commitment_cost', 0.1)
    else:
        vocab_size, pad_id = 4099, 4096
        num_codes, code_dim = 512, 64
        embed_dim, hidden_dim = 128, 256
        commitment_cost = 0.1
    
    # Create model
    model = VQVAE(
        vocab_size, pad_id,
        num_codes=num_codes,
        code_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        commitment_cost=commitment_cost
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"  ✓ Loaded (vocab={vocab_size}, codes={num_codes}, dim={code_dim})")
    return model, code_dim


def load_contrastive_vqvae(checkpoint_path, base_checkpoint_path, device):
    """Load contrastive VQ-VAE model"""
    print(f"\nLoading Contrastive VQ-VAE from: {checkpoint_path}")
    
    # Load base VQ-VAE first
    base_model, code_dim = load_standard_vqvae(base_checkpoint_path, device)
    
    # Load contrastive checkpoint
    contrastive_checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'args' in contrastive_checkpoint:
        proj_dim = contrastive_checkpoint['args'].get('proj_dim', 64)
    else:
        proj_dim = 64
    
    # Create contrastive head
    model = ContrastiveHead(
        encoder=base_model.encoder,
        embed_dim=code_dim,
        proj_dim=proj_dim
    ).to(device)
    
    # Load weights
    state_dict = contrastive_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"  ✓ Loaded (proj_dim={proj_dim})")
    return model, base_model


# ============================================================================
# Embedding Extraction
# ============================================================================

def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([b[1] for b in batch])
    return tokens, lengths


@torch.no_grad()
def extract_embeddings_standard(model, dataloader, device):
    """Extract embeddings from standard VQ-VAE (mean-pooled encoder output)"""
    model.eval()
    all_embeddings = []
    all_codes = []
    all_tokens = []
    
    for batch_tokens, _ in tqdm(dataloader, desc="Extracting (Standard)"):
        batch_tokens = batch_tokens.to(device)
        
        # Get encoder output
        z_e = model.encoder(batch_tokens)  # [B, L, D]
        z_mean = z_e.mean(dim=1)           # [B, D] - mean pool
        
        # Get codes for codebook analysis
        z_q, _, codes = model.vq(z_e)
        
        all_embeddings.append(z_mean.cpu().numpy())
        all_codes.append(codes.cpu().numpy())
        all_tokens.append(batch_tokens.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    codes = np.concatenate(all_codes, axis=0)
    tokens = np.concatenate(all_tokens, axis=0)
    
    return embeddings, codes, tokens


@torch.no_grad()
def extract_embeddings_masked(model, dataloader, device):
    """Extract embeddings from masked VQ-VAE"""
    return extract_embeddings_standard(model, dataloader, device)


@torch.no_grad()
def extract_embeddings_contrastive(model, dataloader, device):
    """Extract embeddings from contrastive VQ-VAE"""
    model.eval()
    all_embeddings = []
    all_tokens = []
    
    for batch_tokens, _ in tqdm(dataloader, desc="Extracting (Contrastive)"):
        batch_tokens = batch_tokens.to(device)
        
        # Get normalized embeddings
        z_norm = model(batch_tokens)  # [B, proj_dim]
        
        all_embeddings.append(z_norm.cpu().numpy())
        all_tokens.append(batch_tokens.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    tokens = np.concatenate(all_tokens, axis=0)
    
    # No codes for contrastive model
    codes = None
    
    return embeddings, codes, tokens


# ============================================================================
# Reconstruction Metrics
# ============================================================================

@torch.no_grad()
def compute_reconstruction_metrics(model, dataloader, device, pad_id):
    """Compute reconstruction metrics for VQ-VAE models"""
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    perfect_sequences = 0
    total_sequences = 0
    edit_distances = []
    
    for batch_tokens, _ in tqdm(dataloader, desc="Computing reconstruction"):
        batch_tokens = batch_tokens.to(device)
        
        # Forward pass
        logits, _, _ = model(batch_tokens)
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Mask for non-padding tokens
        mask = (batch_tokens != pad_id)
        
        # Token-level accuracy
        correct = (preds == batch_tokens) & mask
        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()
        
        # Sequence-level accuracy
        perfect = (preds == batch_tokens).all(dim=1)
        perfect_sequences += perfect.sum().item()
        total_sequences += batch_tokens.size(0)
        
        # Edit distance (sample first 100 sequences for speed)
        if len(edit_distances) < 100:
            for i in range(min(10, batch_tokens.size(0))):
                orig_len = mask[i].sum().item()
                orig_seq = batch_tokens[i, :orig_len].cpu().tolist()
                pred_seq = preds[i, :orig_len].cpu().tolist()
                
                # Simple edit distance
                dist = sum(o != p for o, p in zip(orig_seq, pred_seq))
                edit_distances.append(dist / orig_len if orig_len > 0 else 0)
    
    metrics = {
        'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0,
        'sequence_accuracy': perfect_sequences / total_sequences if total_sequences > 0 else 0,
        'avg_edit_distance': np.mean(edit_distances) if edit_distances else 0,
        'total_sequences': total_sequences
    }
    
    return metrics


# ============================================================================
# Codebook Metrics
# ============================================================================

def compute_codebook_metrics(codes, num_codes):
    """Compute codebook utilization and perplexity"""
    if codes is None:
        return {}
    
    # Flatten codes
    codes_flat = codes.flatten()
    
    # Unique codes used
    unique_codes = np.unique(codes_flat)
    utilization = len(unique_codes) / num_codes * 100
    
    # Code frequency
    code_counts = np.bincount(codes_flat, minlength=num_codes)
    code_probs = code_counts / code_counts.sum()
    
    # Entropy and perplexity
    code_probs_nonzero = code_probs[code_probs > 0]
    entropy = -(code_probs_nonzero * np.log(code_probs_nonzero + 1e-10)).sum()
    perplexity = np.exp(entropy)
    
    # Dead codes
    dead_codes = num_codes - len(unique_codes)
    
    metrics = {
        'codebook_utilization': utilization,
        'perplexity': perplexity,
        'entropy': entropy,
        'unique_codes': len(unique_codes),
        'dead_codes': dead_codes,
        'total_codes': num_codes
    }
    
    return metrics


# ============================================================================
# Embedding Quality Metrics
# ============================================================================

def compute_alignment_uniformity(embeddings, num_pairs=5000, temperature=2.0):
    """
    Compute alignment and uniformity metrics (Wang & Isola, 2020)
    
    Alignment: How well similar sequences align (lower = better)
    Uniformity: How uniformly distributed embeddings are (lower = better)
    """
    N = len(embeddings)
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Uniformity: measure distribution uniformity on hypersphere
    num_pairs = min(num_pairs, N * (N - 1) // 2)
    idx = np.random.choice(N, size=(num_pairs, 2), replace=True)
    
    # Remove self-pairs
    valid = idx[:, 0] != idx[:, 1]
    idx = idx[valid]
    
    diffs = embeddings_norm[idx[:, 0]] - embeddings_norm[idx[:, 1]]
    sq_dists = (diffs ** 2).sum(axis=1)
    uniformity = np.log(np.mean(np.exp(-temperature * sq_dists)))
    
    # Alignment: approximate with consecutive pairs
    num_align_pairs = min(1000, N - 1)
    align_idx = np.random.randint(0, N - 1, num_align_pairs)
    
    align_diffs = embeddings_norm[align_idx] - embeddings_norm[align_idx + 1]
    align_sq_dists = (align_diffs ** 2).sum(axis=1)
    alignment = np.mean(align_sq_dists)
    
    return {
        'alignment': alignment,
        'uniformity': uniformity
    }


# ============================================================================
# Clustering Metrics
# ============================================================================

def compute_clustering_metrics(embeddings, num_clusters=10):
    """Compute clustering quality metrics"""
    print(f"\nComputing clustering metrics (k={num_clusters})...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score
    sample_size = min(5000, len(embeddings))
    silhouette = silhouette_score(
        embeddings, labels, 
        sample_size=sample_size,
        random_state=42
    )
    
    # Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    # Calinski-Harabasz index
    calinski = calinski_harabasz_score(embeddings, labels)
    
    # Cluster sizes
    cluster_sizes = Counter(labels)
    min_cluster_size = min(cluster_sizes.values())
    max_cluster_size = max(cluster_sizes.values())
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski,
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'cluster_balance': min_cluster_size / max_cluster_size
    }
    
    return metrics, labels


# ============================================================================
# Downstream Task Metrics
# ============================================================================

def compute_downstream_metrics(embeddings, labels=None, test_size=0.3):
    """
    Compute downstream task performance
    
    For now, we'll use cluster labels as pseudo-labels
    In a real scenario, you'd use actual virus family labels
    """
    if labels is None:
        # Use K-means cluster labels as pseudo-labels
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Linear probe
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average='weighted')
    
    # K-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average='weighted')
    
    metrics = {
        'linear_probe_accuracy': lr_accuracy,
        'linear_probe_f1': lr_f1,
        'knn_accuracy': knn_accuracy,
        'knn_f1': knn_f1
    }
    
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_embeddings_umap(embeddings_dict, labels_dict, output_path):
    """Plot UMAP visualizations for all models"""
    if not HAS_UMAP:
        print("Skipping UMAP (not installed)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        print(f"Computing UMAP for {model_name}...")
        
        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            metric='cosine', 
            random_state=42
        )
        emb_2d = reducer.fit_transform(embeddings)
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            emb_2d[:, 0], 
            emb_2d[:, 1], 
            c=labels_dict[model_name], 
            cmap='tab10', 
            s=1, 
            alpha=0.5
        )
        ax.set_title(f'{model_name} UMAP', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP visualization: {output_path}")


def plot_embeddings_tsne(embeddings_dict, labels_dict, output_path):
    """Plot t-SNE visualizations for all models"""
    if not HAS_TSNE:
        print("Skipping t-SNE (not installed)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        print(f"Computing t-SNE for {model_name}...")
        
        # Compute t-SNE
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=30,
            n_iter=1000
        )
        emb_2d = tsne.fit_transform(embeddings)
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            emb_2d[:, 0], 
            emb_2d[:, 1], 
            c=labels_dict[model_name], 
            cmap='tab10', 
            s=1, 
            alpha=0.5
        )
        ax.set_title(f'{model_name} t-SNE', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE visualization: {output_path}")


def plot_comparison_bar_charts(results_dict, output_path):
    """Plot bar charts comparing key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    models = list(results_dict.keys())
    
    # Metrics to plot
    metrics_to_plot = [
        ('token_accuracy', 'Token Accuracy (%)', 100, 'higher'),
        ('sequence_accuracy', 'Sequence Accuracy (%)', 100, 'higher'),
        ('codebook_utilization', 'Codebook Utilization (%)', 1, 'higher'),
        ('silhouette_score', 'Silhouette Score', 1, 'higher'),
        ('davies_bouldin_index', 'Davies-Bouldin Index', 1, 'lower'),
        ('linear_probe_accuracy', 'Linear Probe Accuracy (%)', 100, 'higher'),
    ]
    
    for idx, (metric_key, ylabel, scale, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = []
        valid_indices = []
        for i, model in enumerate(models):
            val = results_dict[model].get(metric_key, None)
            if val is not None:
                values.append(val * scale)
                valid_indices.append(i)
            else:
                values.append(0)  # Placeholder for None values
        
        # Create bar colors
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bar_colors = [colors[i] for i in range(len(models))]
        
        bars = ax.bar(models, values, color=bar_colors)
        
        # Highlight best (only among valid values)
        if len(valid_indices) > 0:
            valid_values = [values[i] for i in valid_indices]
            if direction == 'higher':
                best_val = max(valid_values)
                best_idx = values.index(best_val)
            else:
                best_val = min(valid_values)
                best_idx = values.index(best_val)
            bars[best_idx].set_color('#f39c12')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(ylabel, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only for non-None values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            # Check if this was a valid (non-None) value
            if i in valid_indices and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=10)
            elif i not in valid_indices:
                # Mark N/A values
                ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                       'N/A',
                       ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison charts: {output_path}")


def create_comparison_table(results_dict, output_path):
    """Create a detailed comparison table"""
    
    # Metrics to include in table
    metrics = [
        ('token_accuracy', 'Token Accuracy (%)', 100, 2, '↑'),
        ('sequence_accuracy', 'Sequence Accuracy (%)', 100, 2, '↑'),
        ('codebook_utilization', 'Codebook Utilization (%)', 1, 2, '↑'),
        ('perplexity', 'Codebook Perplexity', 1, 1, '↑'),
        ('alignment', 'Alignment', 1, 4, '↓'),
        ('uniformity', 'Uniformity', 1, 4, '↓'),
        ('silhouette_score', 'Silhouette Score', 1, 3, '↑'),
        ('davies_bouldin_index', 'Davies-Bouldin Index', 1, 3, '↓'),
        ('calinski_harabasz_score', 'Calinski-Harabasz', 1, 1, '↑'),
        ('linear_probe_accuracy', 'Linear Probe Acc (%)', 100, 2, '↑'),
        ('knn_accuracy', 'K-NN Accuracy (%)', 100, 2, '↑'),
    ]
    
    # Create table data
    table_data = []
    
    for metric_key, metric_name, scale, decimals, direction in metrics:
        row = {'Metric': f"{metric_name} {direction}"}
        
        values = []
        for model_name in results_dict.keys():
            val = results_dict[model_name].get(metric_key, None)
            if val is not None:
                val_scaled = val * scale
                row[model_name] = f"{val_scaled:.{decimals}f}"
                values.append((model_name, val_scaled))
            else:
                row[model_name] = 'N/A'
        
        # Determine best
        if len(values) > 0:
            if direction == '↑':
                best_model = max(values, key=lambda x: x[1])[0]
            else:
                best_model = min(values, key=lambda x: x[1])[0]
            row['Best'] = best_model
        else:
            row['Best'] = 'N/A'
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = str(output_path).replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as formatted text
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*100 + "\n")
    
    print(f"✓ Saved comparison table: {output_path}")
    print(f"✓ Saved CSV: {csv_path}")
    
    return df


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU {args.gpu_id}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("UNIFIED MODEL COMPARISON")
    print("="*80)
    
    # Create tokenizer
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=args.use_canonical)
    PAD_ID = tokenizer.pad_id
    VOCAB_SIZE = len(tokenizer.stoi)
    
    print(f"\nTokenizer: vocab_size={VOCAB_SIZE}, pad_id={PAD_ID}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Sample subset if specified
    if args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = Subset(dataset, indices)
    
    print(f"Using {len(dataset)} sequences for evaluation")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    standard_model, std_code_dim = load_standard_vqvae(args.standard_checkpoint, device)
    masked_model, msk_code_dim = load_masked_vqvae(args.masked_checkpoint, device)
    contrastive_model, base_model = load_contrastive_vqvae(
        args.contrastive_checkpoint, 
        args.standard_checkpoint,
        device
    )
    
    # Extract embeddings and codes
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    
    std_embeddings, std_codes, std_tokens = extract_embeddings_standard(
        standard_model, dataloader, device
    )
    msk_embeddings, msk_codes, msk_tokens = extract_embeddings_masked(
        masked_model, dataloader, device
    )
    con_embeddings, _, con_tokens = extract_embeddings_contrastive(
        contrastive_model, dataloader, device
    )
    
    # Save embeddings
    np.save(output_dir / 'standard_embeddings.npy', std_embeddings)
    np.save(output_dir / 'masked_embeddings.npy', msk_embeddings)
    np.save(output_dir / 'contrastive_embeddings.npy', con_embeddings)
    print(f"\n✓ Saved embeddings to {output_dir}")
    
    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    
    results = {}
    
    # ========== STANDARD VQ-VAE ==========
    print("\n[1/3] Evaluating Standard VQ-VAE...")
    results['Standard VQ-VAE'] = {}
    
    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(
        standard_model, dataloader, device, PAD_ID
    )
    results['Standard VQ-VAE'].update(recon_metrics)
    
    # Codebook metrics
    codebook_metrics = compute_codebook_metrics(std_codes, 512)
    results['Standard VQ-VAE'].update(codebook_metrics)
    
    # Embedding quality
    align_uni = compute_alignment_uniformity(std_embeddings)
    results['Standard VQ-VAE'].update(align_uni)
    
    # Clustering
    cluster_metrics, std_labels = compute_clustering_metrics(
        std_embeddings, args.num_clusters
    )
    results['Standard VQ-VAE'].update(cluster_metrics)
    
    # Downstream
    downstream_metrics = compute_downstream_metrics(std_embeddings, std_labels)
    results['Standard VQ-VAE'].update(downstream_metrics)
    
    # ========== MASKED VQ-VAE ==========
    print("\n[2/3] Evaluating Masked VQ-VAE...")
    results['Masked VQ-VAE'] = {}
    
    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(
        masked_model, dataloader, device, PAD_ID
    )
    results['Masked VQ-VAE'].update(recon_metrics)
    
    # Codebook metrics
    codebook_metrics = compute_codebook_metrics(msk_codes, 512)
    results['Masked VQ-VAE'].update(codebook_metrics)
    
    # Embedding quality
    align_uni = compute_alignment_uniformity(msk_embeddings)
    results['Masked VQ-VAE'].update(align_uni)
    
    # Clustering
    cluster_metrics, msk_labels = compute_clustering_metrics(
        msk_embeddings, args.num_clusters
    )
    results['Masked VQ-VAE'].update(cluster_metrics)
    
    # Downstream
    downstream_metrics = compute_downstream_metrics(msk_embeddings, msk_labels)
    results['Masked VQ-VAE'].update(downstream_metrics)
    
    # ========== CONTRASTIVE VQ-VAE ==========
    print("\n[3/3] Evaluating Contrastive VQ-VAE...")
    results['Contrastive VQ-VAE'] = {}
    
    # No reconstruction metrics for contrastive
    results['Contrastive VQ-VAE']['token_accuracy'] = None
    results['Contrastive VQ-VAE']['sequence_accuracy'] = None
    results['Contrastive VQ-VAE']['codebook_utilization'] = None
    results['Contrastive VQ-VAE']['perplexity'] = None
    
    # Embedding quality
    align_uni = compute_alignment_uniformity(con_embeddings)
    results['Contrastive VQ-VAE'].update(align_uni)
    
    # Clustering
    cluster_metrics, con_labels = compute_clustering_metrics(
        con_embeddings, args.num_clusters
    )
    results['Contrastive VQ-VAE'].update(cluster_metrics)
    
    # Downstream
    downstream_metrics = compute_downstream_metrics(con_embeddings, con_labels)
    results['Contrastive VQ-VAE'].update(downstream_metrics)
    
    # Save cluster labels for all models
    np.save(output_dir / 'standard_labels.npy', std_labels)
    np.save(output_dir / 'masked_labels.npy', msk_labels)
    np.save(output_dir / 'contrastive_labels.npy', con_labels)
    print(f"\n✓ Saved cluster labels to {output_dir}")
    
    # Save results as JSON
    results_json_path = output_dir / 'comparison_results.json'
    with open(results_json_path, 'w') as f:
        # Convert numpy types to native Python types
        results_serializable = {}
        for model, metrics in results.items():
            results_serializable[model] = {
                k: float(v) if v is not None and not isinstance(v, (str, int)) else v
                for k, v in metrics.items()
            }
        json.dump(results_serializable, f, indent=2)
    print(f"\n✓ Saved JSON results: {results_json_path}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    if not args.skip_visualizations:
        # UMAP plot
        embeddings_dict = {
            'Standard VQ-VAE': std_embeddings,
            'Masked VQ-VAE': msk_embeddings,
            'Contrastive VQ-VAE': con_embeddings
        }
        labels_dict = {
            'Standard VQ-VAE': std_labels,
            'Masked VQ-VAE': msk_labels,
            'Contrastive VQ-VAE': con_labels
        }
        plot_embeddings_umap(
            embeddings_dict, 
            labels_dict, 
            output_dir / 'umap_comparison.png'
        )
        
        # t-SNE plot
        plot_embeddings_tsne(
            embeddings_dict, 
            labels_dict, 
            output_dir / 'tsne_comparison.png'
        )
    
    # Bar charts
    plot_comparison_bar_charts(
        results, 
        output_dir / 'metrics_comparison.png'
    )
    
    # Comparison table
    df = create_comparison_table(
        results,
        output_dir / 'comparison_table.txt'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  • comparison_results.json    - Full metrics in JSON")
    print(f"  • comparison_table.txt       - Formatted comparison table")
    print(f"  • comparison_table.csv       - CSV format table")
    print(f"  • metrics_comparison.png     - Bar chart comparison")
    print(f"  • umap_comparison.png        - UMAP visualizations")
    print(f"  • tsne_comparison.png        - t-SNE visualizations")
    print(f"  • *_embeddings.npy           - Extracted embeddings")
    print(f"  • *_labels.npy               - Cluster labels")
    print("\n" + "="*80)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("-" * 80)
    
    # Best reconstruction
    if results['Standard VQ-VAE']['token_accuracy'] and results['Masked VQ-VAE']['token_accuracy']:
        best_recon = max(
            ('Standard', results['Standard VQ-VAE']['token_accuracy']),
            ('Masked', results['Masked VQ-VAE']['token_accuracy']),
            key=lambda x: x[1]
        )
        print(f"• Best Reconstruction: {best_recon[0]} VQ-VAE ({best_recon[1]*100:.2f}% token accuracy)")
    
    # Best clustering
    best_cluster = max(
        ('Standard', results['Standard VQ-VAE']['silhouette_score']),
        ('Masked', results['Masked VQ-VAE']['silhouette_score']),
        ('Contrastive', results['Contrastive VQ-VAE']['silhouette_score']),
        key=lambda x: x[1]
    )
    print(f"• Best Clustering: {best_cluster[0]} VQ-VAE (silhouette={best_cluster[1]:.3f})")
    
    # Best downstream
    best_downstream = max(
        ('Standard', results['Standard VQ-VAE']['linear_probe_accuracy']),
        ('Masked', results['Masked VQ-VAE']['linear_probe_accuracy']),
        ('Contrastive', results['Contrastive VQ-VAE']['linear_probe_accuracy']),
        key=lambda x: x[1]
    )
    print(f"• Best Downstream: {best_downstream[0]} VQ-VAE ({best_downstream[1]*100:.2f}% linear probe acc)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
