#!/usr/bin/env python3
"""
Create 4-Model Side-by-Side Visualization (UMAP & t-SNE)

This script creates comprehensive visualizations comparing all 4 VQ-VAE variants:
1. Standard VQ-VAE
2. Masked VQ-VAE  
3. Contrastive VQ-VAE (64-dim)
4. Contrastive VQ-VAE (128-dim)

Generates:
- 4-model UMAP comparison
- 4-model t-SNE comparison
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional UMAP
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. UMAP plots will be skipped.")


def load_embeddings(embeddings_path):
    """Load embeddings from .npy file"""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    return np.load(embeddings_path)


def load_labels(labels_path):
    """Load cluster labels from .npy file"""
    if not os.path.exists(labels_path):
        print(f"    Warning: Labels not found: {labels_path}")
        return None
    return np.load(labels_path)


def plot_4model_umap(embeddings_dict, labels_dict, output_path, sample_size=5000):
    """Create 4-model UMAP comparison plot"""
    if not HAS_UMAP:
        print("Skipping UMAP plot (umap-learn not installed)")
        return
    
    print("\nGenerating 4-model UMAP comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    models = [
        'Standard VQ-VAE',
        'Masked VQ-VAE',
        'Contrastive VQ-VAE (64-dim)',
        'Contrastive VQ-VAE (128-dim)'
    ]
    
    for idx, model_name in enumerate(models):
        if model_name not in embeddings_dict:
            print(f"  Warning: {model_name} embeddings not found, skipping...")
            axes[idx].text(0.5, 0.5, f'{model_name}\nNot Available', 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        print(f"  Processing {model_name}...")
        
        emb = embeddings_dict[model_name]
        labels = labels_dict.get(model_name, None)
        
        # Sample if too large
        if len(emb) > sample_size:
            indices = np.random.choice(len(emb), sample_size, replace=False)
            emb = emb[indices]
            if labels is not None:
                labels = labels[indices]
        
        # UMAP reduction
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        emb_2d = reducer.fit_transform(emb)
        
        # Plot
        ax = axes[idx]
        if labels is not None:
            scatter = ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                c=labels, cmap='tab10',
                s=8, alpha=0.6, edgecolors='none'
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                s=8, alpha=0.6, edgecolors='none',
                color='steelblue'
            )
        
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP-1', fontsize=11)
        ax.set_ylabel('UMAP-2', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('4-Model UMAP Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved UMAP comparison: {output_path}")


def plot_4model_tsne(embeddings_dict, labels_dict, output_path, sample_size=5000):
    """Create 4-model t-SNE comparison plot"""
    print("\nGenerating 4-model t-SNE comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    models = [
        'Standard VQ-VAE',
        'Masked VQ-VAE',
        'Contrastive VQ-VAE (64-dim)',
        'Contrastive VQ-VAE (128-dim)'
    ]
    
    for idx, model_name in enumerate(models):
        if model_name not in embeddings_dict:
            print(f"  Warning: {model_name} embeddings not found, skipping...")
            axes[idx].text(0.5, 0.5, f'{model_name}\nNot Available', 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        print(f"  Processing {model_name}...")
        
        emb = embeddings_dict[model_name]
        labels = labels_dict.get(model_name, None)
        
        # Sample if too large
        if len(emb) > sample_size:
            indices = np.random.choice(len(emb), sample_size, replace=False)
            emb = emb[indices]
            if labels is not None:
                labels = labels[indices]
        
        # t-SNE reduction
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            metric='cosine'
        )
        emb_2d = reducer.fit_transform(emb)
        
        # Plot
        ax = axes[idx]
        if labels is not None:
            scatter = ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                c=labels, cmap='tab10',
                s=8, alpha=0.6, edgecolors='none'
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(
                emb_2d[:, 0], emb_2d[:, 1],
                s=8, alpha=0.6, edgecolors='none',
                color='steelblue'
            )
        
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE-1', fontsize=11)
        ax.set_ylabel('t-SNE-2', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('4-Model t-SNE Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved t-SNE comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create 4-model visualization comparison (UMAP & t-SNE)'
    )
    parser.add_argument('--standard-embeddings', type=str, required=True,
                        help='Path to Standard VQ-VAE embeddings (.npy)')
    parser.add_argument('--masked-embeddings', type=str, required=True,
                        help='Path to Masked VQ-VAE embeddings (.npy)')
    parser.add_argument('--contrastive-64-embeddings', type=str, required=True,
                        help='Path to Contrastive 64-dim embeddings (.npy)')
    parser.add_argument('--contrastive-128-embeddings', type=str, required=True,
                        help='Path to Contrastive 128-dim embeddings (.npy)')
    parser.add_argument('--standard-labels', type=str, default=None,
                        help='Path to Standard VQ-VAE cluster labels (.npy)')
    parser.add_argument('--masked-labels', type=str, default=None,
                        help='Path to Masked VQ-VAE cluster labels (.npy)')
    parser.add_argument('--contrastive-64-labels', type=str, default=None,
                        help='Path to Contrastive 64-dim cluster labels (.npy)')
    parser.add_argument('--contrastive-128-labels', type=str, default=None,
                        help='Path to Contrastive 128-dim cluster labels (.npy)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Max samples for visualization (default: 5000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("4-MODEL VISUALIZATION COMPARISON")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Sample size: {args.sample_size}")
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_dict = {}
    labels_dict = {}
    
    try:
        print("  Standard VQ-VAE...")
        embeddings_dict['Standard VQ-VAE'] = load_embeddings(args.standard_embeddings)
        if args.standard_labels:
            labels = load_labels(args.standard_labels)
            if labels is not None:
                labels_dict['Standard VQ-VAE'] = labels
    except Exception as e:
        print(f"    Warning: {e}")
    
    try:
        print("  Masked VQ-VAE...")
        embeddings_dict['Masked VQ-VAE'] = load_embeddings(args.masked_embeddings)
        if args.masked_labels:
            labels = load_labels(args.masked_labels)
            if labels is not None:
                labels_dict['Masked VQ-VAE'] = labels
    except Exception as e:
        print(f"    Warning: {e}")
    
    try:
        print("  Contrastive VQ-VAE (64-dim)...")
        embeddings_dict['Contrastive VQ-VAE (64-dim)'] = load_embeddings(args.contrastive_64_embeddings)
        if args.contrastive_64_labels:
            labels = load_labels(args.contrastive_64_labels)
            if labels is not None:
                labels_dict['Contrastive VQ-VAE (64-dim)'] = labels
    except Exception as e:
        print(f"    Warning: {e}")
    
    try:
        print("  Contrastive VQ-VAE (128-dim)...")
        embeddings_dict['Contrastive VQ-VAE (128-dim)'] = load_embeddings(args.contrastive_128_embeddings)
        if args.contrastive_128_labels:
            labels = load_labels(args.contrastive_128_labels)
            if labels is not None:
                labels_dict['Contrastive VQ-VAE (128-dim)'] = labels
    except Exception as e:
        print(f"    Warning: {e}")
    
    print(f"\n✓ Loaded {len(embeddings_dict)} models")
    
    # Generate visualizations
    if HAS_UMAP:
        plot_4model_umap(
            embeddings_dict,
            labels_dict,
            output_dir / '4model_umap_comparison.png',
            sample_size=args.sample_size
        )
    
    plot_4model_tsne(
        embeddings_dict,
        labels_dict,
        output_dir / '4model_tsne_comparison.png',
        sample_size=args.sample_size
    )
    
    print("\n" + "="*80)
    print("✓ 4-MODEL VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    if HAS_UMAP:
        print(f"  • 4model_umap_comparison.png  - 4-model UMAP side-by-side")
    print(f"  • 4model_tsne_comparison.png  - 4-model t-SNE side-by-side")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
