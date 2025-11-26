#!/usr/bin/env python3
"""
Contrastive VQ-VAE Evaluation with Taxonomic Labels

This script evaluates a contrastive VQ-VAE model and uses Kraken2 taxonomic
classifications to label clusters with biological names instead of just digits.

Usage:
    python scripts/contrastive_evaluate_with_taxa.py \
      --checkpoint-path experiments/3_contrastive_vqvae/checkpoints/best_model.pt \
      --vqvae-checkpoint-path experiments/1_standard_vqvae/checkpoints/checkpoint_epoch_50.pt \
      --data-path /home/adelechinda/home/semester_projects/fall_25/deep_learning/project/data/cleaned_reads.fastq\
      --kraken-classifications data/sequence_classification/kraken_classifications.txt \
      --kraken-report data/sequence_classification/kraken_report.txt \
      --output-dir experiments/3_contrastive_vqvae/results_with_taxa
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
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import re

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


class TaxonomyMapper:
    """Maps Kraken2 taxonomic IDs to human-readable names"""
    
    def __init__(self, kraken_report_path):
        self.taxid_to_name = {}
        self.taxid_to_rank = {}
        self.taxid_to_level = {}
        self._parse_kraken_report(kraken_report_path)
    
    def _parse_kraken_report(self, report_path):
        """Parse Kraken2 report file to build taxonomy mapping"""
        print(f"Parsing Kraken2 report: {report_path}")
        
        with open(report_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                # Format: percent, clade_count, taxon_count, rank, taxid, name
                rank_code = parts[3]
                taxid = int(parts[4])
                name = parts[5].strip()
                
                # Store mapping
                self.taxid_to_name[taxid] = name
                self.taxid_to_rank[taxid] = rank_code
                
                # Determine hierarchical level from indentation
                level = (len(name) - len(name.lstrip())) // 2
                self.taxid_to_level[taxid] = level
        
        print(f"  Loaded {len(self.taxid_to_name)} taxonomic entries")
    
    def get_name(self, taxid, default="Unknown"):
        """Get taxonomic name for a given taxid"""
        if taxid == 0:
            return "Unclassified"
        return self.taxid_to_name.get(taxid, default)
    
    def get_rank(self, taxid):
        """Get rank code (S=species, G=genus, F=family, etc.)"""
        return self.taxid_to_rank.get(taxid, "U")
    
    def get_short_name(self, taxid, max_length=30):
        """Get shortened taxonomic name for visualization"""
        name = self.get_name(taxid)
        
        # Remove leading spaces (indentation)
        name = name.strip()
        
        # Shorten long names
        if len(name) > max_length:
            # Try to keep genus + species for species names
            if ' ' in name:
                parts = name.split()
                if len(parts) >= 2:
                    name = f"{parts[0][:1]}. {parts[1]}"
            
            # If still too long, truncate
            if len(name) > max_length:
                name = name[:max_length-3] + "..."
        
        return name


class SequenceClassificationLoader:
    """Loads and manages Kraken2 sequence classifications"""
    
    def __init__(self, kraken_classifications_path):
        self.seq_to_taxid = {}
        self.seq_to_conf = {}
        self._parse_classifications(kraken_classifications_path)
    
    def _parse_classifications(self, classifications_path):
        """Parse Kraken2 classifications file"""
        print(f"Loading Kraken2 classifications: {classifications_path}")
        
        count_classified = 0
        count_unclassified = 0
        
        with open(classifications_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                # Format: C/U, seq_id, taxid, length, kmer_mapping
                status = parts[0]
                seq_id = parts[1]
                taxid = int(parts[2])
                
                # Store classification
                self.seq_to_taxid[seq_id] = taxid
                
                if status == 'C':
                    count_classified += 1
                else:
                    count_unclassified += 1
        
        total = count_classified + count_unclassified
        print(f"  Loaded {total} sequence classifications")
        print(f"  Classified: {count_classified} ({100*count_classified/total:.1f}%)")
        print(f"  Unclassified: {count_unclassified} ({100*count_unclassified/total:.1f}%)")
    
    def get_taxid(self, seq_id):
        """Get taxonomic ID for a sequence"""
        # Handle different sequence ID formats
        # Remove /1, /2 suffixes if present
        base_id = seq_id.split()[0].rstrip('/12')
        return self.seq_to_taxid.get(base_id, 0)


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
    parser = argparse.ArgumentParser(description='Evaluate Contrastive VQ-VAE with Taxonomic Labels')
    
    # Data arguments
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to contrastive model checkpoint')
    parser.add_argument('--vqvae-checkpoint-path', type=str, required=True,
                        help='Path to base VQ-VAE checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the FASTQ file')
    parser.add_argument('--kraken-classifications', type=str, required=True,
                        help='Path to Kraken2 classifications file')
    parser.add_argument('--kraken-report', type=str, required=True,
                        help='Path to Kraken2 report file')
    parser.add_argument('--output-dir', type=str, default='./contrastive_evaluation_taxa',
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
    parser.add_argument('--num-eval-samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of clusters for K-means (default: 10, matching standard evaluation)')
    parser.add_argument('--min-taxa-count', type=int, default=10,
                        help='Minimum sequences per taxon to include in visualization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Visualization arguments
    parser.add_argument('--skip-umap', action='store_true',
                        help='Skip UMAP visualization')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE visualization')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                        help='Figure size for plots (width height)')
    
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
def extract_embeddings_with_ids(model, dataset, dataloader, device, max_samples=None):
    """Extract embeddings and corresponding sequence IDs"""
    model.eval()
    all_emb = []
    all_seq_ids = []
    num_samples = 0
    
    sample_idx = 0
    for batch_tokens, _ in tqdm(dataloader, desc="Extracting embeddings"):
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        z = model(batch_tokens)  # normalized [B, D]
        all_emb.append(z.cpu().numpy())
        
        # Get sequence IDs for this batch from dataset records
        batch_size = z.size(0)
        for i in range(batch_size):
            if sample_idx < len(dataset.records):
                # Extract sequence ID from BioPython SeqRecord
                seq_id = dataset.records[sample_idx].id
                all_seq_ids.append(seq_id)
            sample_idx += 1
        
        num_samples += batch_size
        if max_samples and num_samples >= max_samples:
            break
    
    embeddings = np.concatenate(all_emb, axis=0)
    if max_samples:
        embeddings = embeddings[:max_samples]
        all_seq_ids = all_seq_ids[:max_samples]
    
    return embeddings, all_seq_ids


def map_sequences_to_taxa(seq_ids, classifier, taxonomy):
    """Map sequence IDs to taxonomic labels"""
    taxa_ids = []
    taxa_names = []
    
    for seq_id in seq_ids:
        taxid = classifier.get_taxid(seq_id)
        taxa_ids.append(taxid)
        taxa_names.append(taxonomy.get_short_name(taxid))
    
    return np.array(taxa_ids), taxa_names


def analyze_cluster_taxonomy(cluster_labels, taxa_ids, taxonomy, min_count=5):
    """Analyze taxonomic composition of each cluster"""
    
    num_clusters = len(set(cluster_labels))
    cluster_taxonomy = {}
    
    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        cluster_taxa = taxa_ids[mask]
        
        # Count taxa in this cluster
        taxa_counts = Counter(cluster_taxa)
        total = len(cluster_taxa)
        
        # Get top taxa
        top_taxa = []
        for taxid, count in taxa_counts.most_common(10):
            if count >= min_count:
                name = taxonomy.get_short_name(taxid)
                rank = taxonomy.get_rank(taxid)
                percentage = 100 * count / total
                top_taxa.append({
                    'taxid': int(taxid),
                    'name': name,
                    'rank': rank,
                    'count': int(count),
                    'percentage': float(percentage)
                })
        
        # Determine dominant taxon
        if top_taxa:
            dominant = top_taxa[0]
            cluster_name = f"C{cluster_id}: {dominant['name']}"
            if dominant['percentage'] < 50:
                cluster_name += " (mixed)"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        cluster_taxonomy[cluster_id] = {
            'name': cluster_name,
            'size': int(total),
            'top_taxa': top_taxa,
            'num_unique_taxa': len(taxa_counts),
            'entropy': float(compute_entropy(list(taxa_counts.values())))
        }
    
    return cluster_taxonomy


def compute_entropy(counts):
    """Compute Shannon entropy of a distribution"""
    counts = np.array(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    
    probs = counts / total
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


def plot_umap_with_taxa(embeddings, taxa_names, taxa_ids, taxonomy, out_png, 
                        min_taxa_count=10, figsize=(14, 10)):
    """Plot UMAP visualization with taxonomic labels"""
    if not HAS_UMAP:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")
        return
    
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(embeddings)
    
    # Count taxa frequencies
    taxa_counts = Counter(taxa_ids)
    
    # Filter to common taxa for cleaner visualization
    common_taxa = {taxid for taxid, count in taxa_counts.items() if count >= min_taxa_count}
    
    # Create color mapping for common taxa
    unique_taxa = sorted(common_taxa)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_taxa))))
    taxa_to_color = {taxid: colors[i % len(colors)] for i, taxid in enumerate(unique_taxa)}
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot rare taxa in gray
    rare_mask = np.array([tid not in common_taxa for tid in taxa_ids])
    if rare_mask.any():
        ax.scatter(emb2d[rare_mask, 0], emb2d[rare_mask, 1], 
                  c='lightgray', s=10, alpha=0.3, label='Rare taxa')
    
    # Plot each common taxon
    legend_handles = []
    for taxid in unique_taxa:
        mask = taxa_ids == taxid
        if mask.any():
            color = taxa_to_color[taxid]
            name = taxonomy.get_short_name(taxid, max_length=25)
            count = taxa_counts[taxid]
            label = f"{name} (n={count})"
            
            scatter = ax.scatter(emb2d[mask, 0], emb2d[mask, 1], 
                               c=[color], s=20, alpha=0.6, label=label)
            legend_handles.append(scatter)
    
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title("UMAP: Contrastive Embeddings Colored by Taxonomy\n(Unclassified sequences excluded)", 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP plot with taxonomic labels: {out_png}")


def plot_tsne_with_taxa(embeddings, taxa_names, taxa_ids, taxonomy, out_png,
                        min_taxa_count=10, figsize=(14, 10)):
    """Plot t-SNE visualization with taxonomic labels"""
    if not HAS_TSNE:
        print("[WARN] scikit-learn not installed; skipping t-SNE plot.")
        return
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb2d = tsne.fit_transform(embeddings)
    
    # Count taxa frequencies
    taxa_counts = Counter(taxa_ids)
    
    # Filter to common taxa
    common_taxa = {taxid for taxid, count in taxa_counts.items() if count >= min_taxa_count}
    
    # Create color mapping
    unique_taxa = sorted(common_taxa)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_taxa))))
    taxa_to_color = {taxid: colors[i % len(colors)] for i, taxid in enumerate(unique_taxa)}
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot rare taxa in gray
    rare_mask = np.array([tid not in common_taxa for tid in taxa_ids])
    if rare_mask.any():
        ax.scatter(emb2d[rare_mask, 0], emb2d[rare_mask, 1],
                  c='lightgray', s=10, alpha=0.3, label='Rare taxa')
    
    # Plot each common taxon
    for taxid in unique_taxa:
        mask = taxa_ids == taxid
        if mask.any():
            color = taxa_to_color[taxid]
            name = taxonomy.get_short_name(taxid, max_length=25)
            count = taxa_counts[taxid]
            label = f"{name} (n={count})"
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                      c=[color], s=20, alpha=0.6, label=label)
    
    ax.set_xlabel("t-SNE-1", fontsize=12)
    ax.set_ylabel("t-SNE-2", fontsize=12)
    ax.set_title("t-SNE: Contrastive Embeddings Colored by Taxonomy\n(Unclassified sequences excluded)", 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE plot with taxonomic labels: {out_png}")


def plot_cluster_taxonomy_bars(cluster_taxonomy, out_png, figsize=(14, 10)):
    """Plot bar charts showing taxonomic composition of each cluster"""
    
    num_clusters = len(cluster_taxonomy)
    
    # Create subplots
    ncols = min(3, num_clusters)
    nrows = (num_clusters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for cluster_id, info in cluster_taxonomy.items():
        ax = axes[cluster_id]
        
        top_taxa = info['top_taxa'][:5]  # Top 5
        if not top_taxa:
            ax.text(0.5, 0.5, 'No taxa\nabove threshold', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Cluster {cluster_id}")
            ax.axis('off')
            continue
        
        names = [t['name'] for t in top_taxa]
        percentages = [t['percentage'] for t in top_taxa]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        ax.barh(y_pos, percentages, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Percentage (%)', fontsize=9)
        ax.set_title(f"{info['name']}\n(n={info['size']}, entropy={info['entropy']:.2f})",
                    fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Taxonomic Composition of Clusters', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster taxonomy bar chart: {out_png}")


def plot_umap_clusters_with_taxa(embeddings, cluster_labels, cluster_taxonomy, out_png, figsize=(14, 10)):
    """Plot UMAP visualization with clusters colored by cluster ID, labeled with dominant taxonomy"""
    if not HAS_UMAP:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")
        return
    
    print("Computing UMAP projection for cluster visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(embeddings)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use tab10 colormap for distinct colors
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    # Plot each cluster
    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        if mask.any():
            color = colors[cluster_id]
            cluster_info = cluster_taxonomy.get(cluster_id, {})
            label = cluster_info.get('name', f'Cluster {cluster_id}')
            size = cluster_info.get('size', np.sum(mask))
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], 
                      c=[color], s=20, alpha=0.7, label=f"{label} (n={size})")
    
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title("UMAP: Embedding Clusters with Taxonomic Labels\n(Unclassified sequences excluded)", 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP cluster plot with taxonomic labels: {out_png}")


def plot_tsne_clusters_with_taxa(embeddings, cluster_labels, cluster_taxonomy, out_png, figsize=(14, 10)):
    """Plot t-SNE visualization with clusters colored by cluster ID, labeled with dominant taxonomy"""
    if not HAS_TSNE:
        print("[WARN] scikit-learn not installed; skipping t-SNE plot.")
        return
    
    print("Computing t-SNE projection for cluster visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb2d = tsne.fit_transform(embeddings)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use tab10 colormap for distinct colors
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    # Plot each cluster
    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        if mask.any():
            color = colors[cluster_id]
            cluster_info = cluster_taxonomy.get(cluster_id, {})
            label = cluster_info.get('name', f'Cluster {cluster_id}')
            size = cluster_info.get('size', np.sum(mask))
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                      c=[color], s=20, alpha=0.7, label=f"{label} (n={size})")
    
    ax.set_xlabel("t-SNE-1", fontsize=12)
    ax.set_ylabel("t-SNE-2", fontsize=12)
    ax.set_title("t-SNE: Embedding Clusters with Taxonomic Labels\n(Unclassified sequences excluded)", 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE cluster plot with taxonomic labels: {out_png}")


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
    print("CONTRASTIVE VQ-VAE EVALUATION WITH TAXONOMIC LABELS")
    print("="*80)
    
    # Load taxonomy mapping
    print("\n" + "="*80)
    print("LOADING TAXONOMY DATA")
    print("="*80)
    
    taxonomy = TaxonomyMapper(args.kraken_report)
    classifier = SequenceClassificationLoader(args.kraken_classifications)
    
    # Load VQ-VAE checkpoint
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
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
    
    # Create tokenizer
    tokenizer = KmerTokenizer(k=args.k_mer, use_canonical=args.use_canonical)
    
    PAD_ID = tokenizer.pad_id
    VOCAB_SIZE = len(tokenizer.stoi)
    
    print(f"\nTokenizer:")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  PAD ID: {PAD_ID}")
    
    # Initialize base VQ-VAE
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
    
    # Load contrastive checkpoint
    print(f"\nLoading contrastive model from: {args.checkpoint_path}")
    contrastive_checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get projection dim
    if 'args' in contrastive_checkpoint:
        proj_dim = contrastive_checkpoint['args'].get('proj_dim', 64)
    else:
        proj_dim = 64
    
    # Build contrastive model
    contrastive_model = ContrastiveHead(
        encoder=base_model.encoder,
        embed_dim=code_dim,
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
    print(f"\nDataset: {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Extract embeddings with sequence IDs
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    
    embeddings, seq_ids = extract_embeddings_with_ids(
        contrastive_model,
        dataset,
        dataloader,
        device,
        max_samples=args.num_eval_samples
    )
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Number of sequence IDs: {len(seq_ids)}")
    
    # Map sequences to taxa
    print("\n" + "="*80)
    print("MAPPING SEQUENCES TO TAXONOMY")
    print("="*80)
    
    taxa_ids, taxa_names = map_sequences_to_taxa(seq_ids, classifier, taxonomy)
    
    # Count taxonomic distribution (before filtering)
    taxa_counts_all = Counter(taxa_ids)
    num_unclassified = taxa_counts_all.get(0, 0)
    
    print(f"\nTaxonomic distribution (before filtering):")
    print(f"  Total sequences: {len(taxa_ids)}")
    print(f"  Unique taxa: {len(taxa_counts_all)}")
    print(f"  Unclassified: {num_unclassified} ({100*num_unclassified/len(taxa_ids):.1f}%)")
    
    # Filter out unclassified sequences
    print(f"\nFiltering out unclassified sequences...")
    classified_mask = taxa_ids != 0
    
    embeddings_classified = embeddings[classified_mask]
    taxa_ids_classified = taxa_ids[classified_mask]
    taxa_names_classified = [taxa_names[i] for i in range(len(taxa_names)) if classified_mask[i]]
    seq_ids_classified = [seq_ids[i] for i in range(len(seq_ids)) if classified_mask[i]]
    
    print(f"  Retained {len(embeddings_classified)} classified sequences ({100*len(embeddings_classified)/len(embeddings):.1f}%)")
    print(f"  Excluded {num_unclassified} unclassified sequences ({100*num_unclassified/len(embeddings):.1f}%)")
    
    # Update working variables to use only classified sequences
    embeddings = embeddings_classified
    taxa_ids = taxa_ids_classified
    taxa_names = taxa_names_classified
    seq_ids = seq_ids_classified
    
    # Count taxonomic distribution (after filtering)
    taxa_counts = Counter(taxa_ids)
    print(f"\nTaxonomic distribution (after filtering):")
    print(f"  Total classified sequences: {len(taxa_ids)}")
    print(f"  Unique taxa: {len(taxa_counts)}")
    
    print(f"\nTop 10 taxa:")
    for taxid, count in taxa_counts.most_common(10):
        name = taxonomy.get_short_name(taxid)
        rank = taxonomy.get_rank(taxid)
        print(f"  {name} ({rank}): {count} ({100*count/len(taxa_ids):.1f}%)")
    
    # Perform clustering
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS")
    print("="*80)
    
    print(f"Running K-means with {args.num_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute clustering metrics
    silhouette = silhouette_score(embeddings, cluster_labels, sample_size=min(5000, len(embeddings)))
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
    
    print(f"\nClustering metrics:")
    print(f"  Silhouette score: {silhouette:.4f}")
    print(f"  Davies-Bouldin index: {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz index: {calinski_harabasz:.2f}")
    
    # Analyze cluster taxonomy
    print("\n" + "="*80)
    print("CLUSTER TAXONOMY ANALYSIS")
    print("="*80)
    
    cluster_taxonomy = analyze_cluster_taxonomy(cluster_labels, taxa_ids, taxonomy, min_count=5)
    
    for cluster_id, info in cluster_taxonomy.items():
        print(f"\n{info['name']}:")
        print(f"  Size: {info['size']} sequences")
        print(f"  Unique taxa: {info['num_unique_taxa']}")
        print(f"  Entropy: {info['entropy']:.2f}")
        print(f"  Top taxa:")
        for taxon in info['top_taxa'][:5]:
            print(f"    - {taxon['name']} ({taxon['rank']}): {taxon['count']} ({taxon['percentage']:.1f}%)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Cluster-based visualizations (matching standard contrastive_evaluate.py)
    if not args.skip_umap:
        umap_clusters_path = os.path.join(args.output_dir, 'embeddings_umap_clusters.png')
        plot_umap_clusters_with_taxa(embeddings, cluster_labels, cluster_taxonomy, umap_clusters_path,
                                     figsize=tuple(args.figsize))
    
    if not args.skip_tsne:
        tsne_clusters_path = os.path.join(args.output_dir, 'embeddings_tsne_clusters.png')
        plot_tsne_clusters_with_taxa(embeddings, cluster_labels, cluster_taxonomy, tsne_clusters_path,
                                     figsize=tuple(args.figsize))
    
    # 2. Taxonomy-based visualizations (color by taxonomy instead of cluster)
    if not args.skip_umap:
        umap_path = os.path.join(args.output_dir, 'embeddings_umap_taxa.png')
        plot_umap_with_taxa(embeddings, taxa_names, taxa_ids, taxonomy, umap_path,
                           min_taxa_count=args.min_taxa_count, figsize=tuple(args.figsize))
    
    if not args.skip_tsne:
        tsne_path = os.path.join(args.output_dir, 'embeddings_tsne_taxa.png')
        plot_tsne_with_taxa(embeddings, taxa_names, taxa_ids, taxonomy, tsne_path,
                           min_taxa_count=args.min_taxa_count, figsize=tuple(args.figsize))
    
    # 3. Cluster taxonomy composition bars
    taxonomy_bars_path = os.path.join(args.output_dir, 'cluster_taxonomy_composition.png')
    plot_cluster_taxonomy_bars(cluster_taxonomy, taxonomy_bars_path, figsize=tuple(args.figsize))
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save embeddings and labels
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'cluster_labels.npy'), cluster_labels)
    np.save(os.path.join(args.output_dir, 'taxa_ids.npy'), taxa_ids)
    
    with open(os.path.join(args.output_dir, 'seq_ids.txt'), 'w') as f:
        f.write('\n'.join(seq_ids))
    
    with open(os.path.join(args.output_dir, 'taxa_names.txt'), 'w') as f:
        f.write('\n'.join(taxa_names))
    
    # Save comprehensive results JSON
    results = {
        'checkpoint_path': args.checkpoint_path,
        'vqvae_checkpoint_path': args.vqvae_checkpoint_path,
        'data_path': args.data_path,
        'kraken_classifications': args.kraken_classifications,
        'kraken_report': args.kraken_report,
        'evaluation_config': {
            'num_eval_samples': args.num_eval_samples,
            'batch_size': args.batch_size,
            'num_clusters': args.num_clusters,
            'min_taxa_count': args.min_taxa_count,
            'seed': args.seed
        },
        'clustering_metrics': {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'calinski_harabasz_index': float(calinski_harabasz)
        },
        'taxonomy_stats': {
            'total_sequences_evaluated': args.num_eval_samples,
            'total_classified_sequences': len(taxa_ids),
            'total_unclassified_sequences': int(num_unclassified),
            'unclassified_percentage': float(100 * num_unclassified / args.num_eval_samples),
            'unique_taxa': len(taxa_counts),
            'top_taxa': [
                {
                    'taxid': int(taxid),
                    'name': taxonomy.get_short_name(taxid),
                    'rank': taxonomy.get_rank(taxid),
                    'count': int(count),
                    'percentage': float(100 * count / len(taxa_ids))
                }
                for taxid, count in taxa_counts.most_common(20)
            ]
        },
        'cluster_taxonomy': cluster_taxonomy,
        'model_config': {
            'vocab_size': VOCAB_SIZE,
            'num_codes': num_codes,
            'code_dim': code_dim,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'proj_dim': proj_dim
        }
    }
    
    results_file = os.path.join(args.output_dir, 'evaluation_results_with_taxa.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved evaluation results: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total sequences evaluated: {args.num_eval_samples}")
    print(f"Classified sequences analyzed: {len(embeddings)} ({100*len(embeddings)/args.num_eval_samples:.1f}%)")
    print(f"Unclassified sequences excluded: {num_unclassified} ({100*num_unclassified/args.num_eval_samples:.1f}%)")
    print(f"Found {len(taxa_counts)} unique taxa")
    print(f"Identified {args.num_clusters} clusters")
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
