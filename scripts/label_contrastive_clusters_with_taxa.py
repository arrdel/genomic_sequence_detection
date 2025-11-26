#!/usr/bin/env python3
"""
Label Contrastive VQ-VAE Clusters with Taxonomy (Genus Level)

This script takes the cluster assignments from contrastive VQ-VAE evaluation and 
assigns biological labels at the GENUS level based on Kraken2 taxonomic classifications.

Usage:
    python scripts/label_contrastive_clusters_with_taxa.py \
      --checkpoint-path experiments/3_contrastive_vqvae/checkpoints/best_model.pt \
      --vqvae-checkpoint-path experiments/1_standard_vqvae/checkpoints/best_model.pt \
      --data-path cleaned_reads.fastq \
      --kraken-classifications data/sequence_classification/kraken_classifications.txt \
      --kraken-report data/sequence_classification/kraken_report.txt \
      --output-dir experiments/3_contrastive_vqvae/results_genus_labeled \
      --k-mer 6 \
      --num-eval-samples 10000 \
      --num-clusters 10
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset
from src.models import VQVAE
from sklearn.cluster import KMeans

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


class TaxonomyMapper:
    """Maps Kraken2 taxonomic IDs to human-readable names at genus level"""
    
    def __init__(self, kraken_report_path):
        self.taxid_to_name = {}
        self.taxid_to_rank = {}
        self.taxid_to_genus = {}  # Map any taxid to its genus
        self._parse_kraken_report(kraken_report_path)
    
    def _parse_kraken_report(self, report_path):
        """Parse Kraken2 report file to build taxonomy mapping"""
        print(f"Parsing Kraken2 report: {report_path}")
        
        # First pass: collect all taxa
        taxa_hierarchy = {}
        
        with open(report_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                rank_code = parts[3]
                taxid = int(parts[4])
                name = parts[5].strip()
                
                self.taxid_to_name[taxid] = name
                self.taxid_to_rank[taxid] = rank_code
                
                # Store for hierarchy building
                taxa_hierarchy[taxid] = {
                    'name': name,
                    'rank': rank_code,
                    'parent': None  # Will be filled if needed
                }
        
        print(f"  Loaded {len(self.taxid_to_name)} taxonomic entries")
        
        # Second pass: map species to genus by parsing the report structure
        # In Kraken2 report, genus comes before species
        current_genus_taxid = None
        current_genus_name = None
        
        with open(report_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                rank_code = parts[3]
                taxid = int(parts[4])
                name = parts[5].strip()
                
                if rank_code == 'G':  # Genus
                    current_genus_taxid = taxid
                    current_genus_name = name
                    self.taxid_to_genus[taxid] = taxid  # Genus maps to itself
                elif rank_code == 'S':  # Species
                    # Map species to its genus
                    if current_genus_taxid is not None:
                        self.taxid_to_genus[taxid] = current_genus_taxid
                elif rank_code in ['D', 'P', 'C', 'O', 'F']:
                    # Higher ranks reset genus tracking
                    current_genus_taxid = None
                    current_genus_name = None
        
        print(f"  Mapped {len(self.taxid_to_genus)} taxa to genus level")
    
    def get_genus_taxid(self, taxid):
        """Get the genus-level taxid for any taxid"""
        if taxid == 0:
            return 0
        
        # If already at genus level
        if self.taxid_to_rank.get(taxid) == 'G':
            return taxid
        
        # Look up genus mapping
        genus_taxid = self.taxid_to_genus.get(taxid)
        if genus_taxid is not None:
            return genus_taxid
        
        # If no genus found, return original (might be higher rank)
        return taxid
    
    def get_name(self, taxid, default="Unknown"):
        """Get taxonomic name for a given taxid"""
        if taxid == 0:
            return "Unclassified"
        return self.taxid_to_name.get(taxid, default)
    
    def get_rank(self, taxid):
        """Get rank code (S=species, G=genus, F=family, etc.)"""
        return self.taxid_to_rank.get(taxid, "U")
    
    def get_genus_name(self, taxid):
        """Get genus name for any taxid"""
        genus_taxid = self.get_genus_taxid(taxid)
        return self.get_name(genus_taxid)
    
    def get_short_name(self, taxid, max_length=30):
        """Get shortened taxonomic name for visualization"""
        name = self.get_name(taxid)
        name = name.strip()
        
        if len(name) > max_length:
            name = name[:max_length-3] + "..."
        
        return name


class SequenceClassificationLoader:
    """Loads and manages Kraken2 sequence classifications"""
    
    def __init__(self, kraken_classifications_path):
        self.seq_to_taxid = {}
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
                
                status = parts[0]
                seq_id = parts[1]
                taxid = int(parts[2])
                
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
        base_id = seq_id.split()[0].rstrip('/12')
        return self.seq_to_taxid.get(base_id, 0)


def compute_entropy(counts):
    """Compute Shannon entropy of a distribution"""
    counts = np.array(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def collate_fn(batch):
    """Collate function for DataLoader"""
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([b[1] for b in batch])
    return tokens, lengths


@torch.no_grad()
def extract_embeddings_and_cluster(model, dataloader, device, num_clusters=10):
    """Extract embeddings from contrastive VQ-VAE and cluster them"""
    model.eval()
    all_embeddings = []
    
    print("\nExtracting embeddings from Contrastive VQ-VAE...")
    for batch_tokens, _ in tqdm(dataloader, desc="Extracting embeddings"):
        batch_tokens = batch_tokens.to(device)
        
        # Get normalized embeddings
        z_norm = model(batch_tokens)  # [B, proj_dim]
        
        all_embeddings.append(z_norm.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Cluster embeddings
    print(f"\nClustering embeddings with K-means (k={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return embeddings, cluster_labels


def assign_genus_labels_to_clusters(cluster_labels, seq_ids, classifier, taxonomy):
    """Assign genus-level biological labels to clusters"""
    
    print("\nAssigning genus-level biological labels to clusters...")
    
    num_clusters = len(set(cluster_labels))
    cluster_info = {}
    
    for cluster_id in range(num_clusters):
        # Get sequences in this cluster
        mask = cluster_labels == cluster_id
        cluster_seq_ids = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
        
        # Get taxonomy for these sequences
        taxa_ids = [classifier.get_taxid(seq_id) for seq_id in cluster_seq_ids]
        
        # Map to genus level
        genus_ids = [taxonomy.get_genus_taxid(tid) for tid in taxa_ids]
        genus_counts = Counter(genus_ids)
        
        total = len(genus_ids)
        num_unclassified = genus_counts.get(0, 0)
        num_classified = total - num_unclassified
        
        # Get top genera (ONLY classified sequences)
        classified_genus_counts = {gid: count for gid, count in genus_counts.items() if gid != 0}
        
        top_genera = []
        for genus_id, count in Counter(classified_genus_counts).most_common(10):
            name = taxonomy.get_genus_name(genus_id)
            rank = taxonomy.get_rank(genus_id)
            # Percentage relative to TOTAL sequences
            percentage_of_total = 100 * count / total if total > 0 else 0
            # Percentage relative to classified sequences only
            percentage_of_classified = 100 * count / num_classified if num_classified > 0 else 0
            top_genera.append({
                'taxid': int(genus_id),
                'name': name,
                'rank': rank,
                'count': int(count),
                'percentage': float(percentage_of_total),
                'percentage_of_classified': float(percentage_of_classified)
            })
        
        # Determine cluster label based ONLY on classified sequences at genus level
        if top_genera and num_classified > 0:
            dominant = top_genera[0]
            # Use the dominant genus name
            cluster_name = f"C{cluster_id}: {dominant['name']}"
            short_name = f"{dominant['name']}"
        else:
            # No classified sequences in this cluster
            cluster_name = f"C{cluster_id}: No classification"
            short_name = "No classification"
        
        cluster_info[cluster_id] = {
            'name': cluster_name,
            'short_name': short_name,
            'size': int(total),
            'num_classified': int(num_classified),
            'num_unclassified': int(num_unclassified),
            'classification_rate': float(num_classified / total) if total > 0 else 0.0,
            'top_genera': top_genera,
            'num_unique_genera': len(classified_genus_counts),
            'entropy': float(compute_entropy(list(classified_genus_counts.values()))) if classified_genus_counts else 0.0
        }
        
        if num_classified > 0:
            print(f"  {cluster_name}: {total} seqs ({num_classified} classified, {100*num_classified/total:.1f}%)")
            if len(top_genera) > 1:
                print(f"    - Dominant: {top_genera[0]['name']} ({top_genera[0]['percentage_of_classified']:.1f}% of classified)")
        else:
            print(f"  {cluster_name}: {total} seqs (0 classified)")
    
    return cluster_info


def plot_umap_labeled(embeddings, cluster_labels, cluster_info, out_png, figsize=(14, 10)):
    """Plot UMAP with genus-level labeled clusters"""
    if not HAS_UMAP:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")
        return
    
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        if mask.any():
            color = colors[cluster_id]
            info = cluster_info.get(cluster_id, {})
            label = info.get('name', f'Cluster {cluster_id}')
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], 
                      c=[color], s=20, alpha=0.7, label=label)
    
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title("Contrastive VQ-VAE: UMAP with Genus-Level Labels", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP plot: {out_png}")


def plot_tsne_labeled(embeddings, cluster_labels, cluster_info, out_png, figsize=(14, 10)):
    """Plot t-SNE with genus-level labeled clusters"""
    if not HAS_TSNE:
        print("[WARN] scikit-learn not installed; skipping t-SNE plot.")
        return
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    num_clusters = len(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        if mask.any():
            color = colors[cluster_id]
            info = cluster_info.get(cluster_id, {})
            label = info.get('name', f'Cluster {cluster_id}')
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                      c=[color], s=20, alpha=0.7, label=label)
    
    ax.set_xlabel("t-SNE-1", fontsize=12)
    ax.set_ylabel("t-SNE-2", fontsize=12)
    ax.set_title("Contrastive VQ-VAE: t-SNE with Genus-Level Labels", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE plot: {out_png}")


def plot_cluster_genus_composition(cluster_info, out_png, figsize=(14, 10)):
    """Plot bar charts showing genus composition of each cluster"""
    
    num_clusters = len(cluster_info)
    ncols = min(3, num_clusters)
    nrows = (num_clusters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for cluster_id, info in cluster_info.items():
        ax = axes[cluster_id]
        
        top_genera = info['top_genera'][:5]
        if not top_genera:
            ax.text(0.5, 0.5, 'No classified\nsequences', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{info['short_name']}\n(n={info['size']}, 0% classified)",
                        fontsize=9, fontweight='bold')
            ax.axis('off')
            continue
        
        names = [g['name'] for g in top_genera]
        # Use percentage of classified sequences
        percentages = [g['percentage_of_classified'] for g in top_genera]
        
        y_pos = np.arange(len(names))
        colors_bar = ['darkblue' if i == 0 else 'steelblue' for i in range(len(names))]
        ax.barh(y_pos, percentages, color=colors_bar, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('% of classified seqs', fontsize=9)
        ax.set_title(f"{info['short_name']}\n(n={info['size']}, {info['classification_rate']*100:.0f}% classified)",
                    fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)
    
    for i in range(num_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Genus-Level Composition of Contrastive VQ-VAE Clusters', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster genus composition: {out_png}")


def main():
    parser = argparse.ArgumentParser(description='Label Contrastive VQ-VAE clusters with genus-level taxonomy')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to contrastive VQ-VAE checkpoint')
    parser.add_argument('--vqvae-checkpoint-path', type=str, required=True,
                        help='Path to base VQ-VAE checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to FASTQ file')
    parser.add_argument('--kraken-classifications', type=str, required=True,
                        help='Path to Kraken2 classifications file')
    parser.add_argument('--kraken-report', type=str, required=True,
                        help='Path to Kraken2 report file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--k-mer', type=int, default=6,
                        help='K-mer size')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum sequence length')
    parser.add_argument('--num-eval-samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of K-means clusters')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                        help='Figure size')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU {args.gpu_id}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CONTRASTIVE VQ-VAE: GENUS-LEVEL TAXONOMIC LABELING")
    print("="*80)
    
    # Load taxonomy data
    print("\n" + "="*80)
    print("LOADING TAXONOMY DATA")
    print("="*80)
    
    taxonomy = TaxonomyMapper(args.kraken_report)
    classifier = SequenceClassificationLoader(args.kraken_classifications)
    
    # Load base VQ-VAE
    print("\n" + "="*80)
    print("LOADING BASE VQ-VAE MODEL")
    print("="*80)
    
    print(f"\nLoading base VQ-VAE checkpoint: {args.vqvae_checkpoint_path}")
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint_path, map_location=device)
    
    # Get config
    if 'args' in vqvae_checkpoint:
        model_args = vqvae_checkpoint['args']
        vocab_size = model_args.get('vocab_size', 4097)
        pad_id = model_args.get('pad_id', 4096)
        num_codes = model_args.get('num_codes', 512)
        code_dim = model_args.get('code_dim', 128)
        embed_dim = model_args.get('embed_dim', 128)
        hidden_dim = model_args.get('hidden_dim', 256)
        commitment_cost = model_args.get('commitment_cost', 0.1)
    else:
        vocab_size, pad_id = 4097, 4096
        num_codes, code_dim = 512, 128
        embed_dim, hidden_dim = 128, 256
        commitment_cost = 0.1
    
    print(f"  Model config: vocab={vocab_size}, codes={num_codes}, dim={code_dim}")
    
    # Create base VQ-VAE
    base_model = VQVAE(
        vocab_size, pad_id,
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
    
    print("✓ Base VQ-VAE loaded successfully")
    
    # Load contrastive checkpoint
    print(f"\nLoading contrastive checkpoint: {args.checkpoint_path}")
    contrastive_checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if 'args' in contrastive_checkpoint:
        proj_dim = contrastive_checkpoint['args'].get('proj_dim', 64)
    else:
        proj_dim = 64
    
    print(f"  Projection dim: {proj_dim}")
    
    # Create contrastive head
    model = ContrastiveHead(
        encoder=base_model.encoder,
        embed_dim=code_dim,
        proj_dim=proj_dim
    ).to(device)
    
    # Load contrastive weights
    state_dict = contrastive_checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✓ Contrastive model loaded successfully")
    
    # Load dataset
    print(f"\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    print(f"\nLoading sequences from: {args.data_path}")
    tokenizer = KmerTokenizer(k=args.k_mer)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=args.max_seq_length)
    
    # Limit samples
    if args.num_eval_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(args.num_eval_samples))
        dataset = Subset(dataset, indices)
    
    print(f"  Using {len(dataset)} sequences")
    
    # Extract sequence IDs
    if hasattr(dataset, 'records'):
        seq_ids = [dataset.records[i].id for i in range(len(dataset))]
    else:
        # If using Subset
        seq_ids = [dataset.dataset.records[i].id for i in dataset.indices]
    
    print(f"  Extracted {len(seq_ids)} sequence IDs")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Extract embeddings and cluster
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS & CLUSTERING")
    print("="*80)
    
    embeddings, cluster_labels = extract_embeddings_and_cluster(
        model, dataloader, device, num_clusters=args.num_clusters
    )
    
    # Save embeddings and labels
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'cluster_labels.npy'), cluster_labels)
    print(f"\n✓ Saved embeddings and cluster labels")
    
    # Assign genus-level labels
    print("\n" + "="*80)
    print("ASSIGNING GENUS-LEVEL LABELS")
    print("="*80)
    
    cluster_info = assign_genus_labels_to_clusters(cluster_labels, seq_ids, classifier, taxonomy)
    
    # Print detailed cluster information
    print("\n" + "="*80)
    print("CLUSTER DETAILS (GENUS LEVEL)")
    print("="*80)
    
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        print(f"\n{info['name']}:")
        print(f"  Total sequences: {info['size']}")
        print(f"  Classified: {info['num_classified']} ({info['classification_rate']*100:.1f}%)")
        print(f"  Unclassified: {info['num_unclassified']}")
        print(f"  Unique genera: {info['num_unique_genera']}")
        print(f"  Entropy: {info['entropy']:.2f}")
        if info['top_genera']:
            print(f"  Top genera (among classified sequences):")
            for genus in info['top_genera'][:5]:
                print(f"    - {genus['name']}: {genus['count']} seqs "
                      f"({genus['percentage_of_classified']:.1f}% of classified, "
                      f"{genus['percentage']:.1f}% of total)")
        else:
            print(f"  No classified sequences in this cluster")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    umap_path = os.path.join(args.output_dir, 'embeddings_umap_genus_labeled.png')
    plot_umap_labeled(embeddings, cluster_labels, cluster_info, umap_path, tuple(args.figsize))
    
    tsne_path = os.path.join(args.output_dir, 'embeddings_tsne_genus_labeled.png')
    plot_tsne_labeled(embeddings, cluster_labels, cluster_info, tsne_path, tuple(args.figsize))
    
    composition_path = os.path.join(args.output_dir, 'cluster_genus_composition.png')
    plot_cluster_genus_composition(cluster_info, composition_path, tuple(args.figsize))
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results = {
        'cluster_labels': cluster_info,
        'summary': {
            'num_clusters': len(cluster_info),
            'total_sequences': int(len(cluster_labels)),
            'avg_cluster_size': float(np.mean([info['size'] for info in cluster_info.values()])),
            'overall_classification_rate': float(sum(info['num_classified'] for info in cluster_info.values()) / len(cluster_labels)),
            'taxonomy_level': 'genus'
        }
    }
    
    results_file = os.path.join(args.output_dir, 'cluster_genus_labels.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {results_file}")
    
    print("\n" + "="*80)
    print("GENUS-LEVEL LABELING COMPLETE")
    print("="*80)
    print(f"Results directory: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  • embeddings.npy")
    print(f"  • cluster_labels.npy")
    print(f"  • embeddings_umap_genus_labeled.png")
    print(f"  • embeddings_tsne_genus_labeled.png")
    print(f"  • cluster_genus_composition.png")
    print(f"  • cluster_genus_labels.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
