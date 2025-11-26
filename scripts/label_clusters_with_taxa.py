#!/usr/bin/env python3
"""
Label Existing Clusters with Taxonomy

This script takes the cluster assignments from contrastive_evaluate.py and 
assigns biological labels based on Kraken2 taxonomic classifications.

Usage:
    python scripts/label_clusters_with_taxa.py \
      --embeddings-npy experiments/3_contrastive_vqvae/results/embeddings.npy \
      --cluster-labels-npy experiments/3_contrastive_vqvae/results/cluster_labels.npy \
      --data-path cleaned_reads.fastq \
      --kraken-classifications data/sequence_classification/kraken_classifications.txt \
      --kraken-report data/sequence_classification/kraken_report.txt \
      --output-dir experiments/3_contrastive_vqvae/results_labeled \
      --k-mer 6
"""

import os
import sys
import argparse
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from Bio import SeqIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import KmerTokenizer, FastqKmerDataset

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
        self._parse_kraken_report(kraken_report_path)
    
    def _parse_kraken_report(self, report_path):
        """Parse Kraken2 report file to build taxonomy mapping"""
        print(f"Parsing Kraken2 report: {report_path}")
        
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
        name = name.strip()
        
        if len(name) > max_length:
            if ' ' in name:
                parts = name.split()
                if len(parts) >= 2:
                    name = f"{parts[0][:1]}. {parts[1]}"
            
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


def assign_cluster_labels(cluster_labels, seq_ids, classifier, taxonomy):
    """Assign biological labels to clusters based on dominant taxonomy (classified sequences only)"""
    
    print("\nAssigning biological labels to clusters (based on classified sequences only)...")
    
    num_clusters = len(set(cluster_labels))
    cluster_info = {}
    
    for cluster_id in range(num_clusters):
        # Get sequences in this cluster
        mask = cluster_labels == cluster_id
        cluster_seq_ids = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
        
        # Get taxonomy for these sequences
        taxa_ids = [classifier.get_taxid(seq_id) for seq_id in cluster_seq_ids]
        taxa_counts = Counter(taxa_ids)
        
        total = len(taxa_ids)
        num_unclassified = taxa_counts.get(0, 0)
        num_classified = total - num_unclassified
        
        # Get top taxa (ONLY classified sequences)
        classified_taxa_counts = {tid: count for tid, count in taxa_counts.items() if tid != 0}
        
        top_taxa = []
        for taxid, count in Counter(classified_taxa_counts).most_common(10):
            name = taxonomy.get_short_name(taxid)
            rank = taxonomy.get_rank(taxid)
            # Percentage relative to TOTAL sequences (not just classified)
            percentage_of_total = 100 * count / total if total > 0 else 0
            # Percentage relative to classified sequences only
            percentage_of_classified = 100 * count / num_classified if num_classified > 0 else 0
            top_taxa.append({
                'taxid': int(taxid),
                'name': name,
                'rank': rank,
                'count': int(count),
                'percentage': float(percentage_of_total),
                'percentage_of_classified': float(percentage_of_classified)
            })
        
        # Determine cluster label based ONLY on classified sequences
        if top_taxa and num_classified > 0:
            dominant = top_taxa[0]
            # Use the dominant taxon name regardless of percentage
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
            'top_taxa': top_taxa,
            'num_unique_taxa': len(classified_taxa_counts),
            'entropy': float(compute_entropy(list(classified_taxa_counts.values()))) if classified_taxa_counts else 0.0
        }
        
        if num_classified > 0:
            print(f"  {cluster_name}: {total} seqs ({num_classified} classified, {100*num_classified/total:.1f}%)")
            if len(top_taxa) > 1:
                print(f"    - Dominant: {top_taxa[0]['name']} ({top_taxa[0]['percentage_of_classified']:.1f}% of classified)")
        else:
            print(f"  {cluster_name}: {total} seqs (0 classified)")
    
    return cluster_info


def plot_umap_labeled(embeddings, cluster_labels, cluster_info, out_png, figsize=(14, 10)):
    """Plot UMAP with biologically labeled clusters"""
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
            size = info.get('size', np.sum(mask))
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], 
                      c=[color], s=20, alpha=0.7, label=label)
    
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title("UMAP: Clusters with Biological Labels", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP plot: {out_png}")


def plot_tsne_labeled(embeddings, cluster_labels, cluster_info, out_png, figsize=(14, 10)):
    """Plot t-SNE with biologically labeled clusters"""
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
            size = info.get('size', np.sum(mask))
            
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                      c=[color], s=20, alpha=0.7, label=label)
    
    ax.set_xlabel("t-SNE-1", fontsize=12)
    ax.set_ylabel("t-SNE-2", fontsize=12)
    ax.set_title("t-SNE: Clusters with Biological Labels", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE plot: {out_png}")


def plot_cluster_taxonomy_bars(cluster_info, out_png, figsize=(14, 10)):
    """Plot bar charts showing taxonomic composition of each cluster (classified sequences only)"""
    
    num_clusters = len(cluster_info)
    ncols = min(3, num_clusters)
    nrows = (num_clusters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for cluster_id, info in cluster_info.items():
        ax = axes[cluster_id]
        
        top_taxa = info['top_taxa'][:5]
        if not top_taxa:
            ax.text(0.5, 0.5, 'No classified\nsequences', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{info['short_name']}\n(n={info['size']}, 0% classified)",
                        fontsize=9, fontweight='bold')
            ax.axis('off')
            continue
        
        names = [t['name'] for t in top_taxa]
        # Use percentage of classified sequences
        percentages = [t['percentage_of_classified'] for t in top_taxa]
        
        y_pos = np.arange(len(names))
        colors = ['steelblue' if i > 0 else 'darkblue' for i in range(len(names))]
        ax.barh(y_pos, percentages, color=colors, alpha=0.7)
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
    
    plt.suptitle('Taxonomic Composition of Clusters (Classified Sequences)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster taxonomy composition: {out_png}")


def main():
    parser = argparse.ArgumentParser(description='Label clusters with taxonomic information')
    parser.add_argument('--embeddings-npy', type=str, required=True,
                        help='Path to embeddings.npy from contrastive_evaluate.py')
    parser.add_argument('--cluster-labels-npy', type=str, default=None,
                        help='Path to cluster_labels.npy (if not provided, will be computed from embeddings)')
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
    parser.add_argument('--num-eval-samples', type=int, default=None,
                        help='Number of samples (must match original evaluation)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                        help='Figure size')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LABELING CLUSTERS WITH TAXONOMY")
    print("="*80)
    
    # Load embeddings and cluster labels
    print(f"\nLoading embeddings from: {args.embeddings_npy}")
    embeddings = np.load(args.embeddings_npy)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    if args.cluster_labels_npy:
        print(f"\nLoading cluster labels from: {args.cluster_labels_npy}")
        cluster_labels = np.load(args.cluster_labels_npy)
    else:
        print("\nNo cluster labels provided, computing K-means...")
        from sklearn.cluster import KMeans
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
    
    print(f"  Cluster labels shape: {cluster_labels.shape}")
    print(f"  Number of clusters: {len(set(cluster_labels))}")
    
    # Load taxonomy data
    print("\n" + "="*80)
    print("LOADING TAXONOMY DATA")
    print("="*80)
    
    taxonomy = TaxonomyMapper(args.kraken_report)
    classifier = SequenceClassificationLoader(args.kraken_classifications)
    
    # Load sequence IDs from FASTQ
    print(f"\nLoading sequence IDs from: {args.data_path}")
    tokenizer = KmerTokenizer(k=args.k_mer)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)
    
    # Extract sequence IDs (matching the number of embeddings)
    num_samples = len(embeddings)
    seq_ids = [dataset.records[i].id for i in range(min(num_samples, len(dataset.records)))]
    print(f"  Loaded {len(seq_ids)} sequence IDs")
    
    # Assign biological labels to clusters
    print("\n" + "="*80)
    print("ASSIGNING BIOLOGICAL LABELS")
    print("="*80)
    
    cluster_info = assign_cluster_labels(cluster_labels, seq_ids, classifier, taxonomy)
    
    # Print detailed cluster information
    print("\n" + "="*80)
    print("CLUSTER DETAILS")
    print("="*80)
    
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        print(f"\n{info['name']}:")
        print(f"  Total sequences: {info['size']}")
        print(f"  Classified: {info['num_classified']} ({info['classification_rate']*100:.1f}%)")
        print(f"  Unclassified: {info['num_unclassified']}")
        print(f"  Unique taxa: {info['num_unique_taxa']}")
        print(f"  Entropy: {info['entropy']:.2f}")
        if info['top_taxa']:
            print(f"  Top taxa (among classified sequences):")
            for taxon in info['top_taxa'][:5]:
                print(f"    - {taxon['name']} ({taxon['rank']}): {taxon['count']} seqs ({taxon['percentage_of_classified']:.1f}% of classified, {taxon['percentage']:.1f}% of total)")
        else:
            print(f"  No classified sequences in this cluster")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    umap_path = os.path.join(args.output_dir, 'embeddings_umap_labeled.png')
    plot_umap_labeled(embeddings, cluster_labels, cluster_info, umap_path, tuple(args.figsize))
    
    tsne_path = os.path.join(args.output_dir, 'embeddings_tsne_labeled.png')
    plot_tsne_labeled(embeddings, cluster_labels, cluster_info, tsne_path, tuple(args.figsize))
    
    taxonomy_path = os.path.join(args.output_dir, 'cluster_taxonomy_composition.png')
    plot_cluster_taxonomy_bars(cluster_info, taxonomy_path, tuple(args.figsize))
    
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
            'overall_classification_rate': float(sum(info['num_classified'] for info in cluster_info.values()) / len(cluster_labels))
        }
    }
    
    results_file = os.path.join(args.output_dir, 'cluster_labels_with_taxa.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {results_file}")
    
    print("\n" + "="*80)
    print("LABELING COMPLETE")
    print("="*80)
    print(f"Results directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
