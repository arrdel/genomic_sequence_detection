#!/usr/bin/env python3
"""
Generate Cluster Labels from Existing Embeddings

This script loads pre-computed embeddings and generates K-means cluster labels.
Useful when embeddings exist but cluster labels were not saved.

Usage:
    python scripts/generate_cluster_labels.py \
        --embeddings-dir experiments/4_final_comparison \
        --num-clusters 10
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(
        description='Generate cluster labels from existing embeddings'
    )
    parser.add_argument('--embeddings-dir', type=str, required=True,
                        help='Directory containing *_embeddings.npy files')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='Number of clusters for K-means (default: 10)')
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    
    print("="*80)
    print("GENERATING CLUSTER LABELS FROM EMBEDDINGS")
    print("="*80)
    print(f"\nEmbeddings directory: {embeddings_dir}")
    print(f"Number of clusters: {args.num_clusters}")
    print()
    
    # Find all embedding files
    embedding_files = list(embeddings_dir.glob('*_embeddings.npy'))
    
    if not embedding_files:
        print(f"ERROR: No *_embeddings.npy files found in {embeddings_dir}")
        return
    
    print(f"Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"  • {f.name}")
    print()
    
    # Process each embedding file
    for emb_file in embedding_files:
        base_name = emb_file.stem.replace('_embeddings', '')
        labels_file = embeddings_dir / f'{base_name}_labels.npy'
        
        print(f"Processing {base_name}...")
        
        # Load embeddings
        embeddings = np.load(emb_file)
        print(f"  Loaded embeddings: {embeddings.shape}")
        
        # Run K-means
        print(f"  Running K-means (k={args.num_clusters})...")
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Save labels
        np.save(labels_file, labels)
        print(f"  ✓ Saved labels: {labels_file.name}")
        print(f"    Label distribution: {np.bincount(labels)}")
        print()
    
    print("="*80)
    print("✓ CLUSTER LABELS GENERATED")
    print("="*80)
    print(f"\nGenerated files in {embeddings_dir}:")
    for emb_file in embedding_files:
        base_name = emb_file.stem.replace('_embeddings', '')
        print(f"  • {base_name}_labels.npy")
    print()


if __name__ == "__main__":
    main()
