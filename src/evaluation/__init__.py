#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework

Evaluates all models (VQ-VAE variants + baselines) on multiple downstream tasks:
1. Clustering quality (K-means, DBSCAN, Hierarchical)
2. Linear probing classification
3. Sequence retrieval (nearest neighbor)  
4. Anomaly/novelty detection
5. Ablation studies

Reports mean ± std over multiple random seeds for statistical rigor.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# =============================================================================
# Clustering Evaluation
# =============================================================================

def evaluate_clustering(
    embeddings: np.ndarray,
    n_clusters_list: List[int] = [5, 10, 15, 20],
    methods: List[str] = ["kmeans", "agglomerative"],
    random_seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Comprehensive clustering evaluation over multiple k values and seeds.
    
    Returns dict with mean ± std for each metric.
    """
    results = defaultdict(lambda: defaultdict(list))
    
    for method in methods:
        for n_clusters in n_clusters_list:
            for seed in random_seeds:
                if method == "kmeans":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
                elif method == "agglomerative":
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    continue
                
                labels = clusterer.fit_predict(embeddings)
                
                # Skip if degenerate clustering
                n_unique = len(set(labels))
                if n_unique < 2:
                    continue
                
                sil = silhouette_score(embeddings, labels)
                db = davies_bouldin_score(embeddings, labels)
                ch = calinski_harabasz_score(embeddings, labels)
                
                key = f"{method}_k{n_clusters}"
                results[key]["silhouette"].append(sil)
                results[key]["davies_bouldin"].append(db)
                results[key]["calinski_harabasz"].append(ch)
    
    # Compute mean ± std
    summary = {}
    for key, metrics in results.items():
        summary[key] = {}
        for metric_name, values in metrics.items():
            summary[key][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    
    return summary


# =============================================================================
# Linear Probing
# =============================================================================

def evaluate_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifiers: List[str] = ["logistic", "knn", "svm"],
    n_folds: int = 5,
    random_seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Linear probing evaluation with cross-validation.
    
    Uses cluster assignments as pseudo-labels if no true labels available.
    """
    results = defaultdict(lambda: defaultdict(list))
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    
    for clf_name in classifiers:
        for seed in random_seeds:
            if clf_name == "logistic":
                clf = LogisticRegression(
                    max_iter=2000, random_state=seed
                )
            elif clf_name == "knn":
                clf = KNeighborsClassifier(n_neighbors=5)
            elif clf_name == "svm":
                clf = LinearSVC(max_iter=2000, random_state=seed)
            else:
                continue
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            # Accuracy
            acc_scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
            results[clf_name]["accuracy"].append(float(acc_scores.mean()))
            
            # F1 macro
            f1_scores = cross_val_score(clf, X, labels, cv=cv, scoring="f1_macro")
            results[clf_name]["f1_macro"].append(float(f1_scores.mean()))
    
    # Summarize
    summary = {}
    for clf_name, metrics in results.items():
        summary[clf_name] = {}
        for metric_name, values in metrics.items():
            summary[clf_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    
    return summary


# =============================================================================
# Sequence Retrieval
# =============================================================================

def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
    metric: str = "cosine",
) -> Dict:
    """
    Evaluate nearest-neighbor retrieval quality.
    
    For each query, check if the k-nearest neighbors share the same label.
    Reports Precision@k and MAP@k.
    """
    # Compute pairwise distances
    dists = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(dists, np.inf)  # exclude self
    
    results = {}
    for k in k_values:
        precisions = []
        for i in range(len(embeddings)):
            # Get k-nearest neighbors
            nn_indices = np.argsort(dists[i])[:k]
            nn_labels = labels[nn_indices]
            
            # Precision@k = fraction of neighbors with same label
            precision = (nn_labels == labels[i]).sum() / k
            precisions.append(precision)
        
        results[f"precision@{k}"] = {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
        }
    
    return results


# =============================================================================
# Anomaly Detection
# =============================================================================

def evaluate_anomaly_detection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    contamination_ratio: float = 0.1,
) -> Dict:
    """
    Evaluate anomaly detection capability.
    
    Simulates anomaly detection by treating the smallest cluster(s) as 
    anomalous and measuring if the model's embedding space separates them.
    Uses distance-to-centroid as anomaly score.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Use K-means to find clusters
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute distances to nearest centroid
    distances = kmeans.transform(embeddings).min(axis=1)
    
    # Label smallest cluster as anomalous
    cluster_sizes = np.bincount(cluster_labels)
    anomaly_cluster = np.argmin(cluster_sizes)
    
    is_anomaly = (cluster_labels == anomaly_cluster).astype(int)
    
    if is_anomaly.sum() == 0 or is_anomaly.sum() == len(is_anomaly):
        return {"note": "Cannot evaluate: degenerate clustering"}
    
    auroc = roc_auc_score(is_anomaly, distances)
    auprc = average_precision_score(is_anomaly, distances)
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "anomaly_fraction": float(is_anomaly.mean()),
    }


# =============================================================================
# Embedding Quality Metrics
# =============================================================================

def evaluate_embedding_quality(
    embeddings: np.ndarray,
) -> Dict:
    """
    Compute intrinsic embedding quality metrics:
    - Alignment: average distance between augmented pairs (if available)
    - Uniformity: how uniformly distributed embeddings are on unit sphere
    - Isotropy: whether all dimensions are equally utilized
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normalized = embeddings / (norms + 1e-9)
    
    # Uniformity (lower is better, measures how uniform on hypersphere)
    n = min(len(embeddings), 5000)  # subsample for efficiency
    idx = np.random.choice(len(embeddings), n, replace=False)
    sub = emb_normalized[idx]
    pairwise_sq = np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1)
    uniformity = np.log(np.exp(-2 * pairwise_sq).mean())
    
    # Isotropy: ratio of min/max singular value
    _, s, _ = np.linalg.svd(emb_normalized[:n], full_matrices=False)
    isotropy = float(s[-1] / (s[0] + 1e-9))
    
    # Effective dimensionality (participation ratio)
    s_normalized = s / s.sum()
    effective_dim = 1.0 / (s_normalized ** 2).sum()
    
    return {
        "uniformity": float(uniformity),
        "isotropy": float(isotropy),
        "effective_dimensionality": float(effective_dim),
        "embedding_dim": embeddings.shape[1],
        "num_samples": len(embeddings),
    }


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def run_full_evaluation(
    embeddings: np.ndarray,
    model_name: str,
    labels: Optional[np.ndarray] = None,
    n_clusters_list: List[int] = [5, 10, 15, 20],
    random_seeds: List[int] = [42, 123, 456],
    output_dir: str = "./evaluation_results",
) -> Dict:
    """
    Run complete evaluation suite on embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"{'='*60}")
    
    results = {"model": model_name}
    
    # 1. Clustering
    print("\n[1/5] Clustering evaluation...")
    results["clustering"] = evaluate_clustering(
        embeddings, n_clusters_list=n_clusters_list, random_seeds=random_seeds
    )
    
    # 2. Embedding quality
    print("[2/5] Embedding quality metrics...")
    results["embedding_quality"] = evaluate_embedding_quality(embeddings)
    
    # Generate cluster labels for downstream tasks if no labels provided
    if labels is None:
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        results["labels_source"] = "kmeans_k10"
    else:
        results["labels_source"] = "provided"
    
    # 3. Linear probing
    print("[3/5] Linear probing...")
    results["linear_probe"] = evaluate_linear_probe(
        embeddings, labels, random_seeds=random_seeds
    )
    
    # 4. Retrieval
    print("[4/5] Retrieval evaluation...")
    results["retrieval"] = evaluate_retrieval(embeddings, labels)
    
    # 5. Anomaly detection
    print("[5/5] Anomaly detection...")
    results["anomaly_detection"] = evaluate_anomaly_detection(embeddings, labels)
    
    # Save results
    output_path = os.path.join(output_dir, f"{model_name}_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    return results


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Header
    models = list(all_results.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:<20}" for m in models)
    print(header)
    print("-" * len(header))
    
    # Clustering (k=10, kmeans)
    for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
        row = f"{metric:<30}"
        for model in models:
            r = all_results[model]
            key = "kmeans_k10"
            if key in r.get("clustering", {}):
                m = r["clustering"][key][metric]
                row += f"{m['mean']:.4f}±{m['std']:.4f}  "
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    print("-" * len(header))
    
    # Embedding quality
    for metric in ["uniformity", "isotropy", "effective_dimensionality"]:
        row = f"{metric:<30}"
        for model in models:
            r = all_results[model]
            if metric in r.get("embedding_quality", {}):
                val = r["embedding_quality"][metric]
                row += f"{val:.4f}            "
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    print("-" * len(header))
    
    # Retrieval
    for k in [1, 5, 10]:
        metric = f"precision@{k}"
        row = f"{metric:<30}"
        for model in models:
            r = all_results[model]
            if metric in r.get("retrieval", {}):
                m = r["retrieval"][metric]
                row += f"{m['mean']:.4f}±{m['std']:.4f}  "
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    print("=" * 80)
