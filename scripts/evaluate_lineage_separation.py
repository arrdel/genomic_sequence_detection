#!/usr/bin/env python3
"""
Evaluate embeddings on synthetic lineage reads with ground-truth labels.

This is the reviewer-facing counterpart to simulate_lineage_reads.py. It avoids
the previous "smallest k-means cluster = anomaly" proxy by using known lineage
labels from synthetic reads generated from Wuhan, Delta, and Omicron genomes.

Typical workflow:
    python scripts/simulate_lineage_reads.py --reads-per-lineage 10000

    python scripts/extract_embeddings.py \
        --model-type masked_vqvae \
        --checkpoint experiments/2_masked_vqvae/mqvae_masked/best_model.pt \
        --data-path data/synthetic_lineages/all_lineages.fastq \
        --output-path data/synthetic_lineages/masked_vqvae_embeddings.npy \
        --num-samples 30000

    python scripts/evaluate_lineage_separation.py \
        --embeddings data/synthetic_lineages/masked_vqvae_embeddings.npy \
        --labels data/synthetic_lineages/labels.tsv \
        --output-dir results/synthetic_lineage_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        average_precision_score,
        balanced_accuracy_score,
        normalized_mutual_info_score,
        roc_auc_score,
        silhouette_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ModuleNotFoundError as exc:
    _MISSING_DEPENDENCY = exc
else:
    _MISSING_DEPENDENCY = None


def load_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    read_ids: List[str] = []
    labels: List[str] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "read_id" not in reader.fieldnames or "lineage" not in reader.fieldnames:
            raise ValueError("labels.tsv must contain columns: read_id, lineage")
        for row in reader:
            read_ids.append(row["read_id"])
            labels.append(row["lineage"])
    return np.asarray(read_ids), np.asarray(labels)


def subsample_aligned(
    embeddings: np.ndarray,
    read_ids: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples is None or len(embeddings) <= max_samples:
        return embeddings, read_ids, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), size=max_samples, replace=False)
    return embeddings[idx], read_ids[idx], labels[idx]


def evaluate_true_label_structure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Dict:
    encoded = LabelEncoder().fit_transform(labels)
    n_classes = len(np.unique(encoded))

    scaler = StandardScaler()
    x = scaler.fit_transform(embeddings)

    result: Dict = {
        "n_samples": int(len(labels)),
        "n_lineages": int(n_classes),
        "lineage_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
    }

    if n_classes > 1:
        result["true_label_silhouette"] = float(silhouette_score(x, encoded))

    kmeans = KMeans(n_clusters=n_classes, random_state=seed, n_init=20)
    clusters = kmeans.fit_predict(x)
    result["kmeans_vs_true_labels"] = {
        "adjusted_rand_index": float(adjusted_rand_score(encoded, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(encoded, clusters)),
    }

    n_splits = min(5, min(np.bincount(encoded)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        knn = KNeighborsClassifier(n_neighbors=5)
        logreg = LogisticRegression(max_iter=2000, random_state=seed)
        result["lineage_probe"] = {
            "knn_accuracy_mean": float(cross_val_score(knn, x, encoded, cv=cv, scoring="accuracy").mean()),
            "knn_f1_macro_mean": float(cross_val_score(knn, x, encoded, cv=cv, scoring="f1_macro").mean()),
            "logistic_accuracy_mean": float(cross_val_score(logreg, x, encoded, cv=cv, scoring="accuracy").mean()),
            "logistic_f1_macro_mean": float(cross_val_score(logreg, x, encoded, cv=cv, scoring="f1_macro").mean()),
            "cv_folds": int(n_splits),
        }

    return result


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int],
) -> Dict:
    x = StandardScaler().fit_transform(embeddings)
    dists = cdist(x, x, metric="cosine")
    np.fill_diagonal(dists, np.inf)

    result = {}
    for k in k_values:
        k = min(k, len(labels) - 1)
        if k < 1:
            continue
        precisions = []
        for i in range(len(labels)):
            nn_idx = np.argsort(dists[i])[:k]
            precisions.append(float(np.mean(labels[nn_idx] == labels[i])))
        result[f"precision@{k}"] = {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
        }
    return result


def evaluate_pair_separation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_a: str,
    label_b: str,
) -> Dict:
    mask = np.isin(labels, [label_a, label_b])
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return {
            "note": f"Skipping pair separation: need both {label_a!r} and {label_b!r}."
        }

    x = StandardScaler().fit_transform(embeddings[mask])
    y = labels[mask]
    centroid_a = x[y == label_a].mean(axis=0, keepdims=True)
    centroid_b = x[y == label_b].mean(axis=0, keepdims=True)

    dist_to_a = cdist(x, centroid_a).ravel()
    dist_to_b = cdist(x, centroid_b).ravel()
    pred = np.where(dist_to_a <= dist_to_b, label_a, label_b)

    true_binary = (y == label_b).astype(int)
    score_binary = dist_to_a - dist_to_b

    return {
        "labels": [label_a, label_b],
        "n_samples": int(mask.sum()),
        "centroid_distance": float(np.linalg.norm(centroid_a - centroid_b)),
        "nearest_centroid_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_b_auroc": float(roc_auc_score(true_binary, score_binary)),
    }


def evaluate_held_out_novelty(
    embeddings: np.ndarray,
    labels: np.ndarray,
    held_out_label: str,
) -> Dict:
    is_held_out = labels == held_out_label
    if is_held_out.sum() == 0:
        return {"note": f"Skipping novelty: held-out label {held_out_label!r} not found."}
    if is_held_out.sum() == len(labels):
        return {"note": "Skipping novelty: all samples are held-out."}

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[~is_held_out])
    x_all = scaler.transform(embeddings)
    train_labels = labels[~is_held_out]

    centroids = []
    for label in sorted(np.unique(train_labels)):
        centroids.append(x_train[train_labels == label].mean(axis=0))
    centroids = np.vstack(centroids)

    novelty_score = cdist(x_all, centroids).min(axis=1)
    y_true = is_held_out.astype(int)

    return {
        "held_out_label": held_out_label,
        "held_out_fraction": float(is_held_out.mean()),
        "distance_to_known_centroid_auroc": float(roc_auc_score(y_true, novelty_score)),
        "distance_to_known_centroid_auprc": float(average_precision_score(y_true, novelty_score)),
        "mean_score_known": float(novelty_score[~is_held_out].mean()),
        "mean_score_held_out": float(novelty_score[is_held_out].mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", required=True, help="Path to .npy embedding matrix.")
    p.add_argument("--labels", required=True, help="Path to simulator labels.tsv.")
    p.add_argument("--output-dir", default="results/synthetic_lineage_eval")
    p.add_argument("--model-name", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--pair-a", default="wuhan")
    p.add_argument("--pair-b", default="omicron")
    p.add_argument("--held-out-label", default="omicron")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if _MISSING_DEPENDENCY is not None:
        raise SystemExit(
            f"Missing dependency: {_MISSING_DEPENDENCY.name}. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    read_ids, labels = load_labels(args.labels)
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Embeddings ({len(embeddings)}) and labels ({len(labels)}) must have the same length. "
            "If you used --num-samples during extraction, regenerate labels/embeddings with matching order."
        )

    embeddings, read_ids, labels = subsample_aligned(
        embeddings, read_ids, labels, args.max_samples, args.seed
    )

    model_name = args.model_name or Path(args.embeddings).stem
    print(f"Evaluating {model_name}: embeddings={embeddings.shape}, labels={len(labels)}")
    print("Lineage counts:", dict(zip(*np.unique(labels, return_counts=True))))

    results = {
        "model": model_name,
        "embeddings_path": args.embeddings,
        "labels_path": args.labels,
        "n_samples": int(len(labels)),
        "embedding_dim": int(embeddings.shape[1]),
        "true_label_structure": evaluate_true_label_structure(embeddings, labels, args.seed),
        "retrieval": evaluate_retrieval(embeddings, labels, args.k_values),
        "pair_separation": evaluate_pair_separation(embeddings, labels, args.pair_a, args.pair_b),
        "held_out_novelty": evaluate_held_out_novelty(embeddings, labels, args.held_out_label),
    }

    report_path = out_dir / f"{model_name}_lineage_eval.json"
    with open(report_path, "w") as fh:
        json.dump(results, fh, indent=2)

    summary_path = out_dir / f"{model_name}_headline_metrics.csv"
    tls = results["true_label_structure"]
    pair = results["pair_separation"]
    novelty = results["held_out_novelty"]
    retrieval = results["retrieval"]
    with open(summary_path, "w") as fh:
        fh.write(
            "model,true_label_silhouette,ari,nmi,knn_acc,p_at_1,"
            "pair_centroid_distance,pair_bal_acc,heldout_auroc,heldout_auprc\n"
        )
        fh.write(
            f"{model_name},"
            f"{tls.get('true_label_silhouette', '')},"
            f"{tls.get('kmeans_vs_true_labels', {}).get('adjusted_rand_index', '')},"
            f"{tls.get('kmeans_vs_true_labels', {}).get('normalized_mutual_info', '')},"
            f"{tls.get('lineage_probe', {}).get('knn_accuracy_mean', '')},"
            f"{retrieval.get('precision@1', {}).get('mean', '')},"
            f"{pair.get('centroid_distance', '')},"
            f"{pair.get('nearest_centroid_balanced_accuracy', '')},"
            f"{novelty.get('distance_to_known_centroid_auroc', '')},"
            f"{novelty.get('distance_to_known_centroid_auprc', '')}\n"
        )

    print(f"Saved full report: {report_path}")
    print(f"Saved headline CSV: {summary_path}")
    print("\nHeadline:")
    print(f"  true-label silhouette: {tls.get('true_label_silhouette', 'N/A')}")
    print(f"  ARI / NMI: {tls.get('kmeans_vs_true_labels', {})}")
    print(f"  P@1: {retrieval.get('precision@1', {}).get('mean', 'N/A')}")
    print(f"  {args.pair_a} vs {args.pair_b}: {pair}")
    print(f"  held-out {args.held_out_label}: {novelty}")


if __name__ == "__main__":
    main()
