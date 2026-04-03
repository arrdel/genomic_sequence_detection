#!/usr/bin/env python3
"""
Full Downstream Evaluation Script

Runs the complete 5-metric evaluation suite on all 8 model embeddings:
  1. Clustering (KMeans, multiple k, multiple seeds)
  2. Embedding quality (uniformity, isotropy, effective dimensionality)
  3. Linear probing (Logistic, KNN, SVM with 5-fold CV)
  4. Retrieval (Precision@k)
  5. Anomaly detection (AUROC, AUPRC)

Usage on ai-lab1:
    python -u scripts/run_full_downstream_eval.py \
        --experiments-dir ~/data/contrastive/experiments \
        --output-dir ~/data/contrastive/experiments/full_evaluation \
        --max-samples 10000
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    evaluate_clustering,
    evaluate_embedding_quality,
    evaluate_retrieval,
    evaluate_anomaly_detection,
)


def fast_linear_probe(embeddings, labels, n_folds=3, seed=42):
    """
    Fast linear probing with KNN + LogReg (saga solver, fast convergence).
    Returns summary dict.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    results = {}

    # KNN (instant, no iterations)
    knn = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc = cross_val_score(knn, X, labels, cv=cv, scoring="accuracy")
    f1 = cross_val_score(knn, X, labels, cv=cv, scoring="f1_macro")
    results["knn"] = {
        "accuracy": {"mean": float(acc.mean()), "std": float(acc.std())},
        "f1_macro": {"mean": float(f1.mean()), "std": float(f1.std())},
    }

    # LogReg (saga solver with tolerance for speed)
    lr = LogisticRegression(max_iter=500, solver="saga", tol=1e-2, random_state=seed)
    acc = cross_val_score(lr, X, labels, cv=cv, scoring="accuracy")
    f1 = cross_val_score(lr, X, labels, cv=cv, scoring="f1_macro")
    results["logistic"] = {
        "accuracy": {"mean": float(acc.mean()), "std": float(acc.std())},
        "f1_macro": {"mean": float(f1.mean()), "std": float(f1.std())},
    }

    return results


# ── Embedding registry ──────────────────────────────────────────────────────
def build_embedding_registry(experiments_dir: str) -> dict:
    """
    Returns { model_name: path_to_npy } for all 8 models.
    """
    return {
        "vqvae_base": os.path.join(
            experiments_dir,
            "1_standard_vqvae/vqvae_base/vqvae_base_eval_embeddings.npy",
        ),
        "masked_vqvae": os.path.join(
            experiments_dir,
            "2_masked_vqvae/mqvae_masked/masked_vqvae_eval_embeddings.npy",
        ),
        "contrastive_64d": os.path.join(
            experiments_dir,
            "3_contrastive_vqvae/contrastive_64dim/run_20260308_030233/val_embeddings.npy",
        ),
        "contrastive_128d": os.path.join(
            experiments_dir,
            "3_contrastive_vqvae_128dim/visualizations/final_embeddings.npy",
        ),
        "autoencoder": os.path.join(
            experiments_dir,
            "baselines/autoencoder/autoencoder_embeddings.npy",
        ),
        "transformer_vae": os.path.join(
            experiments_dir,
            "baselines/transformer_vae/transformer_vae_embeddings.npy",
        ),
        "dnabert2": os.path.join(
            experiments_dir,
            "baselines/dnabert2/dnabert2_embeddings.npy",
        ),
        "kmer_pca_64": os.path.join(
            experiments_dir,
            "baselines/kmer_pca/kmer_pca_64_embeddings.npy",
        ),
    }


def load_and_subsample(path: str, max_samples: int, seed: int = 42) -> np.ndarray:
    """Load .npy embedding file and subsample to max_samples if needed."""
    emb = np.load(path)
    if len(emb) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(emb), max_samples, replace=False)
        emb = emb[idx]
    return emb


def parse_args():
    parser = argparse.ArgumentParser(description="Full downstream evaluation")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=os.path.expanduser("~/data/contrastive/experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/data/contrastive/experiments/full_evaluation"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Max samples per model (subsampled for fair comparison & speed)",
    )
    parser.add_argument(
        "--clustering-methods",
        nargs="+",
        default=["kmeans"],
        help="Clustering methods (kmeans only by default to avoid O(n^2))",
    )
    parser.add_argument(
        "--n-clusters",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    return parser.parse_args()


def extract_headline_metrics(result: dict) -> dict:
    """Pull out a flat dict of the key metrics for the summary table."""
    headline = {}

    # Clustering — use kmeans_k10 as canonical
    cl = result.get("clustering", {})
    k10 = cl.get("kmeans_k10", {})
    headline["sil_k10"] = k10.get("silhouette", {}).get("mean", float("nan"))
    headline["db_k10"] = k10.get("davies_bouldin", {}).get("mean", float("nan"))
    headline["ch_k10"] = k10.get("calinski_harabasz", {}).get("mean", float("nan"))

    # Embedding quality
    eq = result.get("embedding_quality", {})
    headline["uniformity"] = eq.get("uniformity", float("nan"))
    headline["isotropy"] = eq.get("isotropy", float("nan"))
    headline["eff_dim"] = eq.get("effective_dimensionality", float("nan"))

    # Linear probe — best classifier accuracy
    lp = result.get("linear_probe", {})
    best_acc = -1.0
    best_clf = "N/A"
    for clf_name, metrics in lp.items():
        acc = metrics.get("accuracy", {}).get("mean", -1)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf_name
    headline["probe_acc"] = best_acc
    headline["probe_f1"] = lp.get(best_clf, {}).get("f1_macro", {}).get("mean", float("nan"))

    # Retrieval
    ret = result.get("retrieval", {})
    headline["p@1"] = ret.get("precision@1", {}).get("mean", float("nan"))
    headline["p@5"] = ret.get("precision@5", {}).get("mean", float("nan"))
    headline["p@10"] = ret.get("precision@10", {}).get("mean", float("nan"))

    # Anomaly detection
    ad = result.get("anomaly_detection", {})
    headline["auroc"] = ad.get("auroc", float("nan"))
    headline["auprc"] = ad.get("auprc", float("nan"))

    return headline


def print_full_summary(all_headlines: dict):
    """Print a nicely formatted summary table."""
    models = list(all_headlines.keys())

    metrics = [
        ("Silhouette (k=10)", "sil_k10", "{:.4f}"),
        ("Davies-Bouldin (k=10)", "db_k10", "{:.4f}"),
        ("Calinski-Harabasz", "ch_k10", "{:.1f}"),
        ("Uniformity (↓)", "uniformity", "{:.4f}"),
        ("Isotropy", "isotropy", "{:.4f}"),
        ("Effective Dim", "eff_dim", "{:.1f}"),
        ("Probe Accuracy", "probe_acc", "{:.4f}"),
        ("Probe F1-macro", "probe_f1", "{:.4f}"),
        ("Precision@1", "p@1", "{:.4f}"),
        ("Precision@5", "p@5", "{:.4f}"),
        ("Precision@10", "p@10", "{:.4f}"),
        ("Anomaly AUROC", "auroc", "{:.4f}"),
        ("Anomaly AUPRC", "auprc", "{:.4f}"),
    ]

    col_w = 18
    header = f"{'Metric':<28}" + "".join(f"{m:>{col_w}}" for m in models)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("FULL DOWNSTREAM EVALUATION — ALL MODELS")
    print(sep)
    print(header)
    print("-" * len(header))

    for label, key, fmt in metrics:
        row = f"{label:<28}"
        for model in models:
            val = all_headlines[model].get(key, float("nan"))
            if np.isnan(val):
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{fmt.format(val):>{col_w}}"
        print(row)

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    registry = build_embedding_registry(args.experiments_dir)

    all_results = {}
    all_headlines = {}
    timings = {}

    print("=" * 70)
    print("FULL DOWNSTREAM EVALUATION PIPELINE")
    print(f"  Max samples per model: {args.max_samples}")
    print(f"  Clustering methods: {args.clustering_methods}")
    print(f"  Cluster sizes: {args.n_clusters}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    for model_name, emb_path in registry.items():
        if not os.path.exists(emb_path):
            print(f"\n⚠️  Skipping {model_name}: {emb_path} not found")
            continue

        emb = load_and_subsample(emb_path, args.max_samples)
        print(f"\n{'='*60}")
        print(f">>> Evaluating {model_name}: {emb.shape}")
        print(f"{'='*60}")

        t0 = time.time()
        result = {"model": model_name}

        # 1. Clustering
        print("[1/5] Clustering evaluation...")
        result["clustering"] = evaluate_clustering(
            emb,
            n_clusters_list=args.n_clusters,
            methods=args.clustering_methods,
            random_seeds=args.seeds,
        )

        # 2. Embedding quality
        print("[2/5] Embedding quality metrics...")
        result["embedding_quality"] = evaluate_embedding_quality(emb)

        # 3. Generate pseudo-labels for downstream tasks
        from sklearn.cluster import KMeans as _KMeans
        pseudo_labels = _KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(emb)
        result["labels_source"] = "kmeans_k10"

        # 4. Linear probing (fast: knn + logistic with saga solver)
        print("[3/5] Linear probing...")
        result["linear_probe"] = fast_linear_probe(emb, pseudo_labels)

        # 5. Retrieval (subsample for speed: cdist is O(n²))
        MAX_RETRIEVAL = 5_000
        print("[4/5] Retrieval evaluation...")
        if len(emb) > MAX_RETRIEVAL:
            rng_ret = np.random.RandomState(42)
            idx_ret = rng_ret.choice(len(emb), MAX_RETRIEVAL, replace=False)
            emb_ret = emb[idx_ret]
            labels_ret = pseudo_labels[idx_ret]
        else:
            emb_ret = emb
            labels_ret = pseudo_labels
        result["retrieval"] = evaluate_retrieval(emb_ret, labels_ret)

        # 6. Anomaly detection
        print("[5/5] Anomaly detection...")
        result["anomaly_detection"] = evaluate_anomaly_detection(emb, pseudo_labels)

        # Save individual result
        ind_path = os.path.join(args.output_dir, f"{model_name}_evaluation.json")
        with open(ind_path, "w") as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - t0
        all_results[model_name] = result
        all_headlines[model_name] = extract_headline_metrics(result)
        timings[model_name] = elapsed
        print(f"  ⏱  {model_name} evaluated in {elapsed:.1f}s")

    # ── Save combined results ────────────────────────────────────────────
    combined_path = os.path.join(args.output_dir, "all_models_full_evaluation.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined results saved to {combined_path}")

    # ── Save headline CSV for easy copy-paste ────────────────────────────
    csv_path = os.path.join(args.output_dir, "headline_metrics.csv")
    models = list(all_headlines.keys())
    metric_keys = list(next(iter(all_headlines.values())).keys())
    with open(csv_path, "w") as f:
        f.write("model," + ",".join(metric_keys) + "\n")
        for m in models:
            vals = [str(all_headlines[m].get(k, "")) for k in metric_keys]
            f.write(f"{m}," + ",".join(vals) + "\n")
    print(f"✓ Headline CSV saved to {csv_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print_full_summary(all_headlines)

    # ── Timing summary ───────────────────────────────────────────────────
    print(f"\n{'Model':<20} {'Time (s)':>10}")
    print("-" * 30)
    for m, t in timings.items():
        print(f"{m:<20} {t:>10.1f}")
    print(f"{'TOTAL':<20} {sum(timings.values()):>10.1f}")


if __name__ == "__main__":
    main()
