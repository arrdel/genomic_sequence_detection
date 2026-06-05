#!/usr/bin/env python3
"""
Positive-control injection experiment for anomaly detection.

Uses pre-computed embeddings. Fits a wuhan reference centroid, then
injects Omicron reads at controlled rates and evaluates AUROC/AUPRC
with explicit positive labels.

Usage:
    python scripts/positive_control_injection.py \
        --embeddings data/synthetic_lineages/maskedvq-seq_ours_embeddings.npy \
        --labels data/synthetic_lineages/labels.tsv \
        --output-dir results/positive_control \
        --model-name "MaskedVQ-Seq (Ours)"
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# Wong colorblind-safe palette (matches rest of paper figures)
WONG = {
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "black":   "#000000",
}


def load_data(embeddings_path, labels_path):
    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(labels_path) as f:
        rows = [line.strip() for line in f if line.strip()]
    # Skip header if present (first field is not a lineage name)
    if "\t" in rows[0] and rows[0].split("\t")[0].lower() in ("read_id", "id", "name", "index"):
        rows = rows[1:]
    # Accept both single-column (label only) and two-column (id\tlabel) TSVs
    labels = np.array([r.split("\t")[-1] for r in rows])
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Embedding/label count mismatch: {len(embeddings)} vs {len(labels)}"
        )
    return embeddings, labels


def injection_auroc(wuhan_emb, omicron_emb, rate, rng, n_test_wuhan, centroid):
    """Single injection-rate evaluation. Returns metrics dict."""
    # How many omicron reads to inject
    # rate = n_omicron / (n_wuhan_test + n_omicron)
    n_inject = min(
        max(1, int(round(n_test_wuhan * rate / (1.0 - rate)))),
        len(omicron_emb),
    )

    omicron_idx = rng.choice(len(omicron_emb), size=n_inject, replace=False)
    inject_emb = omicron_emb[omicron_idx]

    test_emb = np.concatenate([wuhan_emb, inject_emb], axis=0)
    test_labels = np.array([0] * n_test_wuhan + [1] * n_inject, dtype=np.int32)

    # Anomaly score: L2 distance to wuhan centroid
    scores = np.linalg.norm(test_emb - centroid, axis=1)

    auroc  = float(roc_auc_score(test_labels, scores))
    auprc  = float(average_precision_score(test_labels, scores))
    fpr_v, tpr_v, _ = roc_curve(test_labels, scores)
    tpr_at_5fpr = float(np.interp(0.05, fpr_v, tpr_v))

    actual_rate = n_inject / (n_test_wuhan + n_inject)

    return {
        "injection_rate_target": rate,
        "injection_rate_actual": actual_rate,
        "n_wuhan_test": n_test_wuhan,
        "n_omicron_injected": n_inject,
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_fpr5pct": tpr_at_5fpr,
        "random_auprc_baseline": actual_rate,
    }


def run_experiment(wuhan_emb, omicron_emb, injection_rates, ref_fraction, seed):
    rng = np.random.default_rng(seed)

    n_wuhan = len(wuhan_emb)
    n_ref = int(n_wuhan * ref_fraction)

    # Shuffle wuhan, split reference vs test
    perm = rng.permutation(n_wuhan)
    ref_emb  = wuhan_emb[perm[:n_ref]]
    test_wuhan = wuhan_emb[perm[n_ref:]]
    n_test_wuhan = len(test_wuhan)

    # Reference centroid from held-in wuhan reads
    centroid = ref_emb.mean(axis=0)

    print(f"  Reference centroid fit on {n_ref} wuhan reads")
    print(f"  Test pool: {n_test_wuhan} wuhan + sampled omicron")
    print(f"  {'Rate':>6}  {'n_inject':>8}  {'AUROC':>6}  {'AUPRC':>6}  {'TPR@FPR5%':>9}")
    print(f"  {'-'*50}")

    results = []
    for rate in sorted(injection_rates):
        m = injection_auroc(test_wuhan, omicron_emb, rate, rng,
                            n_test_wuhan, centroid)
        results.append(m)
        print(f"  {rate:>5.0%}  {m['n_omicron_injected']:>8d}  "
              f"{m['auroc']:>6.4f}  {m['auprc']:>6.4f}  "
              f"{m['tpr_at_fpr5pct']:>9.4f}")

    return results, centroid


def save_figure(results, model_name, output_dir, slug):
    rates    = [r["injection_rate_actual"] for r in results]
    aurocs   = [r["auroc"] for r in results]
    auprcs   = [r["auprc"] for r in results]
    baselines = [r["random_auprc_baseline"] for r in results]
    tprs     = [r["tpr_at_fpr5pct"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    pct = [r * 100 for r in rates]

    for ax, vals, baseline, ylabel, title in zip(
        axes,
        [aurocs, auprcs, tprs],
        [0.5, baselines, 0.05],
        ["AUROC", "AUPRC", "TPR @ 5% FPR"],
        ["Anomaly AUROC vs Injection Rate",
         "Anomaly AUPRC vs Injection Rate",
         "TPR @ 5% FPR vs Injection Rate"],
    ):
        ax.plot(pct, vals, "o-", color=WONG["blue"], lw=2, ms=6, label=model_name)
        if isinstance(baseline, list):
            ax.plot(pct, baseline, "--", color=WONG["red"], lw=1.5, label="Random baseline")
        else:
            ax.axhline(baseline, color=WONG["red"], lw=1.5, ls="--",
                       label=f"Random ({baseline:.2f})")
        ax.set_xlabel("Omicron injection rate (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Positive-Control Injection Experiment — {model_name}",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{slug}_injection.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help="Path to .npy embeddings")
    parser.add_argument("--labels", required=True,
                        help="Path to labels.tsv (one label per line, matching embeddings)")
    parser.add_argument("--output-dir", default="results/positive_control")
    parser.add_argument("--model-name", default="Model")
    parser.add_argument("--injection-rates", nargs="+", type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--ref-fraction", type=float, default=0.5,
                        help="Fraction of wuhan reads used to fit centroid (rest go to test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Positive-control injection: {args.model_name}")
    print(f"{'='*60}")

    embeddings, labels = load_data(args.embeddings, args.labels)
    print(f"Loaded embeddings: {embeddings.shape}, labels: {len(labels)}")

    wuhan_emb   = embeddings[labels == "wuhan"]
    omicron_emb = embeddings[labels == "omicron"]
    print(f"Wuhan: {len(wuhan_emb)},  Omicron: {len(omicron_emb)}\n")

    results, _ = run_experiment(
        wuhan_emb, omicron_emb,
        injection_rates=args.injection_rates,
        ref_fraction=args.ref_fraction,
        seed=args.seed,
    )

    slug = (args.model_name.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("-", ""))

    # Save JSON
    out = {
        "model_name": args.model_name,
        "embeddings_path": args.embeddings,
        "ref_fraction": args.ref_fraction,
        "seed": args.seed,
        "results": results,
    }
    json_path = os.path.join(args.output_dir, f"{slug}_injection.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    print("\nGenerating figure ...")
    save_figure(results, args.model_name, args.output_dir, slug)

    # Print summary table for paper
    print("\nSummary table:")
    print(f"  {'Rate':>6}  {'AUROC':>6}  {'AUPRC':>6}  {'TPR@5%FPR':>9}")
    for r in results:
        print(f"  {r['injection_rate_actual']:>5.1%}  "
              f"{r['auroc']:>6.4f}  {r['auprc']:>6.4f}  "
              f"{r['tpr_at_fpr5pct']:>9.4f}")


if __name__ == "__main__":
    main()
