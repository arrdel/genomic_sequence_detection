#!/usr/bin/env python3
"""
Generate publication-quality t-SNE figures for synthetic lineage embeddings.

Directly addresses Reviewer uRw2's request to:
  "evaluate the embedding for the omicron variant vs. the wildtype"

Produces two figures:
  fig_lineage_tsne_all.{pdf,png}    — all three lineages coloured by ground-truth label
  fig_lineage_tsne_pair.{pdf,png}   — Omicron vs. Wuhan only (reviewer's specific ask),
                                       annotated with AUROC / balanced-accuracy from
                                       evaluate_lineage_separation.py if available.

Typical usage (after running simulate_lineage_reads.py and extract_embeddings.py):

    python scripts/plot_lineage_tsne.py \\
        --embeddings data/synthetic_lineages/masked_vqvae_embeddings.npy \\
        --labels     data/synthetic_lineages/labels.tsv \\
        --output-dir paper/figures \\
        --model-name "MaskedVQ-Seq (Ours)"

Optionally, pass --eval-json to overlay headline metrics on the figure:

    python scripts/plot_lineage_tsne.py \\
        --embeddings data/synthetic_lineages/masked_vqvae_embeddings.npy \\
        --labels     data/synthetic_lineages/labels.tsv \\
        --eval-json  results/synthetic_lineage_eval/masked_vqvae_embeddings_lineage_eval.json \\
        --output-dir paper/figures \\
        --model-name "MaskedVQ-Seq (Ours)"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    sys.exit(f"Missing dependency: {exc.name}. Install with: pip install matplotlib scikit-learn")


# ── Visual style (matches generate_paper_figures.py) ──────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.linewidth":     0.6,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Colorblind-friendly Wong palette
LINEAGE_COLORS: Dict[str, str] = {
    "wuhan":   "#0072B2",   # blue
    "delta":   "#E69F00",   # orange
    "omicron": "#D55E00",   # vermillion
}
LINEAGE_LABELS: Dict[str, str] = {
    "wuhan":   "Wuhan (wildtype)",
    "delta":   "Delta (B.1.617.2)",
    "omicron": "Omicron (BA.1)",
}
MARKER_SIZE_SMALL = 2.5
MARKER_SIZE_LARGE = 4.5


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    read_ids: List[str] = []
    labels: List[str] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            read_ids.append(row["read_id"])
            labels.append(row["lineage"])
    return np.asarray(read_ids), np.asarray(labels)


def stratified_subsample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep up to max_per_class samples per unique label (balanced subsample)."""
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    for label in np.unique(labels):
        mask = np.where(labels == label)[0]
        n = min(len(mask), max_per_class)
        idx.extend(rng.choice(mask, size=n, replace=False).tolist())
    idx = np.asarray(idx)
    rng.shuffle(idx)
    return embeddings[idx], labels[idx]


def compute_tsne(embeddings: np.ndarray, perplexity: int = 40, seed: int = 42) -> np.ndarray:
    x = StandardScaler().fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        n_jobs=1,
    )
    return tsne.fit_transform(x)


def load_eval_json(path: Optional[str]) -> Optional[Dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _scatter_lineage(ax, xy: np.ndarray, labels: np.ndarray, lineages: List[str], s: float) -> None:
    for lineage in lineages:
        mask = labels == lineage
        if mask.sum() == 0:
            continue
        color = LINEAGE_COLORS.get(lineage, "#999999")
        label = LINEAGE_LABELS.get(lineage, lineage)
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=color, s=s, alpha=0.55, linewidths=0,
            rasterized=True, label=label, zorder=3,
        )


def _draw_centroids(ax, xy: np.ndarray, labels: np.ndarray, lineages: List[str]) -> None:
    for lineage in lineages:
        mask = labels == lineage
        if mask.sum() == 0:
            continue
        cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
        color = LINEAGE_COLORS.get(lineage, "#999999")
        ax.scatter(cx, cy, c=color, s=120, marker="*",
                   edgecolors="white", linewidths=0.8, zorder=6)


def _clean_ax(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_color("0.7")


def _add_metric_badge(ax, text: str, color: str = "#d62728") -> None:
    ax.text(
        0.97, 0.04, text,
        transform=ax.transAxes, fontsize=7.5, fontweight="bold",
        color="white", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.88),
        zorder=7,
    )


def _save(fig, path: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(path / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  saved {stem}.{{pdf,png}}")


# ── Figure 1: all three lineages ───────────────────────────────────────────────

def fig_all_lineages(
    xy: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    eval_data: Optional[Dict],
    out_dir: Path,
    stem: str,
) -> None:
    present = [l for l in ("wuhan", "delta", "omicron") if (labels == l).any()]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04)

    _scatter_lineage(ax, xy, labels, present, s=MARKER_SIZE_SMALL)
    _draw_centroids(ax, xy, labels, present)
    _clean_ax(ax)

    ax.set_title(f"t-SNE — Synthetic Lineage Reads\n{model_name}", pad=6)

    # Legend (below plot)
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=LINEAGE_COLORS.get(l, "#999"),
               markersize=6, label=LINEAGE_LABELS.get(l, l))
        for l in present
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="0.3", markersize=8, label="Centroid")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=7.5,
              framealpha=0.9, borderpad=0.6)

    # Metric badge from evaluation JSON
    if eval_data is not None:
        tls = eval_data.get("true_label_structure", {})
        sil = tls.get("true_label_silhouette")
        if sil is not None:
            _add_metric_badge(ax, f"Silhouette = {sil:.3f}")

    _save(fig, out_dir, stem)


# ── Figure 2: Omicron vs Wuhan pair ───────────────────────────────────────────

def fig_pair_omicron_wuhan(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    eval_data: Optional[Dict],
    out_dir: Path,
    stem: str,
    seed: int,
) -> None:
    pair = ("wuhan", "omicron")
    mask = np.isin(labels, pair)
    if mask.sum() == 0:
        print("  SKIP pair figure: neither 'wuhan' nor 'omicron' found in labels.")
        return
    if len(np.unique(labels[mask])) < 2:
        print("  SKIP pair figure: need BOTH wuhan and omicron labels.")
        return

    emb_pair = embeddings[mask]
    lbl_pair = labels[mask]

    # Balanced subsample so neither class overwhelms
    emb_pair, lbl_pair = stratified_subsample(emb_pair, lbl_pair, max_per_class=3000, seed=seed)

    print(f"  running t-SNE for Omicron vs Wuhan pair "
          f"({len(emb_pair)} points) …")
    xy_pair = compute_tsne(emb_pair, perplexity=40, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.04, wspace=0.10)

    # Panel (a): t-SNE scatter
    ax = axes[0]
    _scatter_lineage(ax, xy_pair, lbl_pair, list(pair), s=MARKER_SIZE_LARGE)
    _draw_centroids(ax, xy_pair, lbl_pair, list(pair))
    _clean_ax(ax)
    ax.set_title("(a) t-SNE: Omicron vs. Wuhan Wildtype", pad=5)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=LINEAGE_COLORS[l], markersize=7,
               label=LINEAGE_LABELS[l])
        for l in pair
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="0.3", markersize=9, label="Centroid")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    # Metric badge
    if eval_data is not None:
        pair_sep = eval_data.get("pair_separation", {})
        auroc = pair_sep.get("label_b_auroc")
        balacc = pair_sep.get("nearest_centroid_balanced_accuracy")
        if auroc is not None:
            _add_metric_badge(ax, f"AUROC = {auroc:.3f}")
        if balacc is not None:
            ax.text(
                0.97, 0.11, f"Bal-Acc = {balacc:.3f}",
                transform=ax.transAxes, fontsize=7.5, fontweight="bold",
                color="white", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0072B2", ec="none", alpha=0.88),
            )

    # Panel (b): centroid-distance bar chart
    ax2 = axes[1]
    x_scaled = StandardScaler().fit_transform(emb_pair)
    centroid_w = x_scaled[lbl_pair == "wuhan"].mean(axis=0)
    centroid_o = x_scaled[lbl_pair == "omicron"].mean(axis=0)

    dist_to_w = np.linalg.norm(x_scaled - centroid_w, axis=1)
    dist_to_o = np.linalg.norm(x_scaled - centroid_o, axis=1)

    bins = np.linspace(
        min(dist_to_w.min(), dist_to_o.min()),
        max(dist_to_w.max(), dist_to_o.max()),
        40,
    )
    ax2.hist(dist_to_w[lbl_pair == "wuhan"], bins=bins, density=True,
             color=LINEAGE_COLORS["wuhan"], alpha=0.65,
             label=LINEAGE_LABELS["wuhan"], zorder=3)
    ax2.hist(dist_to_o[lbl_pair == "omicron"], bins=bins, density=True,
             color=LINEAGE_COLORS["omicron"], alpha=0.65,
             label=LINEAGE_LABELS["omicron"], zorder=3)
    ax2.set_xlabel("Distance to own centroid (standardized embedding space)", fontsize=8)
    ax2.set_ylabel("Density", fontsize=8)
    ax2.set_title("(b) Within-lineage centroid distances", pad=5)
    ax2.legend(fontsize=7.5)

    if eval_data is not None:
        pair_sep = eval_data.get("pair_separation", {})
        cd = pair_sep.get("centroid_distance")
        if cd is not None:
            ax2.text(
                0.97, 0.95,
                f"Centroid dist = {cd:.3f}",
                transform=ax2.transAxes, fontsize=7.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="0.92", ec="0.7"),
            )

    fig.suptitle(
        f"Omicron vs. Wuhan Wildtype Embedding Separation — {model_name}",
        fontsize=10, fontweight="bold", y=0.97,
    )

    _save(fig, out_dir, stem)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--embeddings", required=True,
                   help="Path to .npy embedding matrix from extract_embeddings.py")
    p.add_argument("--labels", required=True,
                   help="Path to labels.tsv from simulate_lineage_reads.py")
    p.add_argument("--eval-json", default=None,
                   help="Optional: path to *_lineage_eval.json from evaluate_lineage_separation.py "
                        "for metric annotations on the figures.")
    p.add_argument("--output-dir", default="paper/figures")
    p.add_argument("--model-name", default="MaskedVQ-Seq (Ours)",
                   help="Model name used in figure titles.")
    p.add_argument("--max-per-class", type=int, default=3000,
                   help="Max reads per lineage used for t-SNE (balances classes, speeds up TSNE).")
    p.add_argument("--perplexity", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-stem", default="fig_lineage_tsne_all",
                   help="Output filename stem for the all-lineages figure.")
    p.add_argument("--pair-stem", default="fig_lineage_tsne_omicron_vs_wuhan",
                   help="Output filename stem for the Omicron vs Wuhan figure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings: {args.embeddings}")
    embeddings = np.load(args.embeddings)

    print(f"Loading labels:     {args.labels}")
    _, labels = load_labels(args.labels)

    if len(embeddings) != len(labels):
        sys.exit(
            f"Shape mismatch: {len(embeddings)} embeddings vs {len(labels)} labels. "
            "Re-run extract_embeddings.py without --num-samples, or with "
            f"--num-samples {len(labels)}."
        )

    eval_data = load_eval_json(args.eval_json)
    if eval_data is not None:
        print(f"Loaded evaluation metrics: {args.eval_json}")

    print(f"\nLineage counts: "
          + ", ".join(f"{l}={n}" for l, n in
                      zip(*np.unique(labels, return_counts=True))))

    # ── All-lineage figure ─────────────────────────────────────────────────────
    print(f"\nSubsampling (max {args.max_per_class}/class) …")
    emb_sub, lbl_sub = stratified_subsample(
        embeddings, labels, args.max_per_class, args.seed
    )
    print(f"Running t-SNE on {len(emb_sub)} points …")
    xy_all = compute_tsne(emb_sub, perplexity=args.perplexity, seed=args.seed)

    print(f"\nGenerating all-lineage figure → {out_dir}/{args.all_stem}.{{pdf,png}}")
    fig_all_lineages(xy_all, lbl_sub, args.model_name, eval_data, out_dir, args.all_stem)

    # ── Omicron vs Wuhan pair figure ───────────────────────────────────────────
    print(f"\nGenerating Omicron vs Wuhan figure → {out_dir}/{args.pair_stem}.{{pdf,png}}")
    fig_pair_omicron_wuhan(
        embeddings, labels,
        model_name=args.model_name,
        eval_data=eval_data,
        out_dir=out_dir,
        stem=args.pair_stem,
        seed=args.seed,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
