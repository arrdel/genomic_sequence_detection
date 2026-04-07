#!/usr/bin/env python3
"""
Generate all publication-quality figures for the MLHC 2025 paper.
v2 — fully redesigned with clean spacing, no text overlaps, and polished aesthetics.

Figures:
  Fig 2: t-SNE embeddings comparison (2×3 grid)
  Fig 3: Main results — horizontal lollipop chart
  Fig 4: Ablation trends (2×3 grid, 6th cell = legend)
  Fig 5: Masking probability deep-dive (2-panel)
  Fig A1: Reconstruction vs. silhouette Pareto scatter
  Fig A2: Model comparison heatmap
  Fig A3: Training time vs k-mer size
  Fig A4: Effective dimensionality strip chart
"""

import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(ROOT, "results", "embeddings")
ABL_DIR = os.path.join(ROOT, "results", "ablations")
EVAL_DIR= os.path.join(ROOT, "results", "full_evaluation")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────
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
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Palette (colorblind-friendly Wong palette) ────────────────────────────
C = {
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "pink":    "#CC79A7",
    "gray":    "#999999",
}

MODEL_COLORS = {
    "kmer_pca":        C["gray"],
    "autoencoder":     C["orange"],
    "transformer_vae": C["sky"],
    "dnabert2":        C["green"],
    "vqvae_base":      C["yellow"],
    "masked_vqvae":    C["red"],
    "contrastive_64d": C["pink"],
    "contrastive_128d":C["blue"],
}

MODEL_LABELS_SHORT = {
    "kmer_pca":        "k-PCA",
    "autoencoder":     "AE",
    "transformer_vae": "TVAE",
    "dnabert2":        "DB-2",
    "vqvae_base":      "VQ",
    "masked_vqvae":    "Ours",
    "contrastive_64d": "C64",
    "contrastive_128d":"C128",
}

# ── Helpers ────────────────────────────────────────────────────────────────
def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  \u2713 {name}")

def load_embeddings(name, max_n=5000):
    emb = np.load(os.path.join(EMB_DIR, f"{name}.npy"))
    if len(emb) > max_n:
        emb = emb[np.random.RandomState(42).choice(len(emb), max_n, replace=False)]
    return emb

def compute_tsne(emb, perp=30, seed=42):
    emb_s = StandardScaler().fit_transform(emb)
    return TSNE(n_components=2, perplexity=perp, random_state=seed,
                init="pca", learning_rate="auto", max_iter=1000).fit_transform(emb_s)

def load_study(name):
    with open(os.path.join(ABL_DIR, name, f"{name}_summary.json")) as f:
        return json.load(f)

def load_headline():
    d = {}
    with open(os.path.join(EVAL_DIR, "headline_metrics.csv")) as f:
        for row in csv.DictReader(f):
            d[row["model"]] = row
    return d


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — t-SNE embedding comparison (2 x 3)
# ═══════════════════════════════════════════════════════════════════════════
def fig2_tsne():
    print("  Fig 2: t-SNE comparison ...")
    models = ["kmer_pca", "autoencoder", "transformer_vae",
              "dnabert2", "vqvae_base", "masked_vqvae"]
    titles = ["(a) k-mer PCA", "(b) Autoencoder", "(c) Transformer VAE",
              "(d) DNABERT-2", "(e) VQ-VAE Base", "(f) MaskedVQ-Seq (Ours)"]

    fig = plt.figure(figsize=(7.2, 5.0))
    gs  = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.08,
                            left=0.02, right=0.98, top=0.95, bottom=0.03)
    cmap = plt.cm.Set3

    for idx, (name, title) in enumerate(zip(models, titles)):
        ax  = fig.add_subplot(gs[idx])
        emb = load_embeddings(name, 5000)
        lab = KMeans(10, random_state=42, n_init=10).fit_predict(emb)
        xy  = compute_tsne(emb)

        ax.scatter(xy[:, 0], xy[:, 1], c=lab, cmap=cmap,
                   s=1.8, alpha=0.55, rasterized=True, linewidths=0)

        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("0.7")

        # Silhouette badge -- bottom-right with contrasting color
        sil = silhouette_score(emb, lab, sample_size=min(5000, len(emb)))
        badge_color = "#d62728" if name == "masked_vqvae" else "0.25"
        ax.text(0.97, 0.04, f"Sil = {sil:.3f}",
                transform=ax.transAxes, fontsize=7.5, fontweight="bold",
                color="white", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none", alpha=0.85))

    _save(fig, "fig2_tsne_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Main results: horizontal lollipop chart
# ═══════════════════════════════════════════════════════════════════════════
def fig3_main_results():
    print("  Fig 3: Main results lollipop chart ...")
    hl = load_headline()

    models = ["kmer_pca_64", "autoencoder", "transformer_vae", "dnabert2",
              "vqvae_base", "masked_vqvae", "contrastive_64d", "contrastive_128d"]
    ckeys  = ["kmer_pca", "autoencoder", "transformer_vae", "dnabert2",
              "vqvae_base", "masked_vqvae", "contrastive_64d", "contrastive_128d"]
    labels = [MODEL_LABELS_SHORT[k] for k in ckeys]
    colors = [MODEL_COLORS[k] for k in ckeys]

    metrics      = ["sil_k10", "probe_acc", "p@1", "auroc"]
    metric_names = ["Silhouette@10", "Probe Accuracy", "Precision@1", "AUROC"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97, top=0.88, bottom=0.18)

    y = np.arange(len(models))

    for j, (met, mname) in enumerate(zip(metrics, metric_names)):
        ax  = axes[j]
        vals = np.array([float(hl[m][met]) for m in models])
        best = np.argmax(vals)

        # Horizontal lollipop
        for i in range(len(vals)):
            lw = 2.5 if i == best else 1.2
            ms = 8 if i == best else 5
            zz = 4 if i == best else 3
            ax.plot([0, vals[i]], [y[i], y[i]], color=colors[i],
                    linewidth=lw, solid_capstyle="round", zorder=zz)
            ax.plot(vals[i], y[i], "o", color=colors[i], markersize=ms,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=zz+1)

        # Star on best
        ax.plot(vals[best], y[best], "*", color=colors[best],
                markersize=12, markeredgecolor="white", markeredgewidth=0.4, zorder=6)

        ax.set_yticks(y)
        ax.set_yticklabels(labels if j == 0 else [], fontsize=7.5)
        ax.set_xlabel(mname, fontsize=8.5, labelpad=3)
        ax.set_xlim(0, vals.max() * 1.12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2, linewidth=0.4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Model Comparison Across Four Evaluation Metrics",
                 fontsize=11, fontweight="bold", y=0.97)

    _save(fig, "fig3_main_results_bars")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Ablation trends (2 x 3 grid, 6th cell = legend)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_ablation():
    print("  Fig 4: Ablation trends ...")

    fig = plt.figure(figsize=(7.2, 4.6))
    gs  = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.55,
                            left=0.08, right=0.92, top=0.93, bottom=0.08)

    ORANGE, BLUE = "#D55E00", "#0072B2"

    studies = [
        ("codebook_size", "(a) Codebook Size K",
         lambda d: [r["config"]["num_codes"] for r in d],
         lambda v: [str(x) for x in v]),
        ("code_dim", "(b) Code Dimension D",
         lambda d: [r["config"]["code_dim"] for r in d],
         lambda v: [str(x) for x in v]),
        ("kmer_size", "(c) k-mer Size",
         lambda d: [r["config"]["k_mer"] for r in d],
         lambda v: [f"k={x}" for x in v]),
        ("loss_components", "(d) Loss Components",
         lambda d: list(range(len(d))),
         lambda v: ["Recon", "+Comm", "Full", u"Hi-\u03b2", u"Hi-\u03bb"]),
        ("masking", "(e) Masking Probability p",
         lambda d: [r["config"]["mask_training_prob"] for r in d],
         lambda v: [f"{x:.2f}" for x in v]),
    ]

    for idx, (sname, title, xfn, labfn) in enumerate(studies):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        data = load_study(sname)
        xraw = xfn(data)
        xlabels = labfn(xraw)
        sil = [r["clustering"]["kmeans_k10"]["silhouette"]["mean"] for r in data]
        vl  = [r["best_val_loss"] for r in data]
        xs  = np.arange(len(sil))

        # Left axis -- silhouette (filled area + line)
        ax.fill_between(xs, sil, alpha=0.15, color=ORANGE)
        ax.plot(xs, sil, "o-", color=ORANGE, markersize=5, linewidth=1.8,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4)

        # Mark best silhouette with ring
        bst = int(np.argmax(sil))
        ax.plot(xs[bst], sil[bst], "o", color=ORANGE, markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)

        ax.set_xticks(xs)
        rot = 30 if len(xlabels) > 5 else 0
        ha  = "right" if len(xlabels) > 5 else "center"
        ax.set_xticklabels(xlabels, fontsize=6.5, rotation=rot, ha=ha)
        ax.set_ylabel("Silhouette@10", fontsize=7.5, color=ORANGE, labelpad=2)
        ax.tick_params(axis="y", colors=ORANGE, labelsize=6.5)

        # Right axis -- val loss (dashed)
        ax2 = ax.twinx()
        ax2.plot(xs, vl, "s--", color=BLUE, markersize=3.5, linewidth=1.2,
                 markeredgecolor="white", markeredgewidth=0.3, alpha=0.85)
        ax2.set_ylabel("Val Loss", fontsize=7.5, color=BLUE, labelpad=2)
        ax2.tick_params(axis="y", colors=BLUE, labelsize=6.5)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(BLUE)
        ax2.spines["right"].set_linewidth(0.5)
        ax.spines["left"].set_color(ORANGE)
        ax.spines["left"].set_linewidth(0.5)

        ax.set_title(title, fontsize=9, pad=5)

    # Panel (f): legend cell
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    handles = [
        Line2D([0],[0], color=ORANGE, marker="o", markersize=6, linewidth=2,
               markeredgecolor="white", label="Silhouette@10 (\u2191 better)"),
        Line2D([0],[0], color=BLUE, marker="s", markersize=4, linewidth=1.2,
               linestyle="--", markeredgecolor="white", label="Validation Loss (\u2193 better)"),
        Line2D([0],[0], color=ORANGE, marker="o", markersize=9,
               markeredgecolor="black", markeredgewidth=1.0, linestyle="None",
               label="Best Silhouette"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=9, frameon=True,
                  fancybox=True, shadow=False, borderpad=1.2, labelspacing=1.0,
                  edgecolor="0.75")
    ax_leg.set_title("(f) Legend", fontsize=9, pad=5)

    _save(fig, "fig4_ablation_trends")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Masking probability detail (two panels, generous spacing)
# ═══════════════════════════════════════════════════════════════════════════
def fig5_masking():
    print("  Fig 5: Masking detail ...")

    data = load_study("masking")
    xv  = np.array([r["config"]["mask_training_prob"] for r in data])
    sil = [r["clustering"]["kmeans_k10"]["silhouette"]["mean"] for r in data]
    db  = [r["clustering"]["kmeans_k10"]["davies_bouldin"]["mean"] for r in data]
    vl  = [r["best_val_loss"] for r in data]
    ed  = [r["embedding_quality"]["effective_dimensionality"] for r in data]

    ORANGE, BLUE, GREEN, PINK = "#D55E00", "#0072B2", "#009E73", "#CC79A7"

    fig = plt.figure(figsize=(6.5, 3.0))
    gs  = gridspec.GridSpec(1, 2, wspace=0.50, left=0.10, right=0.90, top=0.86, bottom=0.16)

    # -- Panel (a): Silhouette vs Val Loss -----------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()

    # Sweet-spot band
    ax1.axvspan(0.18, 0.32, color=GREEN, alpha=0.08, zorder=0)
    ax1.text(0.25, max(sil)*1.06, "sweet spot", fontsize=7, color=GREEN,
             ha="center", style="italic", alpha=0.8)

    ax1.fill_between(xv, sil, alpha=0.12, color=ORANGE, zorder=1)
    ax1.plot(xv, sil, "o-", color=ORANGE, markersize=6, linewidth=2,
             markeredgecolor="white", markeredgewidth=0.5, zorder=3, label="Silhouette@10")
    ax1b.plot(xv, vl, "s--", color=BLUE, markersize=5, linewidth=1.5,
              markeredgecolor="white", markeredgewidth=0.4, zorder=2, label="Val Loss")

    ax1.set_xlabel("Masking Probability p", fontsize=9)
    ax1.set_ylabel("Silhouette@10", fontsize=9, color=ORANGE)
    ax1b.set_ylabel("Validation Loss", fontsize=9, color=BLUE)
    ax1.tick_params(axis="y", colors=ORANGE, labelsize=7)
    ax1b.tick_params(axis="y", colors=BLUE, labelsize=7)
    ax1b.spines["right"].set_visible(True)
    ax1b.spines["right"].set_color(BLUE)
    ax1b.spines["right"].set_linewidth(0.5)
    ax1.spines["left"].set_color(ORANGE)
    ax1.spines["left"].set_linewidth(0.5)
    ax1.set_title("(a) Cluster Quality vs. Reconstruction", fontsize=9, pad=6)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=6.5, loc="upper left", framealpha=0.9)

    # -- Panel (b): Davies-Bouldin vs Eff Dim --------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()

    ax2.plot(xv, db, "^-", color=GREEN, markersize=6, linewidth=2,
             markeredgecolor="white", markeredgewidth=0.5, zorder=3,
             label=u"Davies-Bouldin (\u2193)")
    ax2b.plot(xv, ed, "D--", color=PINK, markersize=5, linewidth=1.5,
              markeredgecolor="white", markeredgewidth=0.4, zorder=2,
              label="Eff. Dim.")

    ax2.set_xlabel("Masking Probability p", fontsize=9)
    ax2.set_ylabel(u"Davies-Bouldin Index (\u2193)", fontsize=9, color=GREEN)
    ax2b.set_ylabel("Effective Dimensionality", fontsize=9, color=PINK)
    ax2.tick_params(axis="y", colors=GREEN, labelsize=7)
    ax2b.tick_params(axis="y", colors=PINK, labelsize=7)
    ax2b.spines["right"].set_visible(True)
    ax2b.spines["right"].set_color(PINK)
    ax2b.spines["right"].set_linewidth(0.5)
    ax2.spines["left"].set_color(GREEN)
    ax2.spines["left"].set_linewidth(0.5)
    ax2.set_title("(b) Separation vs. Dimensionality", fontsize=9, pad=6)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=6.5, loc="upper left", framealpha=0.9)

    _save(fig, "fig5_masking_detail")


# ═══════════════════════════════════════════════════════════════════════════
# FIG A1 — Reconstruction vs Silhouette Pareto scatter
# ═══════════════════════════════════════════════════════════════════════════
def figA1_scatter():
    print("  Fig A1: Pareto scatter ...")

    study_style = {
        "codebook_size":  ("o", C["red"],    "Codebook K"),
        "code_dim":       ("s", C["blue"],   "Code Dim D"),
        "kmer_size":      ("^", C["green"],  "k-mer Size"),
        "loss_components":("D", C["pink"],   "Loss Config"),
        "masking":        ("v", C["orange"], "Mask Prob p"),
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.14)

    for sn, (marker, col, lab) in study_style.items():
        data = load_study(sn)
        vl  = [r["best_val_loss"] for r in data]
        sil = [r["clustering"]["kmeans_k10"]["silhouette"]["mean"] for r in data]
        ax.scatter(vl, sil, marker=marker, s=55, color=col, label=lab,
                   alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)

    # Desirable-corner arrow
    ax.annotate("", xy=(0.03, 0.08), xytext=(0.25, 0.03),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.2))
    ax.text(0.05, 0.02, u"better \u2192", transform=ax.transAxes,
            fontsize=7, color="0.5", style="italic")

    ax.set_xlabel("Validation Loss (lower = better reconstruction)")
    ax.set_ylabel("Silhouette@10 (higher = better clusters)")
    ax.set_title("Reconstruction vs. Cluster Quality", pad=6)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92,
              borderpad=0.6, handletextpad=0.4)
    ax.grid(alpha=0.15, linewidth=0.4)

    _save(fig, "figA1_recon_vs_sil")


# ═══════════════════════════════════════════════════════════════════════════
# FIG A2 — Model comparison heatmap (redesigned)
# ═══════════════════════════════════════════════════════════════════════════
def figA2_heatmap():
    print("  Fig A2: Heatmap ...")

    hl = load_headline()
    models  = ["kmer_pca_64", "autoencoder", "transformer_vae", "dnabert2",
               "vqvae_base", "masked_vqvae", "contrastive_64d", "contrastive_128d"]
    ckeys   = ["kmer_pca","autoencoder","transformer_vae","dnabert2",
               "vqvae_base","masked_vqvae","contrastive_64d","contrastive_128d"]
    ylabels = ["k-mer PCA", "Autoencoder", "Trans. VAE", "DNABERT-2",
               "VQ-VAE Base", "MaskedVQ-Seq (Ours)", "Contrastive 64d", "Contrastive 128d"]

    metrics      = ["sil_k10", "probe_acc", "p@1", "auroc"]
    metric_names = ["Silhouette\n@10", "Probe\nAccuracy", "Precision\n@1", "AUROC"]

    mat = np.array([[float(hl[m][met]) for met in metrics] for m in models])
    # Normalize per column
    mat_n = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        mn, mx = mat[:,j].min(), mat[:,j].max()
        mat_n[:,j] = (mat[:,j] - mn) / (mx - mn) if mx > mn else 0.5

    # Custom colormap: cream -> amber -> deep red
    cmap = LinearSegmentedColormap.from_list(
        "paper", ["#FFF9E6", "#FDCB6E", "#E17055", "#D63031"])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.subplots_adjust(left=0.30, right=0.86, top=0.91, bottom=0.10)

    im = ax.imshow(mat_n, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Cell annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            v = mat[i, j]
            is_best = (v == mat[:, j].max())
            weight = "bold" if is_best else "normal"
            color  = "white" if mat_n[i,j] > 0.6 else "0.15"
            txt = f"{v:.3f}"
            if is_best:
                txt = f"* {v:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, fontweight=weight, color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_names, fontsize=8.5, ha="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    # Color bar strips on y-axis
    for i, k in enumerate(ckeys):
        rect = plt.Rectangle((-0.55, i-0.45), 0.12, 0.9, linewidth=0,
                               facecolor=MODEL_COLORS[k], clip_on=False,
                               transform=ax.transData)
        ax.add_patch(rect)

    ax.set_title("Model Comparison Heatmap (column-normalized)", fontsize=10, pad=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("Normalized Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _save(fig, "figA2_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIG A3 — Training time vs k-mer size (gradient bars + annotation)
# ═══════════════════════════════════════════════════════════════════════════
def figA3_training_time():
    print("  Fig A3: Training time ...")

    data = load_study("kmer_size")
    k_vals = [r["config"]["k_mer"] for r in data]
    times  = [r["train_time_s"] / 60.0 for r in data]
    vocabs = [4**k + 3 for k in k_vals]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.subplots_adjust(left=0.16, right=0.92, top=0.88, bottom=0.14)

    # Gradient-colored bars
    norm = plt.Normalize(min(times), max(times))
    cmap = plt.cm.YlOrRd
    bars = ax.bar(range(len(k_vals)), times,
                  color=[cmap(norm(t)) for t in times],
                  edgecolor="0.3", linewidth=0.5, width=0.65, zorder=3)

    # Vocab annotations inside bars or above
    for i, (bar, vs, t) in enumerate(zip(bars, vocabs, times)):
        y_pos = bar.get_height() / 2 if bar.get_height() > 15 else bar.get_height() + 2
        va    = "center" if bar.get_height() > 15 else "bottom"
        color = "white" if bar.get_height() > 15 else "0.4"
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"|V|={vs:,}", ha="center", va=va, fontsize=7,
                fontweight="bold" if vs > 10000 else "normal", color=color)

    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k = {k}" for k in k_vals], fontsize=9)
    ax.set_xlabel("k-mer Size", fontsize=10)
    ax.set_ylabel("Training Time (minutes)", fontsize=10)
    ax.set_title("Training Cost vs. k-mer Vocabulary", fontsize=10, pad=6)
    ax.grid(axis="y", alpha=0.15)

    # Speedup annotation
    ratio = times[-1] / times[0]
    ax.annotate(f"{ratio:.0f}x slower",
                xy=(len(k_vals)-1, times[-1]),
                xytext=(len(k_vals)-1.8, times[-1]*0.82),
                fontsize=8, color=C["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2))

    _save(fig, "figA3_training_time")


# ═══════════════════════════════════════════════════════════════════════════
# FIG A4 — Effective dimensionality (lollipop / stem plot)
# ═══════════════════════════════════════════════════════════════════════════
def figA4_effdim():
    print("  Fig A4: Effective dimensionality ...")

    fig = plt.figure(figsize=(7.2, 2.6))
    gs  = gridspec.GridSpec(1, 5, wspace=0.35, left=0.07, right=0.98, top=0.85, bottom=0.22)

    studies = [
        ("codebook_size", "Codebook K",
         lambda d: [str(r["config"]["num_codes"]) for r in d]),
        ("code_dim", "Code Dim D",
         lambda d: [str(r["config"]["code_dim"]) for r in d]),
        ("kmer_size", "k-mer Size",
         lambda d: [f"k={r['config']['k_mer']}" for r in d]),
        ("loss_components", "Loss Config",
         lambda d: ["Recon", "+Comm", "Full", u"Hi-\u03b2", u"Hi-\u03bb"]),
        ("masking", "Mask Prob p",
         lambda d: [f"{r['config']['mask_training_prob']:.2f}" for r in d]),
    ]

    palette = [C["red"], C["blue"], C["green"], C["pink"], C["orange"]]

    for idx, (sname, title, labfn) in enumerate(studies):
        ax = fig.add_subplot(gs[idx])
        data = load_study(sname)
        labels = labfn(data)
        edims  = [r["embedding_quality"]["effective_dimensionality"] for r in data]
        xs     = np.arange(len(edims))

        # Stem plot (lollipop)
        ax.vlines(xs, 0, edims, color=palette[idx], linewidth=2.5, alpha=0.5, zorder=2)
        ax.scatter(xs, edims, color=palette[idx], s=40,
                   edgecolor="white", linewidth=0.5, zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.grid(axis="y", alpha=0.15)

        if idx == 0:
            ax.set_ylabel("Eff. Dimensionality", fontsize=8)

    _save(fig, "figA4_effective_dim")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output -> {FIG_DIR}")
    print("-" * 50)

    fig2_tsne()
    fig3_main_results()
    fig4_ablation()
    fig5_masking()
    figA1_scatter()
    figA2_heatmap()
    figA3_training_time()
    figA4_effdim()

    print("-" * 50)
    figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".pdf"))
    print(f"Done -- {len(figs)} PDFs generated:")
    for f in figs:
        sz = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"  {f:40s} {sz:6.0f} KB")
