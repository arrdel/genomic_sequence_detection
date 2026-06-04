#!/usr/bin/env python3
"""
End-to-end pipeline: Omicron vs. Wildtype lineage separation analysis.

Addresses Reviewer uRw2's specific request:
  "evaluate the embedding for the omicron variant vs. the wildtype"

and Reviewer wkYq's request for:
  "a synthetic data section where ground-truth is known ... enable evaluation
   of FPR/TPR detection"

Pipeline steps
--------------
  1. simulate   — generate labelled short reads from per-lineage reference genomes
  2. extract    — run reads through trained model to obtain embeddings
  3. evaluate   — compute ARI, NMI, silhouette, AUROC, held-out novelty
  4. visualize  — produce t-SNE figures coloured by ground-truth lineage

Each step is skipped automatically if its output already exists (use --force to
re-run from scratch).

Usage (full pipeline)
---------------------
    cd genomic_sequence_detection

    python scripts/run_lineage_separation_analysis.py \\
        --checkpoint  experiments/2_masked_vqvae/mqvae_masked/best_model.pt \\
        --model-type  masked_vqvae \\
        --model-name  "MaskedVQ-Seq (Ours)"

Usage (visualization only — if embeddings already exist)
---------------------------------------------------------
    python scripts/run_lineage_separation_analysis.py \\
        --skip-simulate --skip-extract \\
        --embeddings  data/synthetic_lineages/my_embeddings.npy \\
        --labels      data/synthetic_lineages/labels.tsv \\
        --model-name  "MaskedVQ-Seq (Ours)"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP: {step}")
    print(f"CMD:  {' '.join(cmd)}")
    print("="*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"\n[ERROR] Step '{step}' failed (exit code {result.returncode}).")


def _count_fastq_records(path: str) -> int:
    """Count FASTQ records (4 lines each)."""
    with open(path) as fh:
        return sum(1 for line in fh if line.startswith("@"))


def _exists_nonempty(*paths: str) -> bool:
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Step control
    p.add_argument("--skip-simulate", action="store_true",
                   help="Skip read simulation (use existing data/synthetic_lineages/ outputs).")
    p.add_argument("--skip-extract",  action="store_true",
                   help="Skip embedding extraction (provide --embeddings directly).")
    p.add_argument("--skip-evaluate", action="store_true",
                   help="Skip quantitative evaluation step.")
    p.add_argument("--skip-visualize", action="store_true",
                   help="Skip t-SNE figure generation.")
    p.add_argument("--force", action="store_true",
                   help="Re-run all steps even if outputs already exist.")

    # Paths
    p.add_argument("--syn-dir", default="data/synthetic_lineages",
                   help="Directory for simulated reads and labels.")
    p.add_argument("--embeddings", default=None,
                   help="Override path to embeddings .npy (auto-derived if omitted).")
    p.add_argument("--labels", default=None,
                   help="Override path to labels.tsv (auto-derived if omitted).")
    p.add_argument("--eval-dir", default="results/synthetic_lineage_eval")
    p.add_argument("--fig-dir", default="paper/figures")

    # Simulation
    p.add_argument("--reads-per-lineage", type=int, default=10_000)
    p.add_argument("--read-length", type=int, default=150)
    p.add_argument("--error-rate", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)

    # Model (required unless --skip-extract)
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained model checkpoint (.pt). Required unless --skip-extract.")
    p.add_argument("--model-type", default="masked_vqvae",
                   choices=["vqvae", "masked_vqvae", "contrastive", "autoencoder",
                             "transformer_vae"],
                   help="Model architecture type.")
    p.add_argument("--vqvae-checkpoint", default=None,
                   help="Base VQ-VAE checkpoint (only needed for contrastive models).")
    p.add_argument("--model-name", default=None,
                   help="Human-readable model name for figure titles and output filenames. "
                        "Defaults to --model-type.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu-id", type=int, default=0)

    # Evaluation
    p.add_argument("--pair-a", default="wuhan")
    p.add_argument("--pair-b", default="omicron")
    p.add_argument("--held-out-label", default="omicron")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    model_name = args.model_name or args.model_type
    # Safe filename stem (no spaces / special chars)
    model_stem = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    syn_dir   = Path(args.syn_dir)
    eval_dir  = Path(args.eval_dir)
    fig_dir   = Path(args.fig_dir)

    combined_fastq = syn_dir / "all_lineages.fastq"
    labels_tsv     = Path(args.labels) if args.labels else syn_dir / "labels.tsv"
    embeddings_npy = Path(args.embeddings) if args.embeddings else (
        syn_dir / f"{model_stem}_embeddings.npy"
    )
    eval_json = eval_dir / f"{model_stem}_embeddings_lineage_eval.json"

    # ── Step 1: Simulate ───────────────────────────────────────────────────────
    if not args.skip_simulate:
        if not args.force and _exists_nonempty(str(combined_fastq), str(labels_tsv)):
            print(f"[simulate] Output already exists, skipping. (--force to re-run)")
        else:
            _run(
                [
                    sys.executable, "scripts/simulate_lineage_reads.py",
                    "--reads-per-lineage", str(args.reads_per_lineage),
                    "--read-length",       str(args.read_length),
                    "--error-rate",        str(args.error_rate),
                    "--out-dir",           str(syn_dir),
                    "--seed",              str(args.seed),
                ],
                "Simulate labelled reads",
            )
    else:
        print("[simulate] Skipped (--skip-simulate).")

    if not _exists_nonempty(str(combined_fastq), str(labels_tsv)):
        sys.exit(
            f"\n[ERROR] Required files not found:\n"
            f"  {combined_fastq}\n  {labels_tsv}\n"
            "Run without --skip-simulate, or point --labels at an existing file."
        )

    # ── Step 2: Extract embeddings ─────────────────────────────────────────────
    if not args.skip_extract:
        if args.checkpoint is None:
            sys.exit(
                "\n[ERROR] --checkpoint is required for embedding extraction.\n"
                "Pass --skip-extract if embeddings already exist, or provide "
                "--embeddings /path/to/existing.npy together with --skip-extract."
            )
        if not args.force and _exists_nonempty(str(embeddings_npy)):
            print(f"[extract] Embeddings already exist at {embeddings_npy}, skipping.")
        else:
            # Count total reads so we extract ALL of them (alignment with labels.tsv)
            n_total = _count_fastq_records(str(combined_fastq))
            print(f"[extract] Total reads in {combined_fastq}: {n_total}")

            extract_cmd = [
                sys.executable, "scripts/extract_embeddings.py",
                "--model-type",  args.model_type,
                "--checkpoint",  args.checkpoint,
                "--data-path",   str(combined_fastq),
                "--output-path", str(embeddings_npy),
                "--num-samples", str(n_total),
                "--batch-size",  str(args.batch_size),
                "--gpu-id",      str(args.gpu_id),
            ]
            if args.vqvae_checkpoint:
                extract_cmd += ["--vqvae-checkpoint", args.vqvae_checkpoint]

            _run(extract_cmd, "Extract embeddings")
    else:
        print("[extract] Skipped (--skip-extract).")

    if not _exists_nonempty(str(embeddings_npy)):
        sys.exit(
            f"\n[ERROR] Embeddings not found: {embeddings_npy}\n"
            "Run without --skip-extract, or pass --embeddings /path/to/file.npy."
        )

    # ── Step 3: Evaluate ───────────────────────────────────────────────────────
    if not args.skip_evaluate:
        if not args.force and _exists_nonempty(str(eval_json)):
            print(f"[evaluate] Results already exist at {eval_json}, skipping.")
        else:
            _run(
                [
                    sys.executable, "scripts/evaluate_lineage_separation.py",
                    "--embeddings",     str(embeddings_npy),
                    "--labels",         str(labels_tsv),
                    "--output-dir",     str(eval_dir),
                    "--model-name",     f"{model_stem}_embeddings",
                    "--pair-a",         args.pair_a,
                    "--pair-b",         args.pair_b,
                    "--held-out-label", args.held_out_label,
                    "--k-values",       *[str(k) for k in args.k_values],
                    "--seed",           str(args.seed),
                ],
                "Evaluate lineage separation",
            )
    else:
        print("[evaluate] Skipped (--skip-evaluate).")

    # ── Step 4: Visualize ──────────────────────────────────────────────────────
    if not args.skip_visualize:
        vis_cmd = [
            sys.executable, "scripts/plot_lineage_tsne.py",
            "--embeddings", str(embeddings_npy),
            "--labels",     str(labels_tsv),
            "--output-dir", str(fig_dir),
            "--model-name", model_name,
            "--all-stem",   f"fig_lineage_tsne_all_{model_stem}",
            "--pair-stem",  f"fig_lineage_tsne_omicron_vs_wuhan_{model_stem}",
            "--seed",       str(args.seed),
        ]
        if _exists_nonempty(str(eval_json)):
            vis_cmd += ["--eval-json", str(eval_json)]

        _run(vis_cmd, "Generate t-SNE figures")
    else:
        print("[visualize] Skipped (--skip-visualize).")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  Simulated reads : {syn_dir}")
    print(f"  Embeddings      : {embeddings_npy}")
    print(f"  Evaluation JSON : {eval_json}")
    print(f"  Figures         : {fig_dir}/fig_lineage_tsne_*.{{pdf,png}}")

    if _exists_nonempty(str(eval_json)):
        with open(eval_json) as fh:
            results = json.load(fh)
        print("\nHeadline metrics:")
        tls  = results.get("true_label_structure", {})
        pair = results.get("pair_separation", {})
        nov  = results.get("held_out_novelty", {})
        ret  = results.get("retrieval", {})
        print(f"  true-label silhouette     : {tls.get('true_label_silhouette', 'N/A')}")
        print(f"  ARI / NMI                 : {tls.get('kmeans_vs_true_labels', {})}")
        print(f"  kNN probe accuracy        : "
              f"{tls.get('lineage_probe', {}).get('knn_accuracy_mean', 'N/A')}")
        print(f"  P@1 retrieval             : "
              f"{ret.get('precision@1', {}).get('mean', 'N/A')}")
        print(f"  Omicron vs Wuhan AUROC    : {pair.get('label_b_auroc', 'N/A')}")
        print(f"  Omicron vs Wuhan Bal-Acc  : "
              f"{pair.get('nearest_centroid_balanced_accuracy', 'N/A')}")
        print(f"  Held-out novelty AUROC    : "
              f"{nov.get('distance_to_known_centroid_auroc', 'N/A')}")
        print(f"  Held-out novelty AUPRC    : "
              f"{nov.get('distance_to_known_centroid_auprc', 'N/A')}")


if __name__ == "__main__":
    main()
