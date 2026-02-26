#!/usr/bin/env python3
"""
Unified Experiment Runner

Runs all experiments from a single entry point:
  - Training (VQ-VAE, Masked VQ-VAE, Contrastive fine-tuning)
  - Baselines (DNABERT-2, Standard AE, PCA, Transformer VAE)
  - Ablation studies
  - Full evaluation pipeline

Usage:
    # Train base VQ-VAE
    python scripts/run_experiment.py train-vqvae \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
        --output-dir /media/scratch/adele/contrastive/experiments/vqvae

    # Train masked VQ-VAE
    python scripts/run_experiment.py train-masked \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
        --output-dir /media/scratch/adele/contrastive/experiments/masked

    # Contrastive fine-tuning
    python scripts/run_experiment.py train-contrastive \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
        --checkpoint /media/scratch/adele/contrastive/experiments/vqvae/best_model.pt \
        --output-dir /media/scratch/adele/contrastive/experiments/contrastive

    # Run DNABERT-2 baseline
    python scripts/run_experiment.py baseline-dnabert2 \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
        --output-dir /media/scratch/adele/contrastive/experiments/dnabert2

    # Run all ablations
    python scripts/run_experiment.py ablation \
        --study codebook_size \
        --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq

    # Full evaluation of all models
    python scripts/run_experiment.py evaluate-all \
        --results-dir /media/scratch/adele/contrastive/experiments
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_common_args(parser):
    """Add arguments common to all subcommands."""
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to FASTQ file")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpu", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")


def cmd_train_vqvae(args):
    """Train base VQ-VAE."""
    from scripts.vqvae_train import main as train_main
    # Build argv for the training script
    sys.argv = [
        "vqvae_train.py",
        "--data-path", args.data_path,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--n-gpu", str(args.n_gpu),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
    ]
    if args.no_wandb:
        sys.argv.append("--no-wandb")
    if args.gpu_ids:
        sys.argv.extend(["--gpu-ids", args.gpu_ids])
    train_main()


def cmd_train_masked(args):
    """Train masked VQ-VAE."""
    from scripts.mqvae_train import main as train_main
    sys.argv = [
        "mqvae_train.py",
        "--data-path", args.data_path,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--n-gpu", str(args.n_gpu),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--mask-prob", str(args.mask_prob),
    ]
    if args.no_wandb:
        sys.argv.append("--no-wandb")
    if args.gpu_ids:
        sys.argv.extend(["--gpu-ids", args.gpu_ids])
    train_main()


def cmd_train_contrastive(args):
    """Contrastive fine-tuning."""
    from scripts.contrastive_finetune import main as train_main
    sys.argv = [
        "contrastive_finetune.py",
        "--data-path", args.data_path,
        "--checkpoint-path", args.checkpoint,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--n-gpu", str(args.n_gpu),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--proj-dim", str(args.proj_dim),
        "--temperature", str(args.temperature),
    ]
    if args.no_wandb:
        pass  # contrastive uses --use-wandb flag
    if args.gpu_ids:
        sys.argv.extend(["--gpu-ids", args.gpu_ids])
    train_main()


def cmd_baseline_dnabert2(args):
    """Run DNABERT-2 baseline."""
    import numpy as np
    from src.baselines import DNABERT2Baseline
    from src.data import KmerTokenizer, FastqKmerDataset
    from src.evaluation import run_full_evaluation
    from torch.utils.data import DataLoader
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    tokenizer = KmerTokenizer(k=6)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)
    
    # Subsample for efficiency
    n = min(len(dataset), args.num_samples)
    from torch.utils.data import Subset
    indices = list(range(n))
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=lambda b: (
            __import__("torch").stack([x[0] for x in b]),
            __import__("torch").tensor([x[1] for x in b]),
        ),
    )
    
    # Extract embeddings
    model = DNABERT2Baseline(
        pool_strategy="mean",
        freeze_backbone=True,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    embeddings = model.encode_from_dataloader(loader, tokenizer)
    
    # Save and evaluate
    np.save(os.path.join(args.output_dir, "dnabert2_embeddings.npy"), embeddings)
    results = run_full_evaluation(
        embeddings, "dnabert2", output_dir=args.output_dir
    )
    
    print("\nDNABERT-2 Baseline Complete!")
    return results


def cmd_baseline_kmer_pca(args):
    """Run k-mer PCA baseline."""
    import numpy as np
    from src.baselines import KmerPCABaseline
    from src.data import KmerTokenizer, FastqKmerDataset
    from src.evaluation import run_full_evaluation
    from torch.utils.data import DataLoader, Subset
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = KmerTokenizer(k=6)
    dataset = FastqKmerDataset(args.data_path, tokenizer, max_len=150)
    
    n = min(len(dataset), args.num_samples)
    subset = Subset(dataset, list(range(n)))
    
    loader = DataLoader(
        subset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=lambda b: (
            __import__("torch").stack([x[0] for x in b]),
            __import__("torch").tensor([x[1] for x in b]),
        ),
    )
    
    for n_components in [32, 64, 128]:
        print(f"\n--- K-mer PCA (n_components={n_components}) ---")
        baseline = KmerPCABaseline(k=6, n_components=n_components)
        embeddings = baseline.encode_from_dataloader(loader, tokenizer, fit=True)
        
        np.save(
            os.path.join(args.output_dir, f"kmer_pca_{n_components}_embeddings.npy"),
            embeddings,
        )
        run_full_evaluation(
            embeddings,
            f"kmer_pca_{n_components}",
            output_dir=args.output_dir,
        )


def cmd_ablation(args):
    """Run ablation studies."""
    from src.evaluation.ablation import (
        generate_codebook_size_ablation,
        generate_code_dim_ablation,
        generate_kmer_size_ablation,
        generate_loss_ablation,
        generate_contrastive_ablation,
        generate_masking_ablation,
    )
    
    study_generators = {
        "codebook_size": generate_codebook_size_ablation,
        "code_dimension": generate_code_dim_ablation,
        "kmer_size": generate_kmer_size_ablation,
        "loss_components": generate_loss_ablation,
        "contrastive": generate_contrastive_ablation,
        "masking": generate_masking_ablation,
    }
    
    if args.study not in study_generators:
        print(f"Available studies: {list(study_generators.keys())}")
        return
    
    configs = study_generators[args.study]()
    print(f"\nAblation study: {args.study}")
    print(f"Number of configurations: {len(configs)}")
    
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {cfg.name}: {cfg.description}")
        print(f"{'='*60}")
        
        exp_dir = os.path.join(args.output_dir, args.study, cfg.name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        import json
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        
        print(f"  Config saved to {exp_dir}/config.json")
        print(f"  To train: python scripts/vqvae_train.py "
              f"--data-path {args.data_path} "
              f"--output-dir {exp_dir} "
              f"--num-codes {cfg.num_codes} "
              f"--code-dim {cfg.code_dim} "
              f"--epochs {cfg.epochs} "
              f"--seed {cfg.seed}")


def cmd_evaluate_all(args):
    """Evaluate all trained models."""
    import json
    from src.evaluation import run_full_evaluation, print_comparison_table
    import numpy as np
    
    results_dir = args.results_dir
    all_results = {}
    
    # Find all embedding files
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith("_embeddings.npy"):
                model_name = f.replace("_embeddings.npy", "")
                emb_path = os.path.join(root, f)
                print(f"\nFound: {model_name} at {emb_path}")
                
                embeddings = np.load(emb_path)
                results = run_full_evaluation(
                    embeddings, model_name, output_dir=os.path.join(root, "eval")
                )
                all_results[model_name] = results
    
    if all_results:
        print_comparison_table(all_results)
        
        # Save combined results
        combined_path = os.path.join(results_dir, "combined_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train VQ-VAE
    p = subparsers.add_parser("train-vqvae", help="Train base VQ-VAE")
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=50)
    
    # Train Masked VQ-VAE
    p = subparsers.add_parser("train-masked", help="Train masked VQ-VAE")
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--mask-prob", type=float, default=0.2)
    
    # Contrastive fine-tuning
    p = subparsers.add_parser("train-contrastive", help="Contrastive fine-tuning")
    add_common_args(p)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.5)
    
    # DNABERT-2 baseline
    p = subparsers.add_parser("baseline-dnabert2", help="DNABERT-2 baseline")
    add_common_args(p)
    p.add_argument("--num-samples", type=int, default=10000)
    
    # K-mer PCA baseline
    p = subparsers.add_parser("baseline-kmer-pca", help="K-mer PCA baseline")
    add_common_args(p)
    p.add_argument("--num-samples", type=int, default=10000)
    
    # Ablation studies
    p = subparsers.add_parser("ablation", help="Run ablation studies")
    add_common_args(p)
    p.add_argument("--study", type=str, required=True,
                   choices=["codebook_size", "code_dimension", "kmer_size",
                            "loss_components", "contrastive", "masking"])
    
    # Evaluate all
    p = subparsers.add_parser("evaluate-all", help="Evaluate all models")
    p.add_argument("--results-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    commands = {
        "train-vqvae": cmd_train_vqvae,
        "train-masked": cmd_train_masked,
        "train-contrastive": cmd_train_contrastive,
        "baseline-dnabert2": cmd_baseline_dnabert2,
        "baseline-kmer-pca": cmd_baseline_kmer_pca,
        "ablation": cmd_ablation,
        "evaluate-all": cmd_evaluate_all,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
