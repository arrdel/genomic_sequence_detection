# Suggested Improvements for Top-Conference Submission

## Summary of Changes Made

### Codebase Cleanup ✅
- **Removed**: `trash/` (3.6GB), root `__pycache__/`, `docs/`, `drawio/`, `data/` (moved to scratch), `logs/`, `wandb/`, `Trimmomatic/`
- **Removed**: 14+ redundant shell scripts (numbered pipeline scripts, one-off viz scripts)
- **Updated**: `README.md` (comprehensive, reflects new structure)
- **Updated**: `configs/default_config.yaml` (points to scratch storage)
- **Updated**: `requirements.txt` (added transformers, einops, umap-learn, scipy)
- **Updated**: `src/__init__.py` (clean imports, version bump)

### New Code Added ✅
- `src/baselines/dnabert2.py` — DNABERT-2 pretrained baseline
- `src/baselines/autoencoder.py` — Standard autoencoder (no VQ) baseline
- `src/baselines/transformer_vae.py` — Transformer VAE baseline
- `src/baselines/kmer_pca.py` — K-mer PCA baseline
- `src/evaluation/__init__.py` — Comprehensive evaluation (clustering, linear probe, retrieval, anomaly detection, embedding quality)
- `src/evaluation/ablation.py` — Ablation study configuration generator
- `scripts/run_experiment.py` — Unified experiment CLI
- `scripts/run_all_experiments.sh` — Master experiment runner
- `scripts/train_autoencoder.py` — Standard AE training
- `scripts/train_transformer_vae.py` — Transformer VAE training

---

## Detailed Experiment Plan

### 1. Stronger Baselines (Critical for Top Venue)

| Baseline | Why | Status |
|----------|-----|--------|
| **DNABERT-2** | State-of-the-art DNA foundation model; reviewers will ask "why not use a pretrained model?" | Code ready, needs training run |
| **Standard AE** | Ablation: isolates effect of vector quantization | Code ready |
| **Transformer VAE** | Shows VQ-VAE beats even a stronger-architecture continuous VAE | Code ready |
| **K-mer PCA** | Non-learned baseline establishing floor | Code ready |
| **Continuous VAE (Conv)** | Same architecture as VQ-VAE but with KL divergence | Need to add (minor modification of VQ-VAE) |

**Priority Actions:**
1. Run `python scripts/run_experiment.py baseline-dnabert2` — this is the most important baseline
2. Run Standard AE and Transformer VAE trainings
3. Add results to paper Table comparing all baselines

### 2. Ablation Studies (Critical)

The paper currently has NO ablations. Reviewers will require these:

| Ablation | Configurations | What It Shows |
|----------|---------------|---------------|
| **Codebook size K** | {64, 128, 256, **512**, 1024, 2048} | Optimal codebook capacity for genomic data |
| **Code dimension D** | {16, 32, **64**, 128, 256} | Information capacity per code |
| **K-mer size k** | {4, 5, **6**, 7, 8} | Optimal tokenization granularity |
| **Loss components** | recon only, +commit, +entropy | Justifies each loss term |
| **Contrastive τ** | {0.1, 0.3, **0.5**, 0.7, 1.0} | Temperature sensitivity |
| **Masking p** | {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30} | Optimal corruption level |
| **Augmentation strength** | Multiple mask/drop combinations | Contrastive augmentation sensitivity |

Run ablation configs: `python src/evaluation/ablation.py` generates all configs, then use `run_all_experiments.sh` Phase 3.

### 3. Additional Downstream Tasks (Important)

Currently only clustering is evaluated. Add:

| Task | Implementation | Expected Insight |
|------|---------------|-----------------|
| **Linear probing** | LogReg/KNN/SVM on frozen embeddings with CV | Shows discriminative power without fine-tuning |
| **Sequence retrieval** | Precision@k for nearest-neighbor search | Practical application: finding similar sequences |
| **Anomaly detection** | Distance-to-centroid AUROC | Novel variant detection capability |
| **Embedding quality** | Uniformity, isotropy, effective dimensionality | Intrinsic representation quality |

All implemented in `src/evaluation/__init__.py`.

### 4. Statistical Rigor (Critical)

Current results are single-run. For top venues you need:
- **5 random seeds** (42, 123, 456, 789, 1024) for all main results
- **Mean ± std** for every metric in every table
- **Statistical significance tests** (paired t-test or Wilcoxon) for key comparisons
- Phase 4 in `run_all_experiments.sh` handles multi-seed runs

### 5. Scaling Experiments

You have access to much more data than the 100K subset:
- `subset_100k.fastq` (28MB) — current
- `subset_1M.fastq` (303MB) — 10× more data
- `cleaned_reads.fastq` (29GB) — full dataset

**Experiment**: Train on 100K, 500K, 1M, 5M reads and show scaling curves for:
- Reconstruction accuracy
- Codebook utilization
- Clustering quality
- Training time

### 6. Paper-Specific Improvements

#### New Tables to Add:
1. **Table: Full Baseline Comparison** — VQ-VAE vs DNABERT-2 vs AE vs Transformer VAE vs K-mer PCA
2. **Table: Ablation Studies** — Each ablation with mean±std
3. **Table: Downstream Tasks** — Linear probe, retrieval, anomaly for all models
4. **Table: Scaling Analysis** — Performance vs. dataset size

#### New Figures to Add:
1. **Ablation sensitivity plots** — Line plots showing metric vs. hyperparameter
2. **t-SNE/UMAP with DNABERT-2** — Side-by-side with VQ-VAE variants
3. **Scaling curves** — Performance vs. data size
4. **Codebook utilization heatmap** across different K values
5. **Retrieval examples** — Show qualitative nearest-neighbor results

#### Writing Improvements:
- Remove "codebook" typo in Introduction (line after citation list)
- Section 5 (Results) currently has duplicated t-SNE paragraph
- Baseline comparison section (5.5) has vague "~" metrics — replace with precise numbers after running baselines
- Add "Limitations" section (data scope, single-virus, computational requirements)
- Strengthen "Contributions" to highlight ablations and baselines

---

## Execution Order (Recommended)

```
1. pip install transformers einops umap-learn scipy  (new dependencies)
2. Run DNABERT-2 baseline first (most impactful addition)
3. Run Standard AE and Transformer VAE baselines
4. Run key ablations (codebook size, code dim, loss components)
5. Run multi-seed experiments (5 seeds for main models)
6. Run scaling experiments (100K, 500K, 1M)
7. Run comprehensive evaluation on all models
8. Update paper with new tables/figures
```

## Quick Ablation Priority (If Time Limited)

If you can only run a subset, prioritize in this order:
1. **DNABERT-2 baseline** — most likely reviewer question
2. **Codebook size ablation** — justifies K=512
3. **Loss component ablation** — justifies each loss term
4. **Multi-seed main results** — statistical significance
5. **K-mer size ablation** — justifies k=6
