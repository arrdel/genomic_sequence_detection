# MaskedVQ-Seq: Discrete Representation Learning for Genomic Surveillance

A framework for unsupervised viral variant detection in wastewater genomic sequencing data. MaskedVQ-Seq learns interpretable codebooks of genomic patterns from canonical k-mer tokenized sequences using Vector-Quantized Variational Autoencoders (VQ-VAE), extended with masked reconstruction pretraining, entropy-regularized codebook learning, and contrastive fine-tuning.

## Overview

Wastewater-based genomic surveillance requires methods that can detect emerging viral variants without prior labeled examples. MaskedVQ-Seq addresses this by learning discrete representations that capture meaningful genomic variation through three stages:

1. **VQ-VAE pretraining** with EMA codebook updates and entropy regularization for full codebook utilization
2. **Masked reconstruction** (BERT-style, p=0.20) to learn contextual sequence representations
3. **Contrastive fine-tuning** (SimCLR) to produce embeddings that separate biologically distinct sequences

The model operates on canonical 6-mer tokenized sequences (vocabulary of 2,080 tokens) and achieves strong unsupervised clustering of viral lineages with only ~450K parameters.

## Project Structure

```
genomic_sequence_detection/
├── configs/
│   ├── default_config.yaml              # Default hyperparameters
│   └── experiment_configs/
│       └── large_model.yaml             # Large model configuration
├── scripts/
│   ├── Training
│   │   ├── vqvae_train.py               # Train base VQ-VAE
│   │   ├── mqvae_train.py               # Train Masked VQ-VAE
│   │   ├── contrastive_finetune.py      # Contrastive fine-tuning (64-dim)
│   │   ├── contrastive_finetune_128dim.py  # Contrastive fine-tuning (128-dim)
│   │   ├── train_autoencoder.py         # Standard autoencoder baseline
│   │   └── train_transformer_vae.py     # Transformer VAE baseline (DDP)
│   ├── Evaluation
│   │   ├── evaluation.py                # Model comparison evaluation
│   │   ├── vqvae_evaluate.py            # VQ-VAE evaluation
│   │   ├── mqvae_evaluate.py            # Masked VQ-VAE evaluation
│   │   ├── contrastive_evaluate.py      # Contrastive model evaluation
│   │   ├── run_full_downstream_eval.py  # Full 8-model downstream evaluation
│   │   └── extract_embeddings.py        # Embedding extraction
│   ├── Ablation Studies
│   │   ├── run_ablation_study.py        # Ablation study runner (5 studies, DDP)
│   │   ├── run_entropy_ablation.py      # Entropy regularization lambda sweep
│   │   ├── run_all_ablations.sh         # Sequential ablation launcher
│   │   └── launch_ablation_study.sh     # DDP ablation launcher
│   ├── Visualization
│   │   ├── generate_paper_figures.py    # Publication-quality figures
│   │   ├── analyze_codebook.py          # Codebook utilization analysis
│   │   ├── create_4model_visualization.py  # t-SNE/UMAP comparisons
│   │   └── visualize_clustering_improvements.py
│   ├── Pipelines
│   │   ├── run_experiment.py            # Unified experiment CLI
│   │   ├── masked_vq_train.sh           # Full overnight training pipeline
│   │   ├── run_all_overnight.sh         # Multi-GPU overnight pipeline
│   │   ├── run_all_ailab1.sh            # RTX 4090 multi-GPU pipeline
│   │   └── launch_downstream_eval.sh    # Downstream evaluation launcher
│   └── preprocess.py                    # Data preprocessing (Trimmomatic + FastQC)
├── src/
│   ├── models/
│   │   └── vqvae.py                     # VQ-VAE with EMA codebook + entropy bonus
│   ├── data/
│   │   └── tokenizer.py                 # KmerTokenizer + FastqKmerDataset
│   ├── baselines/
│   │   ├── dnabert2.py                  # DNABERT-2 pretrained baseline
│   │   ├── autoencoder.py               # Standard autoencoder (no VQ)
│   │   ├── transformer_vae.py           # Transformer VAE with attention
│   │   └── kmer_pca.py                  # K-mer frequency + PCA baseline
│   ├── evaluation/
│   │   ├── __init__.py                  # Clustering, linear probe, retrieval, anomaly
│   │   └── ablation.py                  # Ablation study configuration generator
│   └── utils/
│       ├── shuffle_sequences.py         # Sequence shuffling utilities
│       └── wandb_init.py                # Weights & Biases initialization
├── experiments/                          # Experiment outputs (gitignored)
└── requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (multi-GPU supported via DDP)

## Installation

```bash
git clone https://github.com/arrdel/genomic_sequence_detection.git
cd genomic_sequence_detection
pip install -r requirements.txt
```

## Usage

### Data Preparation

Download and preprocess SARS-CoV-2 wastewater sequencing data:

```bash
python scripts/preprocess.py \
    --input-dir /path/to/raw/fastq \
    --output-dir /path/to/processed
```

### Training

**Full pipeline** (all stages sequentially):

```bash
bash scripts/masked_vq_train.sh
```

**Individual stages:**

```bash
# Stage 1: Base VQ-VAE
python scripts/vqvae_train.py \
    --data-path /path/to/processed/cleaned_reads.fastq \
    --output-dir experiments/1_standard_vqvae

# Stage 2: Masked VQ-VAE
python scripts/mqvae_train.py \
    --data-path /path/to/processed/cleaned_reads.fastq \
    --output-dir experiments/2_masked_vqvae

# Stage 3: Contrastive fine-tuning
python scripts/contrastive_finetune.py \
    --data-path /path/to/processed/cleaned_reads.fastq \
    --output-dir experiments/3_contrastive_vqvae
```

**Multi-GPU training** (DDP):

```bash
bash scripts/run_all_ailab1.sh
```

### Evaluation

```bash
# Full downstream evaluation (all 8 models)
python scripts/run_full_downstream_eval.py \
    --results-dir experiments

# Individual model evaluation
python scripts/vqvae_evaluate.py --checkpoint experiments/1_standard_vqvae/checkpoints/best.pt
```

### Ablation Studies

```bash
# Run all 5 ablation studies
bash scripts/run_all_ablations.sh

# Run a specific study
python scripts/run_ablation_study.py --study codebook_size

# Entropy regularization sweep
python scripts/run_entropy_ablation.py
```

## Models

### Proposed Methods

| Model | Description | Parameters |
|-------|-------------|------------|
| **VQ-VAE** | Base model with EMA codebook (K=512, D=64) | ~450K |
| **Masked VQ-VAE** | + masked reconstruction pretraining (p=0.20) | ~450K |
| **Contrastive VQ-VAE (64d)** | + SimCLR fine-tuning with 64-dim projection | ~450K |
| **Contrastive VQ-VAE (128d)** | + SimCLR fine-tuning with 128-dim projection | ~450K |

### Baselines

| Model | Type |
|-------|------|
| **DNABERT-2** | Pretrained transformer (feature extraction + linear probe) |
| **Standard AE** | Deterministic autoencoder without vector quantization |
| **Transformer VAE** | Continuous VAE with multi-head attention |
| **K-mer PCA** | Classical k-mer frequency vectors + PCA |

## Evaluation

All models are evaluated across five complementary metric families:

| Metric Family | Metrics |
|---------------|---------|
| **Clustering** | Silhouette score, Davies-Bouldin index, Calinski-Harabasz index (k=5,10,15,20) |
| **Linear Probing** | Logistic regression, KNN, SVM accuracy (5-fold CV) |
| **Retrieval** | Precision@k for nearest-neighbor search |
| **Anomaly Detection** | AUROC, AUPRC for rare variant identification |
| **Embedding Quality** | Uniformity, isotropy, effective dimensionality |

Results are reported as mean and standard deviation over 5 random seeds.

## Ablation Studies

Six ablation studies characterize the contribution of each design decision:

| Study | Search Space |
|-------|-------------|
| Codebook size | K = {64, 128, 256, **512**, 1024, 2048} |
| Code dimension | D = {16, 32, **64**, 128, 256} |
| K-mer size | k = {4, 5, **6**, 7, 8} |
| Masking probability | p = {0.05, 0.10, 0.15, **0.20**, 0.25, 0.30} |
| Contrastive temperature | tau = {0.1, 0.3, **0.5**, 0.7, 1.0} |
| Entropy regularization | lambda = {0.0, 0.001, **0.003**, 0.01, 0.1} |

Bold values indicate defaults used in the final model.

## Configuration

Hyperparameters are managed via YAML configuration files:

```bash
# Use default configuration
python scripts/run_experiment.py train-vqvae

# Override with experiment-specific config
python scripts/run_experiment.py train-vqvae --config configs/experiment_configs/large_model.yaml
```

See `configs/default_config.yaml` for the full list of configurable parameters.

## Citation

```bibtex
@inproceedings{chinda2025maskedvqseq,
  title={MaskedVQ-Seq: Discrete Representation Learning for Unsupervised 
         Viral Variant Detection in Wastewater Sequencing},
  author={Chinda, Adele and Azumah, Richmond and Venkateswara, Hemanth},
  year={2025}
}
```

## References

- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). NeurIPS.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). ICML.
- Zhou, Z., et al. (2024). [DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome](https://arxiv.org/abs/2306.15006). ICLR.

## License

MIT

