# Contrastive VQ-VAE for Genomic Surveillance

Discrete representation learning for unsupervised viral variant detection in wastewater genomic sequencing data. This framework learns interpretable codebooks of genomic patterns from k-mer tokenized sequences using Vector-Quantized Variational Autoencoders (VQ-VAE), extended with masked pretraining and contrastive fine-tuning.

## Project Structure

```
├── configs/                          # Configuration files
│   ├── default_config.yaml          # Master config with all hyperparameters
│   └── experiment_configs/          # Per-experiment overrides
├── scripts/                          # Runnable scripts
│   ├── run_all_experiments.sh       # Master script: runs everything
│   ├── run_experiment.py            # Unified experiment CLI
│   ├── vqvae_train.py              # Train base VQ-VAE
│   ├── mqvae_train.py              # Train masked VQ-VAE
│   ├── contrastive_finetune.py     # Contrastive fine-tuning (64-dim)
│   ├── contrastive_finetune_128dim.py  # Contrastive fine-tuning (128-dim)
│   ├── train_autoencoder.py        # Standard AE baseline
│   ├── train_transformer_vae.py    # Transformer VAE baseline
│   ├── evaluation.py               # Model comparison evaluation
│   ├── contrastive_evaluate.py     # Contrastive model evaluation
│   ├── vqvae_evaluate.py           # VQ-VAE evaluation
│   ├── mqvae_evaluate.py           # Masked VQ-VAE evaluation
│   ├── analyze_codebook.py         # Codebook analysis & visualization
│   ├── create_4model_visualization.py  # t-SNE/UMAP comparisons
│   ├── visualize_clustering_improvements.py  # Clustering bar charts
│   └── preprocess.py               # Data preprocessing (Trimmomatic + FastQC)
├── src/                              # Source code (importable package)
│   ├── models/
│   │   └── vqvae.py                # VQ-VAE: Encoder, Decoder, VectorQuantizerEMA
│   ├── data/
│   │   └── tokenizer.py            # KmerTokenizer, FastqKmerDataset
│   ├── baselines/
│   │   ├── dnabert2.py             # DNABERT-2 pretrained baseline
│   │   ├── autoencoder.py          # Standard autoencoder (no VQ)
│   │   ├── transformer_vae.py      # Transformer VAE with continuous latent
│   │   └── kmer_pca.py             # K-mer frequency + PCA baseline
│   ├── evaluation/
│   │   ├── __init__.py             # Clustering, linear probe, retrieval, anomaly
│   │   └── ablation.py             # Ablation study config generator
│   └── utils/
│       ├── shuffle_sequences.py
│       └── wandb_init.py
├── experiments/                      # Experiment outputs (git-ignored)
├── report/                           # Paper LaTeX source
└── requirements.txt
```

## Data

Datasets are stored on scratch storage at `/media/scratch/adele/contrastive/`:
- `raw/` — Raw FASTQ files from SRA (SRR14596438–SRR14596445)
- `processed/` — Quality-controlled reads after Trimmomatic
- `external/` — Reference genome (NC_045512.2, SARS-CoV-2)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments
bash scripts/run_all_experiments.sh

# 3. Or run individual experiments:
python scripts/run_experiment.py train-vqvae \
    --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
    --output-dir /media/scratch/adele/contrastive/experiments/vqvae

# 4. Run DNABERT-2 baseline
python scripts/run_experiment.py baseline-dnabert2 \
    --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq \
    --output-dir /media/scratch/adele/contrastive/experiments/dnabert2

# 5. Run ablation studies
python scripts/run_experiment.py ablation --study codebook_size \
    --data-path /media/scratch/adele/contrastive/processed/cleaned_reads.fastq

# 6. Evaluate all models
python scripts/run_experiment.py evaluate-all \
    --results-dir /media/scratch/adele/contrastive/experiments
```

## Models

| Model | Description | Key Metric |
|-------|-------------|------------|
| **VQ-VAE** | Base model with EMA codebook | 99.52% token accuracy |
| **Masked VQ-VAE** | BERT-style masked pretraining | ~95% acc under 20% masking |
| **Contrastive (64d)** | SimCLR fine-tuning | Silhouette 0.42 (+35%) |
| **Contrastive (128d)** | Higher-dim projection | Silhouette 0.44 (+42%) |

## Baselines

| Baseline | Type | Notes |
|----------|------|-------|
| **DNABERT-2** | Pretrained transformer | Feature extraction + linear probe |
| **Standard AE** | Deterministic autoencoder | Ablation: no VQ bottleneck |
| **Transformer VAE** | Continuous VAE + attention | Stronger architecture baseline |
| **K-mer PCA** | Classical ML | Non-learned representation |

## Evaluation Suite

All models evaluated on:
1. **Clustering** — Silhouette, Davies-Bouldin, Calinski-Harabasz (k=5,10,15,20)
2. **Linear probing** — Logistic regression, KNN, SVM with 5-fold CV
3. **Retrieval** — Precision@k for nearest-neighbor search
4. **Anomaly detection** — AUROC/AUPRC for rare variant detection
5. **Embedding quality** — Uniformity, isotropy, effective dimensionality

All results reported as **mean ± std** over 5 random seeds.

## Ablation Studies

- Codebook size: K ∈ {64, 128, 256, 512, 1024, 2048}
- Code dimension: D ∈ {16, 32, 64, 128, 256}
- K-mer size: k ∈ {4, 5, 6, 7, 8}
- Loss components: reconstruction ± commitment ± entropy
- Contrastive temperature: τ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- Masking probability: p ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}

## Citation

```bibtex
@inproceedings{chinda2025contrastive,
  title={Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing},
  author={Chinda, Adele and Azumah, Richmond and Venkateswara, Hemanth},
  year={2025}
}
```
- CUDA-capable GPU (recommended)
- Java (for Trimmomatic)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd project
   ```

## License

MIT
### Evaluation Outputs

- `evaluation_results/`
  - `evaluation_results.json`: Metrics summary
  - `sequence_reconstructions.txt`: Reconstruction examples

## Tips and Best Practices

1. **Data Quality**: Always run preprocessing to ensure high-quality input sequences

2. **K-mer Selection**: 
   - k=6 works well for viral sequences
   - Larger k = more specific patterns but larger vocabulary

3. **Codebook Size**:
   - Monitor codebook utilization
   - Increase if utilization is very high (>95%)
   - Decrease if many codes are unused (<50% utilization)

4. **Training Time**:
   - Expect ~1-2 hours per 100 epochs on GPU for medium datasets
   - Use multiple GPUs for faster training

5. **Hyperparameter Tuning**:
   - Start with default config
   - Adjust based on reconstruction accuracy
   - Use experiment configs for systematic tuning

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 32`
   - Reduce model dimensions: `--hidden-dim 128`

2. **Low Reconstruction Accuracy**:
   - Train longer: `--epochs 200`
   - Increase model capacity: `--num-codes 1024`
   - Check data quality

3. **Low Codebook Utilization**:
   - Increase commitment cost: `--commitment-cost 0.5`
   - Reduce number of codes: `--num-codes 256`


## References

- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [Trimmomatic: A flexible read trimming tool](https://academic.oup.com/bioinformatics/article/30/15/2114/2390096)

