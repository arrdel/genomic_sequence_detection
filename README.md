# VQ-VAE for Genomics

A Vector Quantized Variational Autoencoder (VQ-VAE) implementation for viral genome sequence analysis and reconstruction. This project uses discrete representation learning to compress and reconstruct genomic sequences, enabling downstream tasks like anomaly detection and sequence generation.

## Project Overview

This implementation applies VQ-VAE to genomic sequences using k-mer tokenization. The model learns a discrete codebook of sequence patterns and can reconstruct sequences with high fidelity.

### Features

- **K-mer Tokenization**: Efficient sequence encoding using overlapping k-mers (default k=6)
- **VQ-VAE Architecture**: Discrete latent space for genomic sequence representation
- **Quality Control**: Integrated preprocessing pipeline with Trimmomatic and FastQC
- **Experiment Tracking**: Weights & Biases (wandb) integration for experiment monitoring
- **Comprehensive Evaluation**: Reconstruction metrics and codebook usage analysis

## Project Structure

```
project/
├── configs/                      # Configuration files
│   ├── default_config.yaml      # Default hyperparameters
│   └── experiment_configs/      # Experiment-specific configs
│       └── large_model.yaml
├── scripts/                      # Executable scripts
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Model evaluation script
│   └── preprocess.py            # Data preprocessing script
├── src/                         # Source code
│   ├── __init__.py
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── vqvae.py            # VQ-VAE implementation
│   ├── data/                    # Data processing
│   │   ├── __init__.py
│   │   └── tokenizer.py        # K-mer tokenization
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── wandb_init.py       # Logging utilities
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Java (for Trimmomatic)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Trimmomatic** (for preprocessing):
   - Trimmomatic is included in the `Trimmomatic/` directory
   - Ensure Java is installed: `java -version`

4. **Install FastQC** (optional, for quality control):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install fastqc
   
   # On macOS
   brew install fastqc
   ```

## Usage

### 1. Data Preprocessing

Preprocess raw FASTQ files using Trimmomatic:

```bash
python scripts/preprocess.py \
    --input-fastq wastewater_seq_dataset.fastq \
    --output-fastq cleaned_reads.fastq \
    --run-fastqc \
    --threads 4
```

**Options:**
- `--input-fastq`: Path to raw FASTQ file
- `--output-fastq`: Path to save cleaned sequences
- `--run-fastqc`: Run quality control before and after
- `--threads`: Number of CPU threads to use
- `--skip-trimmomatic`: Skip trimming (only run FastQC)

### 2. Training

Train the VQ-VAE model:

```bash
python scripts/train.py \
    --data-path cleaned_reads.fastq \
    --output-dir ./outputs \
    --batch-size 64 \
    --epochs 100 \
    --n-gpu 1
```

**Key Training Arguments:**

**Data:**
- `--data-path`: Path to preprocessed FASTQ file
- `--max-seq-length`: Maximum sequence length (default: 150)
- `--k-mer`: K-mer size for tokenization (default: 6)

**Model:**
- `--vocab-size`: Vocabulary size (default: 4097 = 4^6 + 1)
- `--num-codes`: Number of codebook vectors (default: 512)
- `--code-dim`: Dimension of codebook vectors (default: 64)
- `--embed-dim`: Token embedding dimension (default: 128)
- `--hidden-dim`: Hidden layer dimension (default: 256)

**Training:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 64)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--save-freq`: Save checkpoint every N epochs (default: 1)

**GPU:**
- `--n-gpu`: Number of GPUs to use (default: 1)
- `--gpu-ids`: Specific GPU IDs, e.g., "0,1,2"
- `--no-cuda`: Disable CUDA

**Logging:**
- `--wandb-project`: W&B project name (default: vqvae-genomics)
- `--experiment-name`: Name for this run
- `--no-wandb`: Disable W&B logging

### 3. Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint-path ./checkpoints/checkpoint_epoch_100.pt \
    --data-path cleaned_reads.fastq \
    --output-dir ./evaluation_results
```

**Evaluation Outputs:**
- Reconstruction accuracy metrics
- Codebook usage statistics
- Example sequence reconstructions
- JSON results file

### 4. Using Configuration Files

You can also use YAML configuration files instead of command-line arguments:

```bash
python scripts/train.py --config configs/default_config.yaml
```

Create custom experiment configs in `configs/experiment_configs/`.

## Model Architecture

### VQ-VAE Components

1. **Encoder**: 
   - Token embeddings → 1D Convolutions
   - Maps sequences to continuous latent space

2. **Vector Quantizer**:
   - Discretizes continuous representations
   - Uses codebook of learnable vectors
   - Straight-through gradient estimator

3. **Decoder**:
   - 1D Convolutions → Linear projection
   - Reconstructs sequences from discrete codes

### Loss Function

```
Total Loss = Reconstruction Loss + VQ Loss + Commitment Loss
```

- **Reconstruction Loss**: Cross-entropy between input and output
- **VQ Loss**: Codebook learning
- **Commitment Loss**: Encourages encoder to commit to codebook entries

## Experiment Tracking

This project uses Weights & Biases (wandb) for experiment tracking.

**Logged Metrics:**
- Training loss (total, reconstruction, VQ)
- Reconstruction accuracy
- Codebook utilization
- Model checkpoints

**Setup W&B:**
```bash
wandb login
```

## Results and Outputs

### Training Outputs

- `outputs/run_YYYYMMDD_HHMMSS/`
  - `checkpoint_epoch_N.pt`: Model checkpoints
  - `config.json`: Training configuration
  - `sequence_reconstructions.txt`: Example reconstructions

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

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vqvae_genomics,
  title={VQ-VAE for Genomic Sequence Analysis},
  author={Your Name},
  year={2025}
}
```

## References

- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [Trimmomatic: A flexible read trimming tool](https://academic.oup.com/bioinformatics/article/30/15/2114/2390096)

## License

This project is provided for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact [your email].

## Acknowledgments

- VQ-VAE implementation inspired by the original paper
- Preprocessing pipeline uses Trimmomatic and FastQC
- Experiment tracking powered by Weights & Biases
