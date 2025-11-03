# Genomic Sequence Detection using Contrastive Deep Learning

A PyTorch-based framework for detecting genetic variants in wastewater genomic sequencing data using contrastive learning approaches.

## Overview

This repository implements a state-of-the-art deep learning system for variant detection in wastewater genomic sequences. The system uses contrastive learning to learn robust representations of genomic sequences, enabling effective detection and classification of genetic variants.

### Key Features

- **Contrastive Learning**: Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for learning discriminative sequence representations
- **Genomic Encoder**: 1D CNN-based encoder specifically designed for genomic sequences
- **Variant Detection**: Multi-class classification head for identifying different variant types
- **Data Augmentation**: Built-in sequence augmentation strategies for contrastive learning
- **Flexible Architecture**: Configurable model dimensions and hyperparameters
- **Comprehensive Utilities**: Tools for visualization, evaluation, and analysis

## Architecture

The framework consists of three main components:

1. **Genomic Encoder**: A convolutional neural network that encodes DNA/RNA sequences into fixed-size embeddings
2. **Contrastive Learning Module**: Learns to distinguish between similar and dissimilar sequences using contrastive loss
3. **Variant Detection Head**: Classifies sequences into different variant categories

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional, but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/arrdel/genomic_sequence_detection.git
cd genomic_sequence_detection

# Install dependencies
pip install -r requirements.txt

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Quick Start

### Training a Model

Train a contrastive model on your genomic sequences:

```bash
python examples/train.py \
    --data_path path/to/sequences.fasta \
    --output_dir outputs \
    --epochs 20 \
    --batch_size 32 \
    --embedding_dim 512
```

If you don't have data, the script will generate synthetic sequences for demonstration:

```bash
python examples/train.py --epochs 5
```

### Running Inference

Detect variants in new sequences using a trained model:

```bash
python examples/inference.py \
    --model_path outputs/best_model.pth \
    --input_path path/to/test_sequences.fasta \
    --visualize \
    --compute_similarity
```

## Usage Examples

### Loading Genomic Sequences

```python
from genomic_detection import SequenceDataLoader

# Initialize data loader
loader = SequenceDataLoader(batch_size=32, max_length=1000)

# Load sequences from FASTA file
sequences = loader.load_from_fasta("sequences.fasta")

# Create PyTorch DataLoader
data_loader = loader.get_dataloader(sequences, shuffle=True)
```

### Creating and Training a Model

```python
import torch
from genomic_detection import ContrastiveGenomicModel, ContrastiveTrainer
from genomic_detection.models import NTXentLoss

# Create model
model = ContrastiveGenomicModel(
    input_channels=5,
    hidden_dims=(64, 128, 256),
    embedding_dim=512,
    num_variant_classes=10
)

# Initialize trainer
trainer = ContrastiveTrainer(
    model=model,
    device="cuda",
    learning_rate=1e-4
)

# Train with contrastive loss
contrastive_loss = NTXentLoss(temperature=0.07)
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    contrastive_loss_fn=contrastive_loss,
    save_path="best_model.pth"
)
```

### Variant Detection

```python
# Load trained model
model = ContrastiveGenomicModel(embedding_dim=512)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions
with torch.no_grad():
    predictions = model.detect_variants(sequences)
    variant_classes = torch.argmax(predictions, dim=1)
```

### Visualizing Embeddings

```python
from genomic_detection.utils import visualize_embeddings

# Get embeddings
embeddings = model.get_embeddings(sequences)

# Visualize using t-SNE
visualize_embeddings(
    embeddings.cpu().numpy(),
    labels=["Variant_A", "Variant_B", "Variant_C"],
    save_path="embeddings_visualization.png"
)
```

## Project Structure

```
genomic_sequence_detection/
├── src/
│   └── genomic_detection/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── contrastive_model.py    # Model architectures
│       ├── data/
│       │   ├── __init__.py
│       │   └── sequence_loader.py      # Data loading and preprocessing
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py              # Training utilities
│       └── utils/
│           ├── __init__.py
│           └── helpers.py              # Utility functions
├── examples/
│   ├── train.py                         # Training script
│   └── inference.py                     # Inference script
├── tests/                               # Unit tests
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Model Architecture Details

### Genomic Encoder

The encoder uses 1D convolutions to process one-hot encoded DNA/RNA sequences:

- **Input**: One-hot encoded sequences (batch_size, seq_len, 5)
- **Conv Layers**: 3 convolutional blocks with batch normalization and max pooling
- **Hidden Dimensions**: [64, 128, 256] (configurable)
- **Global Pooling**: Adaptive average pooling
- **Projection Head**: 2-layer MLP for embedding projection
- **Output**: L2-normalized embeddings (batch_size, embedding_dim)

### Contrastive Learning

Uses NT-Xent loss to learn representations:

- Maximizes similarity between augmented views of the same sequence
- Minimizes similarity between different sequences
- Temperature-scaled similarity for better gradient flow

### Variant Detection Head

Multi-layer perceptron for classification:

- Input: Sequence embeddings
- Hidden layer with dropout for regularization
- Output: Logits for variant classes

## Configuration

Key hyperparameters that can be adjusted:

- `embedding_dim`: Dimension of sequence embeddings (default: 512)
- `hidden_dims`: Hidden layer dimensions (default: [64, 128, 256])
- `max_length`: Maximum sequence length (default: 1000)
- `temperature`: Temperature for contrastive loss (default: 0.07)
- `learning_rate`: Optimizer learning rate (default: 1e-4)
- `batch_size`: Training batch size (default: 32)

## Data Format

The framework expects genomic sequences in FASTA format:

```
>sequence_id_1
ATCGATCGATCGATCG...
>sequence_id_2
GCTAGCTAGCTAGCTA...
```

Sequences are automatically:
- Converted to uppercase
- One-hot encoded (A=0, C=1, G=2, T=3, N=4)
- Padded/truncated to `max_length`

## Evaluation

The framework provides utilities for model evaluation:

- **Classification metrics**: Precision, recall, F1-score
- **Confusion matrix**: Visualization of classification performance
- **Embedding visualization**: t-SNE plots of learned representations
- **Training curves**: Loss and timing plots

## Advanced Usage

### Custom Data Augmentation

```python
from genomic_detection.data import SequenceDataLoader

loader = SequenceDataLoader()
augmented_pairs = loader.create_contrastive_pairs(
    sequences,
    augmentation_prob=0.1  # 10% mutation rate
)
```

### Computing Sequence Similarity

```python
from genomic_detection.utils import compute_embedding_similarity

similarity = compute_embedding_similarity(
    embedding1,
    embedding2,
    metric="cosine"  # or "euclidean"
)
```

### Creating Synthetic Variants

```python
from genomic_detection.utils import create_synthetic_variants

base_sequence = "ATCGATCG" * 125  # 1000 bp
variants = create_synthetic_variants(
    base_sequence,
    num_variants=50,
    mutation_rate=0.02
)
```

## Performance Tips

1. **Use GPU**: Training is significantly faster on CUDA-enabled GPUs
2. **Batch Size**: Larger batches improve contrastive learning (if memory allows)
3. **Sequence Length**: Adjust `max_length` based on your data characteristics
4. **Data Augmentation**: Higher augmentation rates provide more training signal
5. **Temperature**: Lower temperature (0.05-0.1) works well for genomic sequences

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{genomic_sequence_detection,
  title={Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing},
  author={Genomic Detection Team},
  year={2025},
  url={https://github.com/arrdel/genomic_sequence_detection}
}
```

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

This framework is designed for wastewater genomic surveillance and variant detection applications, supporting public health monitoring and epidemiological research.