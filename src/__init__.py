"""
Contrastive VQ-VAE for Genomic Surveillance

Discrete representation learning for unsupervised viral variant detection
in wastewater genomic sequencing data.

Modules:
- models: VQ-VAE architecture (encoder, decoder, vector quantizer)
- data: K-mer tokenization and dataset loaders
- baselines: DNABERT-2, Standard AE, Transformer VAE, K-mer PCA
- evaluation: Clustering, linear probing, retrieval, ablation studies
- utils: Logging, experiment tracking
"""

__version__ = "1.0.0"

from .models import VQVAE, Encoder, Decoder
from .data import KmerTokenizer, FastqKmerDataset

__all__ = [
    "VQVAE",
    "Encoder",
    "Decoder",
    "KmerTokenizer",
    "FastqKmerDataset",
]
