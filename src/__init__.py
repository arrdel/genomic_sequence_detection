"""
VQ-VAE for Genomics

A Vector Quantized Variational Autoencoder (VQ-VAE) implementation for 
genomic sequence analysis and reconstruction.

This package provides:
- Model architectures (VQ-VAE, Encoder, Decoder)
- Data processing and tokenization
- Utility functions for training and logging
"""

__version__ = "0.1.0"

from .models import VQVAE, VectorQuantizer, Encoder, Decoder
from .data import KmerTokenizer, FastqKmerDataset
from .utils import init_wandb, log_metrics, finish_run

__all__ = [
    'VQVAE', 
    'VectorQuantizer', 
    'Encoder', 
    'Decoder',
    'KmerTokenizer', 
    'FastqKmerDataset',
    'init_wandb', 
    'log_metrics', 
    'finish_run'
]
