"""
Genomic Sequence Detection using Contrastive Deep Learning

A framework for detecting genetic variants in wastewater genomic sequencing data
using contrastive learning approaches.
"""

__version__ = "0.1.0"
__author__ = "Genomic Detection Team"

from .models.contrastive_model import ContrastiveGenomicModel
from .data.sequence_loader import SequenceDataLoader
from .training.trainer import ContrastiveTrainer

__all__ = [
    "ContrastiveGenomicModel",
    "SequenceDataLoader",
    "ContrastiveTrainer",
]
