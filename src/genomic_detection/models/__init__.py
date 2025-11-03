"""Initialize models module."""

from .contrastive_model import (
    ContrastiveGenomicModel,
    GenomicEncoder,
    VariantDetectionHead,
    NTXentLoss
)

__all__ = [
    "ContrastiveGenomicModel",
    "GenomicEncoder",
    "VariantDetectionHead",
    "NTXentLoss"
]
