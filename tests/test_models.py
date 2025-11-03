"""Unit tests for contrastive learning models."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genomic_detection.models import (
    GenomicEncoder,
    VariantDetectionHead,
    ContrastiveGenomicModel,
    NTXentLoss
)


def test_genomic_encoder():
    """Test genomic encoder forward pass."""
    encoder = GenomicEncoder(
        input_channels=5,
        hidden_dims=(64, 128),
        embedding_dim=256
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 5)
    
    # Forward pass
    embeddings = encoder(x)
    
    assert embeddings.shape == (batch_size, 256)
    # Check L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_variant_detection_head():
    """Test variant detection head."""
    head = VariantDetectionHead(embedding_dim=256, num_classes=10)
    
    # Create dummy embeddings
    batch_size = 4
    embeddings = torch.randn(batch_size, 256)
    
    # Forward pass
    logits = head(embeddings)
    
    assert logits.shape == (batch_size, 10)


def test_contrastive_genomic_model():
    """Test complete contrastive model."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128),
        embedding_dim=256,
        num_variant_classes=10
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 5)
    
    # Forward pass without embeddings
    predictions, embeddings = model(x, return_embeddings=False)
    assert predictions.shape == (batch_size, 10)
    assert embeddings is None
    
    # Forward pass with embeddings
    predictions, embeddings = model(x, return_embeddings=True)
    assert predictions.shape == (batch_size, 10)
    assert embeddings.shape == (batch_size, 256)


def test_get_embeddings():
    """Test getting embeddings from model."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128),
        embedding_dim=256
    )
    
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 5)
    
    embeddings = model.get_embeddings(x)
    assert embeddings.shape == (batch_size, 256)


def test_detect_variants():
    """Test variant detection."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128),
        embedding_dim=256,
        num_variant_classes=10
    )
    
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 5)
    
    predictions = model.detect_variants(x)
    assert predictions.shape == (batch_size, 10)


def test_contrastive_loss():
    """Test contrastive loss computation."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        embedding_dim=256,
        temperature=0.07
    )
    
    batch_size = 4
    embedding_dim = 256
    
    # Create dummy embeddings
    embeddings1 = torch.randn(batch_size, embedding_dim)
    embeddings2 = torch.randn(batch_size, embedding_dim)
    
    # Create labels (half positive, half negative)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    
    # Compute loss
    loss = model.compute_contrastive_loss(embeddings1, embeddings2, labels)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_ntxent_loss():
    """Test NT-Xent loss."""
    loss_fn = NTXentLoss(temperature=0.07)
    
    batch_size = 4
    embedding_dim = 256
    
    # Create normalized embeddings
    z_i = torch.randn(batch_size, embedding_dim)
    z_j = torch.randn(batch_size, embedding_dim)
    z_i = torch.nn.functional.normalize(z_i, p=2, dim=1)
    z_j = torch.nn.functional.normalize(z_j, p=2, dim=1)
    
    # Compute loss
    loss = loss_fn(z_i, z_j)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_model_parameter_count():
    """Test model has reasonable number of parameters."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128, 256),
        embedding_dim=512,
        num_variant_classes=10
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Should have a reasonable number of parameters
    assert num_params > 10000
    assert num_params < 10000000


def test_model_device_transfer():
    """Test model can be transferred to different devices."""
    model = ContrastiveGenomicModel(
        input_channels=5,
        embedding_dim=128
    )
    
    # Test CPU
    model = model.to('cpu')
    x = torch.randn(2, 50, 5)
    predictions, _ = model(x)
    assert predictions.device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        x = x.to('cuda')
        predictions, _ = model(x)
        assert predictions.device.type == 'cuda'


if __name__ == "__main__":
    test_genomic_encoder()
    test_variant_detection_head()
    test_contrastive_genomic_model()
    test_get_embeddings()
    test_detect_variants()
    test_contrastive_loss()
    test_ntxent_loss()
    test_model_parameter_count()
    test_model_device_transfer()
    
    print("All model tests passed!")
