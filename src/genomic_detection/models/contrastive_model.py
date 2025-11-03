"""
Contrastive learning model for genomic sequence analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GenomicEncoder(nn.Module):
    """
    Neural network encoder for genomic sequences using 1D convolutions.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 512,
        kernel_size: int = 7,
        dropout: float = 0.1
    ):
        """
        Initialize the genomic encoder.
        
        Args:
            input_channels: Number of input channels (5 for one-hot encoded nucleotides)
            hidden_dims: Tuple of hidden layer dimensions
            embedding_dim: Dimension of the output embedding
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Projection to embedding space
        x = self.projection(x)
        
        # L2 normalization for contrastive learning
        x = F.normalize(x, p=2, dim=1)
        
        return x


class VariantDetectionHead(nn.Module):
    """
    Detection head for identifying variants in genomic sequences.
    """
    
    def __init__(self, embedding_dim: int = 512, num_classes: int = 10):
        """
        Initialize the variant detection head.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of variant classes to detect
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for variant detection.
        
        Args:
            embeddings: Input embeddings of shape (batch, embedding_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(embeddings)


class ContrastiveGenomicModel(nn.Module):
    """
    Complete contrastive learning model for genomic variant detection.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 512,
        num_variant_classes: int = 10,
        temperature: float = 0.07
    ):
        """
        Initialize the contrastive genomic model.
        
        Args:
            input_channels: Number of input channels
            hidden_dims: Hidden dimensions for the encoder
            embedding_dim: Embedding dimension
            num_variant_classes: Number of variant classes
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        
        self.encoder = GenomicEncoder(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim
        )
        
        self.detection_head = VariantDetectionHead(
            embedding_dim=embedding_dim,
            num_classes=num_variant_classes
        )
        
        self.temperature = temperature
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences of shape (batch, seq_len, input_channels)
            return_embeddings: Whether to return embeddings along with predictions
            
        Returns:
            Tuple of (variant_predictions, embeddings) if return_embeddings is True,
            otherwise just variant_predictions
        """
        # Get embeddings
        embeddings = self.encoder(x)
        
        # Get variant predictions
        predictions = self.detection_head(embeddings)
        
        if return_embeddings:
            return predictions, embeddings
        return predictions, None
    
    def compute_contrastive_loss(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between pairs of embeddings.
        
        Args:
            embeddings1: First set of embeddings (batch, embedding_dim)
            embeddings2: Second set of embeddings (batch, embedding_dim)
            labels: Binary labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss value
        """
        # Compute cosine similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Contrastive loss: maximize similarity for positive pairs, minimize for negative
        positive_loss = labels * (1 - similarity)
        negative_loss = (1 - labels) * torch.clamp(similarity, min=0.0)
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input sequences.
        
        Args:
            x: Input sequences of shape (batch, seq_len, input_channels)
            
        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.encoder(x)
    
    def detect_variants(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect variants in input sequences.
        
        Args:
            x: Input sequences of shape (batch, seq_len, input_channels)
            
        Returns:
            Variant predictions of shape (batch, num_classes)
        """
        embeddings = self.encoder(x)
        predictions = self.detection_head(embeddings)
        return predictions


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: Embeddings of augmented samples 1
            z_j: Embeddings of augmented samples 2
            
        Returns:
            NT-Xent loss value
        """
        batch_size = z_i.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, embedding_dim)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        
        # Create mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: positive pairs are at distance batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
