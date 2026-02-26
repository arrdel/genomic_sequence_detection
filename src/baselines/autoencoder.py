#!/usr/bin/env python3
"""
Standard Autoencoder Baseline (no vector quantization)

A deterministic autoencoder with the same encoder/decoder architecture as VQ-VAE 
but without the discrete bottleneck. Serves as an ablation to isolate the effect
of vector quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAutoencoder(nn.Module):
    """
    Standard autoencoder baseline with continuous bottleneck.
    
    Same encoder/decoder architecture as VQ-VAE but:
    - No vector quantization
    - No codebook
    - Continuous latent space (optionally with bottleneck dim reduction)
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        bottleneck_dim: int = None,  # If set, adds extra compression
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim or latent_dim
        
        # Encoder
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=1),
        )
        self.encoder_post = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
        )
        
        # Optional bottleneck
        if bottleneck_dim != latent_dim:
            self.bottleneck_down = nn.Linear(latent_dim, bottleneck_dim)
            self.bottleneck_up = nn.Linear(bottleneck_dim, latent_dim)
        else:
            self.bottleneck_down = None
            self.bottleneck_up = None
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def encode(self, x):
        """Encode input tokens to latent representations."""
        emb = self.token_emb(x)               # (B, L, E)
        h = emb.transpose(1, 2)               # (B, E, L)
        h = self.encoder_conv(h)              # (B, D, L)
        h = h.transpose(1, 2)                 # (B, L, D)
        z = self.encoder_post(h)              # (B, L, D)
        
        if self.bottleneck_down is not None:
            z = self.bottleneck_down(z)        # (B, L, B_dim)
        
        return z
    
    def decode(self, z):
        """Decode latent representations to logits."""
        if self.bottleneck_up is not None:
            z = self.bottleneck_up(z)          # (B, L, D)
        
        h = z.transpose(1, 2)                 # (B, D, L)
        h = self.decoder_conv(h)              # (B, E, L)
        h = h.transpose(1, 2)                 # (B, L, E)
        h = self.out_norm(h)
        logits = self.output_proj(h)           # (B, L, V)
        return logits
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            logits: (B, L, V) reconstruction logits
            z: (B, L, D) latent representations
        """
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z
    
    def get_embeddings(self, x, pool: str = "mean"):
        """Extract sequence-level embeddings for downstream tasks."""
        z = self.encode(x)  # (B, L, D)
        
        # Create mask for non-padding positions
        mask = (x != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        
        if pool == "mean":
            emb = (z * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        elif pool == "max":
            z_masked = z.masked_fill(~mask.bool(), float('-inf'))
            emb = z_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {pool}")
        
        return emb  # (B, D)
