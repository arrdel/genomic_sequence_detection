#!/usr/bin/env python3
"""
Transformer-based VAE Baseline

A standard VAE with transformer encoder/decoder and continuous latent space.
Serves as a stronger baseline than the simple convolutional VAE, showing
that VQ-VAE's discrete bottleneck provides benefits even compared to
a more powerful continuous-latent architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerVAE(nn.Module):
    """
    Transformer-based VAE with continuous latent space.
    
    Uses transformer encoder to produce mu/logvar for reparameterization,
    and transformer decoder for reconstruction.
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 128,
        latent_dim: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 150,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # Token embedding + positional encoding
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # VAE: mu and logvar projections
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        # Latent to decoder input
        self.latent_to_dec = nn.Linear(latent_dim, embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def encode(self, x):
        """
        Encode input tokens to mu, logvar.
        
        Args:
            x: (B, L) input token IDs
            
        Returns:
            mu: (B, L, latent_dim)
            logvar: (B, L, latent_dim)
            encoder_output: (B, L, embed_dim)
        """
        # Create padding mask
        pad_mask = (x == self.pad_id)  # (B, L), True where padded
        
        emb = self.token_emb(x)
        emb = self.pos_enc(emb)
        
        h = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar, h
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, encoder_output, x):
        """
        Decode latent representations to logits.
        
        Args:
            z: (B, L, latent_dim) latent samples
            encoder_output: (B, L, embed_dim) from encoder
            x: (B, L) original input for padding mask
        """
        pad_mask = (x == self.pad_id)
        
        dec_input = self.latent_to_dec(z)
        dec_input = self.pos_enc(dec_input)
        
        h = self.transformer_decoder(
            dec_input,
            encoder_output,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )
        
        logits = self.output_proj(h)
        return logits
    
    def forward(self, x):
        """
        Full forward pass.
        
        Returns:
            logits: (B, L, V)
            mu: (B, L, latent_dim)
            logvar: (B, L, latent_dim)
        """
        mu, logvar, encoder_output = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, encoder_output, x)
        return logits, mu, logvar
    
    def loss(self, x, logits, mu, logvar, beta=1.0):
        """
        Compute VAE loss = reconstruction + beta * KL divergence.
        """
        # Reconstruction loss (cross-entropy over non-padding)
        mask = (x != self.pad_id).float()
        
        # Flatten for cross entropy
        B, L, V = logits.shape
        recon_loss = F.cross_entropy(
            logits.view(-1, V), x.view(-1), reduction="none"
        ).view(B, L)
        recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-9)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mask.sum() + 1e-9)
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_embeddings(self, x, pool: str = "mean"):
        """Extract sequence-level embeddings for downstream tasks."""
        mu, _, _ = self.encode(x)
        mask = (x != self.pad_id).unsqueeze(-1).float()
        
        if pool == "mean":
            emb = (mu * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        elif pool == "max":
            mu_masked = mu.masked_fill(~mask.bool(), float('-inf'))
            emb = mu_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {pool}")
        
        return emb
