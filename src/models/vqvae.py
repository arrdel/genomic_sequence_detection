import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os

# -----------------------------
# Vector Quantizer (VQ-VAE)
# -----------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=512, code_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        # codebook embedding: (K, D)
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0/num_codes, 1.0/num_codes)

    def forward(self, z_e):
        """
        z_e: (B, L, D) continuous encoder outputs
        returns:
          z_q: (B, L, D) quantized embeddings (grad flows through z_e)
          loss_vq: scalar vq loss (codebook + commitment)
          codes: (B, L) int indices of nearest code
        """
        B, L, D = z_e.shape
        assert D == self.code_dim
        # flatten to (B*L, D)
        flat = z_e.reshape(-1, D)  # (N, D)
        # compute distances to codebook: ||x - e||^2 = x^2 + e^2 - 2 x.e
        emb = self.embedding.weight  # (K, D)
        # Efficient distance computation
        # x2: (N,1), e2: (1,K)
        x2 = torch.sum(flat**2, dim=1, keepdim=True)  # (N,1)
        e2 = torch.sum(emb**2, dim=1)  # (K,)
        # flat @ emb.T -> (N,K)
        dot = torch.matmul(flat, emb.t())
        dists = x2 + e2.unsqueeze(0) - 2.0 * dot  # (N,K)
        codes = torch.argmin(dists, dim=1)  # (N,)
        z_q_flat = F.embedding(codes, emb)  # (N, D)

        # reshape back
        z_q = z_q_flat.view(B, L, D)
        codes = codes.view(B, L)

        # losses (see VQ-VAE paper)
        # stop gradients for codebook update on the first term
        embedding_loss = F.mse_loss(z_q.detach(), z_e)  # ||sg(z_e) - e||^2
        commitment_loss = F.mse_loss(z_q, z_e.detach())  # ||z_e - sg(e)||^2
        loss_vq = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: pass gradients to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, loss_vq, codes

    def get_codebook(self):
        return self.embedding.weight.data.clone()

# -----------------------------
# Simple Encoder & Decoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, code_dim=64, pad_id=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # a small 1D conv / feedforward stack to get per-position latent
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=1)  # final projection to code_dim
        )

    def forward(self, x):  # x: (B, L) token ids
        # Check input dimensions
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape (B, L), got {x.shape}")
            
        emb = self.token_emb(x)  # (B, L, E)
        
        # Add debug prints
        # print(f"Embedding shape: {emb.shape}")
        
        # Ensure we have 3 dimensions before transpose
        if len(emb.shape) != 3:
            raise ValueError(f"Expected embedding shape (B, L, E), got {emb.shape}")
            
        # conv expects (B, E, L)
        h = emb.transpose(1, 2)  # Transpose sequence length and embedding dim
        h = self.conv(h)  # (B, D, L)
        h = h.transpose(1, 2)  # (B, L, D)
        
        # Add debug print
        # print(f"Output shape: {h.shape}")
        
        return h

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, code_dim=64, pad_id=None):
        super().__init__()
        # take quantized latents (B, L, D) -> decode per position to logits over vocab
        self.conv = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU()
        )
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, z_q):  # z_q: (B, L, D)
        h = z_q.transpose(1, 2)  # (B, D, L)
        h = self.conv(h)
        h = h.transpose(1, 2)  # (B, L, E)
        logits = self.output_proj(h)  # (B, L, V)
        return logits

# -----------------------------
# Full VQ-VAE model
# -----------------------------
class VQVAE(nn.Module):
    def __init__(self, vocab_size, pad_id, num_codes=512, code_dim=64, embed_dim=128, hidden_dim=256, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)
        self.vq = VectorQuantizer(num_codes, code_dim, commitment_cost)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)

    def forward(self, x):  # x: (B, L) token ids
        z_e = self.encoder(x)           # (B, L, D)
        z_q, loss_vq, codes = self.vq(z_e)  # (B, L, D), scalar, (B, L)
        logits = self.decoder(z_q)      # (B, L, V)
        return logits, loss_vq, codes
