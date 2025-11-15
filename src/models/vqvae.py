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
    
    
# --- Add to vqvae.py ---
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=512, code_dim=64, commitment_cost=0.1, decay=0.95, eps=1e-5): # try cocommitment_cost = 0.25
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1/num_codes, 1/num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e):  # (B, L, D)
        B, L, D = z_e.shape
        assert D == self.code_dim
        flat = z_e.reshape(-1, D)  # (N, D)

        # compute nearest codes
        emb = self.embedding.weight  # (K, D)
        dists = (flat.pow(2).sum(dim=1, keepdim=True)
                 - 2 * flat @ emb.t()
                 + emb.pow(2).sum(dim=1, keepdim=True).t())  # (N, K)
        codes = torch.argmin(dists, dim=1)  # (N,)
        z_q = F.embedding(codes, emb).view(B, L, D)

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # EMA updates (no grad)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(codes, num_classes=self.num_codes).type_as(z_e)  # (N, K)
                cluster_size = one_hot.sum(dim=0)  # (K,)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1 - self.decay))

                # sum of encoder outputs per code
                dw = one_hot.t().float() @ flat   # (K, D)
                self.ema_w.mul_(self.decay).add_(dw * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + self.eps) /
                                (n + self.num_codes * self.eps) * n)

                # normalize to get new embeddings
                self.embedding.weight.data.copy_(self.ema_w / (cluster_size.unsqueeze(1) + self.eps))

        # losses: commitment only (embedding updated via EMA)
        commit = F.mse_loss(z_e.detach(), z_q)  # ||sg(z_e)-z_q||^2
        loss_vq = self.commitment_cost * commit

        # expose codes for logging
        return z_q_st, loss_vq, codes.view(B, L)

    def perplexity(self, codes):
        with torch.no_grad():
            hist = torch.bincount(codes.reshape(-1), minlength=self.num_codes).float()
            p = hist / (hist.sum() + 1e-9)
            H = -(p * (p + 1e-9).log()).sum()
            return (H.exp()).item()





# -----------------------------
# Simple Encoder & Decoder
# -----------------------------
# class Encoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, code_dim=64, pad_id=None):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
#         # a small 1D conv / feedforward stack to get per-position latent
#         self.conv = nn.Sequential(
#             nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(hidden_dim, code_dim, kernel_size=1)  # final projection to code_dim
#         )


# -----------------------------
# Simple Encoder & Decoder
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, code_dim=64, pad_id=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        # 1D conv stack for local context encoding
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=1)  # final projection to code_dim
        )

        # 🔧 Post-processing layer to stabilize latent distribution
        self.post = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):  # x: (B, L) token ids
        emb = self.token_emb(x)  # (B, L, E)
        h = emb.transpose(1, 2)  # (B, E, L)
        h = self.conv(h)         # (B, D, L)
        h = h.transpose(1, 2)    # (B, L, D)
        z = self.post(h)         # normalize & regularize latent codes
        return z


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
        # self.vq = VectorQuantizer(num_codes, code_dim, commitment_cost)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)
        self.vq = VectorQuantizerEMA(num_codes, code_dim, commitment_cost=0.1, decay=0.95)
        

    def forward(self, x):  # x: (B, L) token ids
        z_e = self.encoder(x)           # (B, L, D)
        z_q, loss_vq, codes = self.vq(z_e)  # (B, L, D), scalar, (B, L)
        logits = self.decoder(z_q)      # (B, L, V)
        return logits, loss_vq, codes


