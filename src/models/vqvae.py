import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Vector Quantizer with EMA
# -----------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=512, code_dim=64,
                 commitment_cost=0.1, decay=0.90, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e):  # (B, L, D)
        B, L, D = z_e.shape
        assert D == self.code_dim
        flat = z_e.reshape(-1, D)  # (N, D)

        # distances to codes
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

                dw = one_hot.t().float() @ flat  # (K, D)
                self.ema_w.mul_(self.decay).add_(dw * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + self.eps) /
                                (n + self.num_codes * self.eps) * n)

                self.embedding.weight.data.copy_(
                    self.ema_w / (cluster_size.unsqueeze(1) + self.eps)
                )

        # commitment loss only (codes updated via EMA)
        commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = self.commitment_cost * commit

        return z_q_st, loss_vq, codes.view(B, L)

    def perplexity(self, codes):
        with torch.no_grad():
            hist = torch.bincount(codes.reshape(-1),
                                  minlength=self.num_codes).float()
            p = hist / (hist.sum() + 1e-9)
            H = -(p * (p + 1e-9).log()).sum()
            return (H.exp()).item()


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 code_dim=64, pad_id=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=1)
        )

        # stabilize latent distribution
        self.post = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):  # x: (B, L)
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L), got {x.shape}")

        emb = self.token_emb(x)      # (B, L, E)
        h = emb.transpose(1, 2)      # (B, E, L)
        h = self.conv(h)             # (B, D, L)
        h = h.transpose(1, 2)        # (B, L, D)
        z = self.post(h)             # (B, L, D)
        return z


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 code_dim=64, pad_id=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU()
        )
        # normalize before logits to avoid huge logits → single-token collapse
        self.out_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, z_q):  # z_q: (B, L, D)
        h = z_q.transpose(1, 2)   # (B, D, L)
        h = self.conv(h)          # (B, E, L)
        h = h.transpose(1, 2)     # (B, L, E)
        h = self.out_norm(h)
        logits = self.output_proj(h)  # (B, L, V)
        return logits


# -----------------------------
# Full VQ-VAE model
# -----------------------------
class VQVAE(nn.Module):
    def __init__(self, vocab_size, pad_id,
                 num_codes=512, code_dim=64,
                 embed_dim=128, hidden_dim=256,
                 commitment_cost=0.1, decay=0.95):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)
        self.vq = VectorQuantizerEMA(num_codes, code_dim,
                                     commitment_cost=commitment_cost,
                                     decay=decay)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, code_dim, pad_id)

    def forward(self, x):  # x: (B, L)
        # 1. Encode
        z_e = self.encoder(x)  # (B, L, D)

        # 2. Add noise during training to encourage code usage / avoid collapse
        if self.training:
            z_e = z_e + 0.1 * torch.randn_like(z_e)

        # 3. Quantize
        z_q, loss_vq, codes = self.vq(z_e)

        # 4. Decode
        logits = self.decoder(z_q)

        return logits, loss_vq, codes