#!/usr/bin/env python3
"""
DNABERT-2 Baseline for Genomic Sequence Representation

Uses the pretrained DNABERT-2 model to extract embeddings, then evaluates
on the same downstream tasks (clustering, linear probing, retrieval) as VQ-VAE.

Reference:
    Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for
    Multi-Species Genome", arXiv:2306.15006, 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm


class DNABERT2Baseline(nn.Module):
    """
    DNABERT-2 baseline that extracts embeddings from pretrained model.
    
    Supports:
    - Feature extraction (frozen backbone)
    - Linear probing (train linear head on frozen features)
    - Fine-tuning (update backbone + head)
    """
    
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        pool_strategy: str = "mean",  # mean, cls, max
        proj_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.pool_strategy = pool_strategy
        self.freeze_backbone = freeze_backbone
        self.device_name = device
        
        # Lazy load to avoid import errors if transformers not installed
        self._model = None
        self._tokenizer = None
        self._hidden_dim = None
        
        # Optional projection head
        self.proj_dim = proj_dim
        self.proj_head = None
        
    def _load_model(self):
        """Lazy load DNABERT-2 model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for DNABERT-2 baseline. "
                "Install with: pip install transformers"
            )
        
        print(f"Loading DNABERT-2 from {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._hidden_dim = self._model.config.hidden_size
        
        if self.freeze_backbone:
            for param in self._model.parameters():
                param.requires_grad = False
            print("  ✓ Backbone frozen")
        
        # Create projection head if requested
        if self.proj_dim is not None:
            self.proj_head = nn.Sequential(
                nn.Linear(self._hidden_dim, self.proj_dim),
                nn.ReLU(),
                nn.Linear(self.proj_dim, self.proj_dim),
            )
            print(f"  ✓ Projection head: {self._hidden_dim} → {self.proj_dim}")
        
        self._model.to(self.device_name)
        if self.proj_head is not None:
            self.proj_head.to(self.device_name)
        
        print(f"  ✓ Model loaded ({self._hidden_dim}-dim hidden)")
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property 
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def hidden_dim(self):
        if self._hidden_dim is None:
            self._load_model()
        return self._hidden_dim
    
    def encode_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode DNA sequences into embeddings.
        
        Args:
            sequences: List of DNA sequence strings
            batch_size: Batch size for inference
            max_length: Maximum token length
            show_progress: Show progress bar
            
        Returns:
            embeddings: np.ndarray of shape (N, D)
        """
        _ = self.model  # ensure loaded
        self.model.eval()
        
        all_embeddings = []
        
        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding sequences")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch_seqs = sequences[start_idx:start_idx + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # (B, L, D)
                
                # Pool
                if self.pool_strategy == "cls":
                    embeddings = hidden_states[:, 0, :]
                elif self.pool_strategy == "mean":
                    attention_mask = inputs.get("attention_mask", None)
                    if attention_mask is not None:
                        mask = attention_mask.unsqueeze(-1).float()
                        embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
                    else:
                        embeddings = hidden_states.mean(dim=1)
                elif self.pool_strategy == "max":
                    embeddings = hidden_states.max(dim=1).values
                else:
                    raise ValueError(f"Unknown pool strategy: {self.pool_strategy}")
                
                # Optional projection
                if self.proj_head is not None:
                    embeddings = self.proj_head(embeddings)
                    embeddings = F.normalize(embeddings, dim=-1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def encode_from_dataloader(
        self,
        dataloader,
        tokenizer_kmer,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode sequences from a FastqKmerDataset DataLoader.
        
        Reconstructs the nucleotide sequences from k-mer tokens, then
        encodes them with DNABERT-2.
        """
        all_sequences = []
        for batch_tokens, batch_lengths in dataloader:
            for i in range(batch_tokens.size(0)):
                length = int(batch_lengths[i].item())
                token_ids = batch_tokens[i, :length].numpy()
                seq = tokenizer_kmer.decode_tokens(token_ids, remove_pad=True, reconstruct=True)
                all_sequences.append(seq)
        
        return self.encode_sequences(all_sequences, show_progress=show_progress)
    
    def forward(self, sequences: List[str], max_length: int = 512):
        """Forward pass for training (e.g., linear probing)."""
        _ = self.model  # ensure loaded
        
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        
        hidden_states = outputs.last_hidden_state
        
        # Pool
        if self.pool_strategy == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
        elif self.pool_strategy == "cls":
            embeddings = hidden_states[:, 0, :]
        else:
            embeddings = hidden_states.max(dim=1).values
        
        if self.proj_head is not None:
            embeddings = self.proj_head(embeddings)
            embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
