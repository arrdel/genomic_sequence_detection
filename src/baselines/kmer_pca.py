#!/usr/bin/env python3
"""
K-mer PCA Baseline

Simple baseline: represent sequences as k-mer frequency vectors, then apply
PCA for dimensionality reduction. Evaluates the same downstream tasks to show
what a non-learned approach achieves.
"""

import numpy as np
from collections import Counter
from itertools import product
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class KmerPCABaseline:
    """
    K-mer frequency + PCA baseline for sequence representation.
    
    Pipeline:
    1. Count k-mer frequencies in each sequence
    2. Normalize to relative frequencies
    3. Apply PCA for dimensionality reduction
    """
    
    def __init__(
        self,
        k: int = 6,
        n_components: int = 64,
        normalize: bool = True,
        use_canonical: bool = False,
    ):
        self.k = k
        self.n_components = n_components
        self.normalize = normalize
        self.use_canonical = use_canonical
        
        # Build k-mer vocabulary
        all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
        if use_canonical:
            canonical = set()
            for kmer in all_kmers:
                rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                canonical.add(min(kmer, rc))
            self.kmers = sorted(canonical)
        else:
            self.kmers = all_kmers
        
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.kmers)}
        self.vocab_size = len(self.kmers)
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False
    
    def _sequence_to_kmer_vector(self, seq: str) -> np.ndarray:
        """Convert a DNA sequence to k-mer frequency vector."""
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]
            if set(kmer) <= {"A", "C", "G", "T"}:
                if self.use_canonical:
                    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    kmer = min(kmer, rc)
                idx = self.kmer_to_idx.get(kmer)
                if idx is not None:
                    counts[idx] += 1
        
        # Normalize to relative frequencies
        total = counts.sum()
        if total > 0 and self.normalize:
            counts /= total
        
        return counts
    
    def fit(self, sequences: List[str]) -> "KmerPCABaseline":
        """Fit PCA on k-mer frequency vectors."""
        print(f"Computing {self.k}-mer frequencies for {len(sequences)} sequences...")
        X = np.array([self._sequence_to_kmer_vector(s) for s in sequences])
        
        print(f"Fitting PCA ({self.vocab_size}d → {self.n_components}d)...")
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self._is_fitted = True
        
        explained_var = self.pca.explained_variance_ratio_.sum() * 100
        print(f"  ✓ PCA fitted, explained variance: {explained_var:.1f}%")
        
        return self
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences to PCA embeddings."""
        assert self._is_fitted, "Must call fit() first"
        
        X = np.array([self._sequence_to_kmer_vector(s) for s in sequences])
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, sequences: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(sequences)
        X = np.array([self._sequence_to_kmer_vector(s) for s in sequences])
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def encode_from_dataloader(
        self, 
        dataloader, 
        tokenizer_kmer,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings from a DataLoader of k-mer tokenized sequences.
        Reconstructs nucleotide sequences from k-mer tokens first.
        """
        sequences = []
        for batch_tokens, batch_lengths in dataloader:
            for i in range(batch_tokens.size(0)):
                length = int(batch_lengths[i].item())
                token_ids = batch_tokens[i, :length].numpy()
                seq = tokenizer_kmer.decode_tokens(
                    token_ids, remove_pad=True, reconstruct=True
                )
                sequences.append(seq)
        
        if fit:
            return self.fit_transform(sequences)
        else:
            return self.transform(sequences)
