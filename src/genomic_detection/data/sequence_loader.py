"""
Data loading and preprocessing for genomic sequences.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class GenomicSequence:
    """Represents a genomic sequence with metadata."""
    
    NUCLEOTIDE_ENCODING = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'N': 4,  # Unknown nucleotide
    }
    
    def __init__(self, sequence: str, variant_info: Optional[Dict] = None, sample_id: str = ""):
        """
        Initialize a genomic sequence.
        
        Args:
            sequence: DNA/RNA sequence string
            variant_info: Dictionary containing variant information
            sample_id: Identifier for the sample
        """
        self.sequence = sequence.upper()
        self.variant_info = variant_info or {}
        self.sample_id = sample_id
        
    def encode(self) -> np.ndarray:
        """
        Encode the genomic sequence into numerical representation.
        
        Returns:
            One-hot encoded numpy array of shape (len(sequence), 5)
        """
        encoded = np.zeros((len(self.sequence), 5))
        for i, nucleotide in enumerate(self.sequence):
            if nucleotide in self.NUCLEOTIDE_ENCODING:
                encoded[i, self.NUCLEOTIDE_ENCODING[nucleotide]] = 1
            else:
                encoded[i, 4] = 1  # Unknown
        return encoded
    
    def __len__(self):
        return len(self.sequence)
    
    def __repr__(self):
        return f"GenomicSequence(id={self.sample_id}, length={len(self)}, variants={len(self.variant_info)})"


class SequenceDataset(Dataset):
    """PyTorch dataset for genomic sequences."""
    
    def __init__(self, sequences: List[GenomicSequence], max_length: int = 1000):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of GenomicSequence objects
            max_length: Maximum sequence length (sequences will be padded/truncated)
        """
        self.sequences = sequences
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single item from the dataset.
        
        Returns:
            Tuple of (encoded_sequence, metadata)
        """
        seq = self.sequences[idx]
        encoded = seq.encode()
        
        # Pad or truncate to max_length
        if len(encoded) < self.max_length:
            padded = np.zeros((self.max_length, 5))
            padded[:len(encoded)] = encoded
            encoded = padded
        else:
            encoded = encoded[:self.max_length]
        
        # Convert to tensor
        encoded_tensor = torch.FloatTensor(encoded)
        
        metadata = {
            'sample_id': seq.sample_id,
            'variant_info': seq.variant_info,
            'original_length': len(seq)
        }
        
        return encoded_tensor, metadata


class SequenceDataLoader:
    """Data loader for genomic sequences with contrastive learning support."""
    
    def __init__(self, batch_size: int = 32, max_length: int = 1000, num_workers: int = 4):
        """
        Initialize the data loader.
        
        Args:
            batch_size: Batch size for training
            max_length: Maximum sequence length
            num_workers: Number of worker processes for data loading
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        
    def load_from_fasta(self, fasta_path: str) -> List[GenomicSequence]:
        """
        Load sequences from a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            List of GenomicSequence objects
        """
        sequences = []
        current_id = ""
        current_seq = []
        
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_seq:
                            sequences.append(
                                GenomicSequence(''.join(current_seq), sample_id=current_id)
                            )
                        # Start new sequence
                        current_id = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                        
                # Save last sequence
                if current_seq:
                    sequences.append(
                        GenomicSequence(''.join(current_seq), sample_id=current_id)
                    )
        except FileNotFoundError:
            print(f"Warning: FASTA file not found at {fasta_path}")
            
        return sequences
    
    def create_contrastive_pairs(
        self, 
        sequences: List[GenomicSequence],
        augmentation_prob: float = 0.1
    ) -> List[Tuple[GenomicSequence, GenomicSequence, int]]:
        """
        Create pairs of sequences for contrastive learning.
        
        Args:
            sequences: List of sequences
            augmentation_prob: Probability of nucleotide mutation for augmentation
            
        Returns:
            List of (seq1, seq2, label) tuples where label is 1 for similar, 0 for dissimilar
        """
        pairs = []
        
        # Create positive pairs (same variant with augmentation)
        for seq in sequences:
            # Create augmented version
            augmented_seq = self._augment_sequence(seq, augmentation_prob)
            pairs.append((seq, augmented_seq, 1))
        
        # Create negative pairs (different variants)
        for i in range(len(sequences)):
            for j in range(i + 1, min(i + 3, len(sequences))):  # Limit negative pairs
                pairs.append((sequences[i], sequences[j], 0))
                
        return pairs
    
    def _augment_sequence(self, seq: GenomicSequence, prob: float) -> GenomicSequence:
        """
        Augment a sequence by randomly mutating nucleotides.
        
        Args:
            seq: Original sequence
            prob: Probability of mutation per nucleotide
            
        Returns:
            Augmented GenomicSequence
        """
        nucleotides = ['A', 'C', 'G', 'T']
        augmented = []
        
        for nucleotide in seq.sequence:
            if np.random.random() < prob and nucleotide in nucleotides:
                # Mutate to a different nucleotide
                options = [n for n in nucleotides if n != nucleotide]
                augmented.append(np.random.choice(options))
            else:
                augmented.append(nucleotide)
        
        return GenomicSequence(
            ''.join(augmented),
            variant_info=seq.variant_info.copy(),
            sample_id=f"{seq.sample_id}_aug"
        )
    
    def get_dataloader(
        self, 
        sequences: List[GenomicSequence], 
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from sequences.
        
        Args:
            sequences: List of GenomicSequence objects
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = SequenceDataset(sequences, self.max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
