"""Unit tests for data loading and preprocessing."""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genomic_detection.data import GenomicSequence, SequenceDataset, SequenceDataLoader


def test_genomic_sequence_creation():
    """Test creating a genomic sequence."""
    seq = GenomicSequence("ATCGATCG", sample_id="test_seq")
    assert len(seq) == 8
    assert seq.sample_id == "test_seq"
    assert seq.sequence == "ATCGATCG"


def test_genomic_sequence_encoding():
    """Test one-hot encoding of sequences."""
    seq = GenomicSequence("ATCG")
    encoded = seq.encode()
    
    # Check shape
    assert encoded.shape == (4, 5)
    
    # Check one-hot encoding
    assert encoded[0, 0] == 1  # A
    assert encoded[1, 3] == 1  # T
    assert encoded[2, 1] == 1  # C
    assert encoded[3, 2] == 1  # G


def test_genomic_sequence_unknown_nucleotide():
    """Test handling of unknown nucleotides."""
    seq = GenomicSequence("ATCN")
    encoded = seq.encode()
    
    # Unknown nucleotide should be encoded as position 4
    assert encoded[3, 4] == 1


def test_sequence_dataset():
    """Test sequence dataset."""
    sequences = [
        GenomicSequence("ATCG" * 100, sample_id="seq1"),
        GenomicSequence("GCTA" * 100, sample_id="seq2"),
    ]
    
    dataset = SequenceDataset(sequences, max_length=500)
    assert len(dataset) == 2
    
    # Get a sample
    encoded, metadata = dataset[0]
    assert encoded.shape == (500, 5)
    assert metadata['sample_id'] == "seq1"


def test_sequence_padding():
    """Test sequence padding to max length."""
    sequences = [GenomicSequence("ATCG", sample_id="short_seq")]
    dataset = SequenceDataset(sequences, max_length=100)
    
    encoded, _ = dataset[0]
    assert encoded.shape == (100, 5)
    
    # First 4 positions should have data
    assert encoded[:4].sum() == 4
    # Rest should be zeros
    assert encoded[4:].sum() == 0


def test_sequence_truncation():
    """Test sequence truncation to max length."""
    long_seq = "ATCG" * 300  # 1200 bp
    sequences = [GenomicSequence(long_seq, sample_id="long_seq")]
    dataset = SequenceDataset(sequences, max_length=100)
    
    encoded, _ = dataset[0]
    assert encoded.shape == (100, 5)


def test_load_from_fasta():
    """Test loading sequences from FASTA file."""
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\n")
        f.write("ATCGATCGATCG\n")
        f.write(">seq2\n")
        f.write("GCTAGCTAGCTA\n")
        fasta_path = f.name
    
    try:
        loader = SequenceDataLoader()
        sequences = loader.load_from_fasta(fasta_path)
        
        assert len(sequences) == 2
        assert sequences[0].sample_id == "seq1"
        assert sequences[1].sample_id == "seq2"
        assert sequences[0].sequence == "ATCGATCGATCG"
        assert sequences[1].sequence == "GCTAGCTAGCTA"
    finally:
        Path(fasta_path).unlink()


def test_sequence_augmentation():
    """Test sequence augmentation."""
    seq = GenomicSequence("ATCG" * 100, sample_id="test")
    loader = SequenceDataLoader()
    
    augmented = loader._augment_sequence(seq, prob=0.1)
    
    # Should have same length
    assert len(augmented) == len(seq)
    
    # Should be different (with high probability)
    assert augmented.sequence != seq.sequence


def test_create_contrastive_pairs():
    """Test creating contrastive pairs."""
    sequences = [
        GenomicSequence("ATCG" * 50, sample_id="seq1"),
        GenomicSequence("GCTA" * 50, sample_id="seq2"),
    ]
    
    loader = SequenceDataLoader()
    pairs = loader.create_contrastive_pairs(sequences, augmentation_prob=0.1)
    
    # Should create positive pairs (one per sequence) and negative pairs
    assert len(pairs) > 0
    
    # Check structure of pairs
    seq1, seq2, label = pairs[0]
    assert isinstance(seq1, GenomicSequence)
    assert isinstance(seq2, GenomicSequence)
    assert label in [0, 1]


if __name__ == "__main__":
    test_genomic_sequence_creation()
    test_genomic_sequence_encoding()
    test_genomic_sequence_unknown_nucleotide()
    test_sequence_dataset()
    test_sequence_padding()
    test_sequence_truncation()
    test_load_from_fasta()
    test_sequence_augmentation()
    test_create_contrastive_pairs()
    
    print("All data tests passed!")
