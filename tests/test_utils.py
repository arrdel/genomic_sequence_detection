"""Unit tests for utility functions."""

import sys
from pathlib import Path
import torch
import numpy as np
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genomic_detection.utils import (
    compute_embedding_similarity,
    calculate_variant_statistics,
    create_synthetic_variants,
    save_sequences_to_fasta
)


def test_compute_embedding_similarity_cosine():
    """Test cosine similarity computation."""
    emb1 = torch.tensor([1.0, 0.0, 0.0])
    emb2 = torch.tensor([0.0, 1.0, 0.0])
    
    similarity = compute_embedding_similarity(emb1, emb2, metric="cosine")
    
    # Perpendicular vectors should have 0 cosine similarity
    assert abs(similarity) < 0.01


def test_compute_embedding_similarity_euclidean():
    """Test Euclidean distance computation."""
    emb1 = torch.tensor([0.0, 0.0, 0.0])
    emb2 = torch.tensor([1.0, 0.0, 0.0])
    
    similarity = compute_embedding_similarity(emb1, emb2, metric="euclidean")
    
    # Distance of 1, so similarity is -1
    assert abs(similarity + 1.0) < 0.01


def test_calculate_variant_statistics():
    """Test sequence statistics calculation."""
    sequences = [
        "ATCGATCG",
        "GCTAGCTA",
        "ATATATAT"
    ]
    
    stats = calculate_variant_statistics(sequences)
    
    assert stats['num_sequences'] == 3
    assert stats['mean_length'] == 8.0
    assert stats['min_length'] == 8
    assert stats['max_length'] == 8
    assert 'gc_content' in stats
    assert 'nucleotide_composition' in stats


def test_calculate_variant_statistics_gc_content():
    """Test GC content calculation."""
    # 50% GC content
    sequences = ["ATGC"]
    stats = calculate_variant_statistics(sequences)
    
    assert abs(stats['gc_content'] - 50.0) < 0.01


def test_create_synthetic_variants():
    """Test synthetic variant creation."""
    base_sequence = "ATCGATCG" * 10
    num_variants = 5
    
    variants = create_synthetic_variants(
        base_sequence,
        num_variants=num_variants,
        mutation_rate=0.1
    )
    
    assert len(variants) == num_variants
    
    # All variants should have same length
    for variant in variants:
        assert len(variant) == len(base_sequence)
    
    # At least some variants should be different from base
    num_different = sum(1 for v in variants if v != base_sequence)
    assert num_different > 0


def test_save_sequences_to_fasta():
    """Test saving sequences to FASTA format."""
    sequences = [
        ("seq1", "ATCGATCG"),
        ("seq2", "GCTAGCTA"),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        fasta_path = f.name
    
    try:
        save_sequences_to_fasta(sequences, fasta_path)
        
        # Read back and verify
        with open(fasta_path, 'r') as f:
            content = f.read()
        
        assert ">seq1" in content
        assert ">seq2" in content
        assert "ATCGATCG" in content
        assert "GCTAGCTA" in content
    finally:
        Path(fasta_path).unlink()


def test_save_sequences_long_lines():
    """Test FASTA saves with proper line wrapping."""
    # Create a long sequence (> 80 chars)
    long_sequence = "A" * 200
    sequences = [("long_seq", long_sequence)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        fasta_path = f.name
    
    try:
        save_sequences_to_fasta(sequences, fasta_path)
        
        # Read back and check line lengths
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        
        # All sequence lines should be <= 80 characters
        for line in sequence_lines:
            assert len(line) <= 80
    finally:
        Path(fasta_path).unlink()


def test_nucleotide_composition():
    """Test nucleotide composition calculation."""
    sequences = ["AAAA"]  # 100% A
    stats = calculate_variant_statistics(sequences)
    
    composition = stats['nucleotide_composition']
    assert abs(composition['A'] - 100.0) < 0.01
    assert abs(composition['C']) < 0.01
    assert abs(composition['G']) < 0.01
    assert abs(composition['T']) < 0.01


def test_empty_sequences():
    """Test handling of empty sequence list."""
    stats = calculate_variant_statistics([])
    assert stats == {}


if __name__ == "__main__":
    test_compute_embedding_similarity_cosine()
    test_compute_embedding_similarity_euclidean()
    test_calculate_variant_statistics()
    test_calculate_variant_statistics_gc_content()
    test_create_synthetic_variants()
    test_save_sequences_to_fasta()
    test_save_sequences_long_lines()
    test_nucleotide_composition()
    test_empty_sequences()
    
    print("All utility tests passed!")
