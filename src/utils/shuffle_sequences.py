#!/usr/bin/env python3
"""
Shuffle genome sequences in a FASTQ file while preserving headers and quality scores.

Usage:
    python shuffle_sequences.py input.fastq output.fastq
    python shuffle_sequences.py --seed 42 input.fastq output.fastq
"""

import argparse
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def shuffle_sequences(input_fastq, output_fastq, seed=None):
    """
    Read a FASTQ file, shuffle only the sequences, and write to output file.
    
    Args:
        input_fastq: Path to input FASTQ file
        output_fastq: Path to output FASTQ file
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    # Read all records from input file
    records = list(SeqIO.parse(input_fastq, "fastq"))
    
    # Create list of indices to shuffle
    indices = list(range(len(records)))
    random.shuffle(indices)
    
    # Create new records with shuffled sequences AND their corresponding quality scores
    shuffled_records = []
    for i, record in enumerate(records):
        # Get the shuffled sequence and its original quality scores from the shuffled index
        shuffled_idx = indices[i]
        shuffled_seq = str(records[shuffled_idx].seq)
        shuffled_qual = records[shuffled_idx].letter_annotations
        
        # Create new record with original header but shuffled sequence and matching quality
        new_record = SeqRecord(
            Seq(shuffled_seq),
            id=record.id,
            name=record.name,
            description=record.description,
            letter_annotations=shuffled_qual  # Quality scores match the shuffled sequence
        )
        shuffled_records.append(new_record)
    
    # Write shuffled records to output file
    SeqIO.write(shuffled_records, output_fastq, "fastq")
    
    print(f"✓ Shuffled {len(records)} sequences")
    print(f"✓ Input:  {input_fastq}")
    print(f"✓ Output: {output_fastq}")
    if seed is not None:
        print(f"✓ Random seed: {seed}")


def main():
    parser = argparse.ArgumentParser(
        description='Shuffle genome sequences in a FASTQ file while preserving headers and quality scores.'
    )
    parser.add_argument('input', help='Input FASTQ file')
    parser.add_argument('output', help='Output FASTQ file with shuffled sequences')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (optional)')
    
    args = parser.parse_args()
    
    shuffle_sequences(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
