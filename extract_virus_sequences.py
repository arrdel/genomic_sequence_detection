#!/usr/bin/env python3
"""
Extract virus sequences from FASTQ file based on Kraken2 classification.

This script:
1. Finds all virus classifications in kraken_report.txt (column 6 contains "virus")
2. Extracts their taxonomy codes (column 5)
3. Maps these codes to kraken_classification.txt (column 3)
4. Extracts corresponding SRR IDs (column 2)
5. Retrieves sequences from cleaned_reads.fastq
6. Outputs a new file with sequences, codes, and virus names

Usage:
    python extract_virus_sequences.py
    python extract_virus_sequences.py --kraken-report path/to/kraken_report.txt \
                                       --kraken-classification path/to/kraken_classifications.txt \
                                       --fastq path/to/cleaned_reads.fastq \
                                       --output virus_sequences.fastq
"""

import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import re
from collections import defaultdict


def parse_kraken_report(kraken_report_file):
    """
    Parse kraken_report.txt to find all virus classifications.
    
    Returns:
        dict: {taxonomy_code: virus_name}
    """
    virus_taxonomy = {}
    
    print("Parsing Kraken report for virus classifications...")
    with open(kraken_report_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                # Column indices (0-based): 0=percentage, 1=fragments, 2=direct_fragments,
                # 3=rank_code, 4=taxon_id, 5=scientific_name
                taxon_id = parts[4].strip()
                scientific_name = parts[5].strip()
                
                # Check if "virus" appears in the name (case-insensitive)
                if 'virus' in scientific_name.lower():
                    virus_taxonomy[taxon_id] = scientific_name
    
    print(f"Found {len(virus_taxonomy)} virus classifications")
    return virus_taxonomy


def parse_kraken_classification(kraken_classification_file, virus_taxonomy_codes):
    """
    Parse kraken_classification.txt to map taxonomy codes to sequence IDs.
    
    Args:
        kraken_classification_file: Path to kraken classification file
        virus_taxonomy_codes: Set of taxonomy codes to look for
    
    Returns:
        dict: {sequence_id: (taxonomy_code, classification_status)}
    """
    sequence_to_taxon = {}
    
    print("Parsing Kraken classification file...")
    with open(kraken_classification_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # Column indices: 0=C/U (classified/unclassified), 1=sequence_id, 2=taxonomy_id
                classification_status = parts[0].strip()
                sequence_id = parts[1].strip()
                taxonomy_id = parts[2].strip()
                
                # Check if this sequence matches any virus taxonomy
                if taxonomy_id in virus_taxonomy_codes:
                    sequence_to_taxon[sequence_id] = (taxonomy_id, classification_status)
    
    print(f"Found {len(sequence_to_taxon)} sequences classified as viruses")
    return sequence_to_taxon


def extract_virus_sequences(fastq_file, sequence_to_taxon, virus_taxonomy, output_file):
    """
    Extract virus sequences from FASTQ file and write to output.
    
    Args:
        fastq_file: Path to input FASTQ file
        sequence_to_taxon: Dict mapping sequence IDs to taxonomy codes
        virus_taxonomy: Dict mapping taxonomy codes to virus names
        output_file: Path to output file
    """
    print("Extracting virus sequences from FASTQ file...")
    
    virus_records = []
    matched_count = 0
    total_count = 0
    
    # Read FASTQ file
    for record in SeqIO.parse(fastq_file, "fastq"):
        total_count += 1
        
        # Extract the base sequence ID (remove everything after first space or period)
        # Handle formats like "SRR35556007.1" or "SRR35556007.1 VH01473:..."
        seq_id = record.id.split()[0]  # Get first part before space
        
        # Check if this sequence is classified as a virus
        if seq_id in sequence_to_taxon:
            taxonomy_id, status = sequence_to_taxon[seq_id]
            virus_name = virus_taxonomy.get(taxonomy_id, "Unknown virus")
            
            # Create new record with enhanced description
            new_description = (
                f"{record.description} | "
                f"TaxonID:{taxonomy_id} | "
                f"Classification:{status} | "
                f"Virus:{virus_name}"
            )
            
            new_record = SeqRecord(
                record.seq,
                id=record.id,
                name=record.name,
                description=new_description,
                letter_annotations=record.letter_annotations
            )
            
            virus_records.append(new_record)
            matched_count += 1
            
            if matched_count <= 5:
                print(f"  Example {matched_count}: {seq_id} -> {virus_name}")
    
    # Write virus sequences to output file
    SeqIO.write(virus_records, output_file, "fastq")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"Total sequences processed: {total_count}")
    print(f"Virus sequences found: {matched_count}")
    print(f"Output file: {output_file}")
    
    # Print summary by virus type
    if matched_count > 0:
        virus_counts = defaultdict(int)
        for seq_id, (taxon_id, _) in sequence_to_taxon.items():
            virus_counts[virus_taxonomy.get(taxon_id, "Unknown")] += 1
        
        print(f"\nVirus distribution:")
        for virus_name, count in sorted(virus_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {virus_name}: {count} sequences")


def main():
    parser = argparse.ArgumentParser(
        description='Extract virus sequences based on Kraken2 classification.'
    )
    parser.add_argument(
        '--kraken-report',
        default='sequence_classification/kraken_report.txt',
        help='Path to kraken_report.txt file (default: sequence_classification/kraken_report.txt)'
    )
    parser.add_argument(
        '--kraken-classification',
        default='sequence_classification/kraken_classification.txt',
        help='Path to kraken_classification.txt file (default: sequence_classification/kraken_classification.txt)'
    )
    parser.add_argument(
        '--fastq',
        default='cleaned_reads.fastq',
        help='Path to cleaned_reads.fastq file (default: cleaned_reads.fastq)'
    )
    parser.add_argument(
        '--output',
        default='virus_sequences.fastq',
        help='Path to output FASTQ file (default: virus_sequences.fastq)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Parse kraken report to find virus taxonomy codes
    virus_taxonomy = parse_kraken_report(args.kraken_report)
    
    if not virus_taxonomy:
        print("Warning: No virus classifications found in kraken report!")
        return
    
    # Step 2: Parse kraken classification to map sequences to virus taxonomy
    virus_taxonomy_codes = set(virus_taxonomy.keys())
    sequence_to_taxon = parse_kraken_classification(args.kraken_classification, virus_taxonomy_codes)
    
    if not sequence_to_taxon:
        print("Warning: No sequences found with virus classifications!")
        return
    
    # Step 3: Extract virus sequences from FASTQ file
    extract_virus_sequences(args.fastq, sequence_to_taxon, virus_taxonomy, args.output)


if __name__ == "__main__":
    main()
