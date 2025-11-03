#!/usr/bin/env python3
"""
Data Preprocessing Script for Viral Genome Sequences

This script handles preprocessing of FASTQ files using Trimmomatic:
- Quality trimming
- Adapter removal
- FastQC quality control

Usage:
    python preprocess.py --input-fastq /path/to/raw.fastq --output-fastq /path/to/cleaned.fastq
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess viral genome sequences')
    
    # Input/Output arguments
    parser.add_argument('--input-fastq', type=str, required=True,
                        help='Path to input FASTQ file')
    parser.add_argument('--output-fastq', type=str, required=True,
                        help='Path to output cleaned FASTQ file')
    parser.add_argument('--output-dir', type=str, default='./preprocessing_output',
                        help='Directory for intermediate outputs and logs')
    
    # Trimmomatic arguments
    parser.add_argument('--trimmomatic-jar', type=str, 
                        default='./Trimmomatic/trimmomatic-0.39.jar',
                        help='Path to Trimmomatic JAR file')
    parser.add_argument('--adapter-file', type=str,
                        default='./Trimmomatic/adapters/TruSeq3-SE.fa',
                        help='Path to adapter sequences file')
    parser.add_argument('--leading', type=int, default=3,
                        help='Cut bases off the start if below threshold quality')
    parser.add_argument('--trailing', type=int, default=3,
                        help='Cut bases off the end if below threshold quality')
    parser.add_argument('--slidingwindow', type=str, default='4:15',
                        help='Sliding window parameters (window_size:quality)')
    parser.add_argument('--minlen', type=int, default=36,
                        help='Drop reads below this length')
    
    # FastQC arguments
    parser.add_argument('--run-fastqc', action='store_true',
                        help='Run FastQC on input and output files')
    parser.add_argument('--fastqc-before-dir', type=str, default='./fastqc_before',
                        help='Directory for FastQC results before preprocessing')
    parser.add_argument('--fastqc-after-dir', type=str, default='./fastqc_after',
                        help='Directory for FastQC results after preprocessing')
    
    # Other arguments
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads to use')
    parser.add_argument('--skip-trimmomatic', action='store_true',
                        help='Skip Trimmomatic (only run FastQC)')
    
    args = parser.parse_args()
    return args


def run_fastqc(input_file, output_dir, threads=4):
    """Run FastQC on a FASTQ file"""
    print(f"\nRunning FastQC on {input_file}...")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'fastqc',
        input_file,
        '-o', output_dir,
        '-t', str(threads)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"FastQC completed successfully")
        print(f"Output directory: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FastQC failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("FastQC not found. Please install FastQC or skip this step.")
        return False


def run_trimmomatic(input_file, output_file, trimmomatic_jar, adapter_file, 
                   leading, trailing, slidingwindow, minlen, threads, log_file):
    """Run Trimmomatic for quality trimming"""
    print(f"\nRunning Trimmomatic...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Trimmomatic command for single-end reads
    cmd = [
        'java', '-jar', trimmomatic_jar,
        'SE',  # Single-end mode
        '-threads', str(threads),
        input_file,
        output_file,
        f'ILLUMINACLIP:{adapter_file}:2:30:10',
        f'LEADING:{leading}',
        f'TRAILING:{trailing}',
        f'SLIDINGWINDOW:{slidingwindow}',
        f'MINLEN:{minlen}'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)
        
        print(f"Trimmomatic completed successfully")
        print(f"Log file: {log_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Trimmomatic failed with error code {e.returncode}")
        print(f"Check log file for details: {log_file}")
        return False
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure Java and Trimmomatic are properly installed.")
        return False


def get_fastq_stats(fastq_file):
    """Get basic statistics from a FASTQ file"""
    try:
        from Bio import SeqIO
        
        print(f"\nComputing statistics for {fastq_file}...")
        
        num_sequences = 0
        total_length = 0
        lengths = []
        
        for record in SeqIO.parse(fastq_file, "fastq"):
            num_sequences += 1
            seq_len = len(record.seq)
            total_length += seq_len
            lengths.append(seq_len)
        
        if num_sequences > 0:
            avg_length = total_length / num_sequences
            min_length = min(lengths)
            max_length = max(lengths)
            
            print(f"  Number of sequences: {num_sequences}")
            print(f"  Average length: {avg_length:.2f}")
            print(f"  Min length: {min_length}")
            print(f"  Max length: {max_length}")
            
            return {
                'num_sequences': num_sequences,
                'avg_length': avg_length,
                'min_length': min_length,
                'max_length': max_length
            }
        else:
            print("  No sequences found in file")
            return None
            
    except ImportError:
        print("  Biopython not installed. Skipping statistics.")
        return None
    except Exception as e:
        print(f"  Error computing statistics: {e}")
        return None


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Data Preprocessing Pipeline")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(args.input_fastq):
        print(f"Error: Input file not found: {args.input_fastq}")
        sys.exit(1)
    
    # Get input statistics
    print("\n" + "="*60)
    print("Input File Statistics")
    print("="*60)
    input_stats = get_fastq_stats(args.input_fastq)
    
    # Run FastQC on input (before preprocessing)
    if args.run_fastqc:
        print("\n" + "="*60)
        print("Running FastQC on Input File")
        print("="*60)
        run_fastqc(args.input_fastq, args.fastqc_before_dir, args.threads)
    
    # Run Trimmomatic
    if not args.skip_trimmomatic:
        print("\n" + "="*60)
        print("Running Trimmomatic")
        print("="*60)
        
        log_file = os.path.join(args.output_dir, 
                                f'trimmomatic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        success = run_trimmomatic(
            args.input_fastq,
            args.output_fastq,
            args.trimmomatic_jar,
            args.adapter_file,
            args.leading,
            args.trailing,
            args.slidingwindow,
            args.minlen,
            args.threads,
            log_file
        )
        
        if not success:
            print("\nPreprocessing failed!")
            sys.exit(1)
    else:
        print("\nSkipping Trimmomatic (--skip-trimmomatic flag set)")
    
    # Get output statistics
    if not args.skip_trimmomatic and os.path.exists(args.output_fastq):
        print("\n" + "="*60)
        print("Output File Statistics")
        print("="*60)
        output_stats = get_fastq_stats(args.output_fastq)
        
        # Run FastQC on output (after preprocessing)
        if args.run_fastqc:
            print("\n" + "="*60)
            print("Running FastQC on Output File")
            print("="*60)
            run_fastqc(args.output_fastq, args.fastqc_after_dir, args.threads)
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    if not args.skip_trimmomatic:
        print(f"Cleaned sequences saved to: {args.output_fastq}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
