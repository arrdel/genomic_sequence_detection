#!/usr/bin/env python3
"""
Classify wastewater genomic sequences using Kraken 2.
Author: Richmond E. K. Azumah
Date: 2025-11-08

python sequence_classifier.py \
  --input cleaned_reads.fastq \
  --db k2_minusb \
  --outdir sequence_classification \
  --threads 16
"""

import os
import argparse 
import subprocess

def run_kraken(input_file, db_path, output_dir, threads=8):
    """
    Runs Kraken2 on the given input file using the specified database.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_report = os.path.join(output_dir, "kraken_report.txt")
    output_labels = os.path.join(output_dir, "kraken_classifications.txt")

    cmd = [
        "kraken2",
        "--db", db_path,
        "--threads", str(threads),
        "--report", output_report,
        "--output", output_labels,
        input_file
    ]

    print(f"Running Kraken2 on {input_file} ...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Classification complete!")
        print(f"- Report saved to: {output_report}")
        print(f"- Classifications saved to: {output_labels}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Kraken2 failed: {e}")
    except FileNotFoundError:
        print("\n❌ Kraken2 not found. Please ensure Kraken2 is installed and in PATH.")


def main():
    parser = argparse.ArgumentParser(description="Run Kraken2 classification for wastewater genomic sequences.")
    parser.add_argument("--input", required=True, help="Path to FASTQ/FASTA file.")
    parser.add_argument("--db", required=True, help="Path to Kraken2 database.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default=8).")

    args = parser.parse_args()
    run_kraken(args.input, args.db, args.outdir, args.threads)


if __name__ == "__main__":
    main()