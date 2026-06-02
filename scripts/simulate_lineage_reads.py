#!/usr/bin/env python3
"""
Simulate ground-truth labeled short reads from per-lineage SARS-CoV-2 reference
genomes.

This produces a synthetic dataset that mirrors the format of our training data
(150 bp single-end reads, FASTQ) but with KNOWN lineage labels for every read.
It addresses three reviewer concerns from the MLHC 2026 submission:

  * GHgX  - "evaluation uses k-means pseudo-labels (not ground-truth variants)"
  * uRw2  - "evaluate the embedding for the omicron variant vs. the wildtype"
  * wkYq  - "expect a synthetic data section where ground-truth is known
            ... enable evaluation of FPR/TPR detection"

Outputs (under --out-dir, default data/synthetic_lineages/):
  - <lineage>_reads.fastq        per-lineage reads
  - all_lineages.fastq           shuffled union of all lineages
  - labels.tsv                   read_id<TAB>lineage  (parallel to all_lineages)
  - metadata.json                config + counts for reproducibility

Read IDs are of the form `{lineage}_read_{i}` so downstream code can recover
the ground-truth label from the FASTQ header alone if needed.

Usage:
    python scripts/simulate_lineage_reads.py \\
        --reads-per-lineage 10000 \\
        --read-length 150 \\
        --error-rate 0.005 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Default lineage -> reference genome mapping for this repo.
# Filenames with spaces are intentional (matches what is in data/).
DEFAULT_LINEAGES: Dict[str, str] = {
    "wuhan":   "data/wuhan wildtype.fna",
    "delta":   "data/delta.fna",
    "omicron": "data/omicron.fna",
}

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_NUCLEOTIDES = ("A", "C", "G", "T")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def add_substitution_errors(seq: str, error_rate: float, rng: random.Random) -> str:
    """Apply per-base substitution errors at the given rate (Illumina-like)."""
    if error_rate <= 0:
        return seq
    bases = list(seq)
    for i, b in enumerate(bases):
        if b not in _NUCLEOTIDES:
            continue
        if rng.random() < error_rate:
            choices = [n for n in _NUCLEOTIDES if n != b]
            bases[i] = rng.choice(choices)
    return "".join(bases)


def load_genome(fasta_path: str) -> Tuple[str, str]:
    """Return (record_id, uppercased nucleotide sequence)."""
    record_id = None
    seq_parts: List[str] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    break
                record_id = line[1:].split()[0]
                continue
            if record_id is not None:
                seq_parts.append(line)

    if record_id is None or not seq_parts:
        raise ValueError(f"No FASTA record found in {fasta_path}")
    return record_id, "".join(seq_parts).upper()


def simulate_reads(
    genome: str,
    lineage_name: str,
    n_reads: int,
    read_length: int,
    error_rate: float,
    rng: random.Random,
    max_n_ratio: float = 0.07,
) -> List[Tuple[str, str]]:
    """
    Uniform-random sliding-window simulator with strand mixing and optional
    substitution noise. Returns a list of (read_id, sequence).
    """
    genome_len = len(genome)
    if genome_len < read_length:
        raise ValueError(
            f"Genome for {lineage_name} is {genome_len}bp, shorter than "
            f"read length {read_length}bp."
        )

    reads: List[Tuple[str, str]] = []
    max_n = int(read_length * max_n_ratio)
    attempts = 0
    max_attempts = n_reads * 10  # safety net against pathological genomes

    while len(reads) < n_reads and attempts < max_attempts:
        attempts += 1
        start = rng.randint(0, genome_len - read_length)
        read = genome[start:start + read_length]

        if read.count("N") > max_n:
            continue

        if rng.random() > 0.5:
            read = reverse_complement(read)

        read = add_substitution_errors(read, error_rate, rng)

        reads.append((f"{lineage_name}_read_{len(reads)}", read))

    if len(reads) < n_reads:
        print(
            f"  warn: only simulated {len(reads)}/{n_reads} reads for "
            f"{lineage_name} (too many ambiguous regions?)",
            file=sys.stderr,
        )
    return reads


def write_fastq(path: Path, reads: List[Tuple[str, str]], quality_char: str = "I") -> None:
    """Write reads as single-end FASTQ with a constant quality string."""
    with open(path, "w") as fh:
        for rid, seq in reads:
            qual = quality_char * len(seq)
            fh.write(f"@{rid}\n{seq}\n+\n{qual}\n")


def write_labels(path: Path, reads: List[Tuple[str, str]], labels: List[str]) -> None:
    with open(path, "w") as fh:
        fh.write("read_id\tlineage\n")
        for (rid, _), label in zip(reads, labels):
            fh.write(f"{rid}\t{label}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lineage", action="append", metavar="NAME=PATH",
        help="Override defaults. Repeatable. Example: --lineage wuhan=data/wuhan.fna",
    )
    p.add_argument("--reads-per-lineage", type=int, default=10_000)
    p.add_argument("--read-length", type=int, default=150,
                   help="Must match the training data (default 150).")
    p.add_argument("--error-rate", type=float, default=0.005,
                   help="Per-base substitution error rate (Illumina-like). "
                        "Set 0 to disable.")
    p.add_argument("--max-n-ratio", type=float, default=0.07,
                   help="Reject reads with more than this fraction of Ns.")
    p.add_argument("--out-dir", type=str,
                   default="data/synthetic_lineages")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-combined", action="store_true",
                   help="Skip writing all_lineages.fastq + labels.tsv.")
    return p.parse_args()


def resolve_lineages(overrides: List[str] | None) -> Dict[str, str]:
    if not overrides:
        return dict(DEFAULT_LINEAGES)
    mapping: Dict[str, str] = {}
    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--lineage expects NAME=PATH, got: {spec!r}")
        name, path = spec.split("=", 1)
        mapping[name.strip()] = path.strip()
    return mapping


def main() -> None:
    args = parse_args()
    lineages = resolve_lineages(args.lineage)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"Simulating {args.reads_per_lineage} reads/lineage "
          f"@ {args.read_length}bp, err={args.error_rate}, seed={args.seed}")
    print(f"Output dir: {out_dir.resolve()}\n")

    per_lineage_counts: Dict[str, int] = {}
    genome_ids: Dict[str, str] = {}
    combined: List[Tuple[str, str]] = []
    combined_labels: List[str] = []

    for lineage, fasta_path in lineages.items():
        if not os.path.exists(fasta_path):
            raise SystemExit(f"Missing reference for {lineage}: {fasta_path}")
        record_id, genome = load_genome(fasta_path)
        genome_ids[lineage] = record_id
        print(f"  {lineage:<10} {record_id}  ({len(genome)} bp)")

        reads = simulate_reads(
            genome=genome,
            lineage_name=lineage,
            n_reads=args.reads_per_lineage,
            read_length=args.read_length,
            error_rate=args.error_rate,
            rng=rng,
            max_n_ratio=args.max_n_ratio,
        )

        per_path = out_dir / f"{lineage}_reads.fastq"
        write_fastq(per_path, reads)
        per_lineage_counts[lineage] = len(reads)
        combined.extend(reads)
        combined_labels.extend([lineage] * len(reads))

    print()
    for lineage, n in per_lineage_counts.items():
        print(f"  wrote {n:>7d} reads -> {out_dir / f'{lineage}_reads.fastq'}")

    if not args.no_combined:
        order = list(range(len(combined)))
        rng.shuffle(order)
        shuffled = [combined[i] for i in order]
        shuffled_labels = [combined_labels[i] for i in order]

        all_path = out_dir / "all_lineages.fastq"
        labels_path = out_dir / "labels.tsv"
        write_fastq(all_path, shuffled)
        write_labels(labels_path, shuffled, shuffled_labels)
        print(f"\n  wrote {len(shuffled):>7d} reads -> {all_path} (shuffled)")
        print(f"  wrote labels -> {labels_path}")

    metadata = {
        "reads_per_lineage": args.reads_per_lineage,
        "read_length": args.read_length,
        "error_rate": args.error_rate,
        "max_n_ratio": args.max_n_ratio,
        "seed": args.seed,
        "lineages": lineages,
        "genome_ids": genome_ids,
        "counts": per_lineage_counts,
        "outputs": {
            "combined_fastq": None if args.no_combined else str(out_dir / "all_lineages.fastq"),
            "labels": None if args.no_combined else str(out_dir / "labels.tsv"),
        },
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  wrote metadata -> {meta_path}")
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()