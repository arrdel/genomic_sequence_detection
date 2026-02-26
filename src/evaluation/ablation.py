#!/usr/bin/env python3
"""
Ablation Study Framework

Systematic ablation studies to justify architectural and hyperparameter choices:

1. Codebook size: K ∈ {64, 128, 256, 512, 1024, 2048}
2. Code dimension: D ∈ {16, 32, 64, 128, 256}
3. K-mer size: k ∈ {4, 5, 6, 7, 8}
4. VQ method: EMA vs. straight-through vs. Gumbel-Softmax
5. Loss components: reconstruction only, +commitment, +entropy
6. Contrastive: temperature τ, augmentation strategy, projection dim
7. Masking: mask probability ∈ {0.1, 0.15, 0.2, 0.25, 0.3}
"""

import os
import sys
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    
    # Model
    num_codes: int = 512
    code_dim: int = 64
    embed_dim: int = 128
    hidden_dim: int = 256
    commitment_cost: float = 0.1
    entropy_weight: float = 0.003
    
    # Data
    k_mer: int = 6
    max_seq_length: int = 150
    
    # Training
    epochs: int = 30  # Shorter for ablations
    batch_size: int = 32
    learning_rate: float = 2e-4
    seed: int = 42
    
    # Contrastive (if applicable)
    contrastive_epochs: int = 10
    temperature: float = 0.5
    mask_prob: float = 0.15
    drop_prob: float = 0.10
    proj_dim: int = 64
    
    # Masking (if applicable)
    mask_training_prob: float = 0.2
    
    def to_dict(self):
        return asdict(self)


def generate_codebook_size_ablation() -> List[AblationConfig]:
    """Ablation over codebook size K."""
    configs = []
    for k in [64, 128, 256, 512, 1024, 2048]:
        configs.append(AblationConfig(
            name=f"codebook_K{k}",
            description=f"Codebook size K={k}",
            num_codes=k,
        ))
    return configs


def generate_code_dim_ablation() -> List[AblationConfig]:
    """Ablation over code dimension D."""
    configs = []
    for d in [16, 32, 64, 128, 256]:
        configs.append(AblationConfig(
            name=f"codedim_D{d}",
            description=f"Code dimension D={d}",
            code_dim=d,
        ))
    return configs


def generate_kmer_size_ablation() -> List[AblationConfig]:
    """Ablation over k-mer size."""
    configs = []
    for k in [4, 5, 6, 7, 8]:
        vocab_size = 4 ** k + 3  # +3 for PAD, UNK, MASK
        configs.append(AblationConfig(
            name=f"kmer_k{k}",
            description=f"K-mer size k={k}, vocab={vocab_size}",
            k_mer=k,
        ))
    return configs


def generate_loss_ablation() -> List[AblationConfig]:
    """Ablation over loss components."""
    configs = [
        AblationConfig(
            name="loss_recon_only",
            description="Reconstruction loss only (no commitment, no entropy)",
            commitment_cost=0.0,
            entropy_weight=0.0,
        ),
        AblationConfig(
            name="loss_recon_commit",
            description="Reconstruction + commitment loss (no entropy)",
            commitment_cost=0.1,
            entropy_weight=0.0,
        ),
        AblationConfig(
            name="loss_full",
            description="Full loss (reconstruction + commitment + entropy)",
            commitment_cost=0.1,
            entropy_weight=0.003,
        ),
        AblationConfig(
            name="loss_high_commit",
            description="High commitment weight β=0.25",
            commitment_cost=0.25,
            entropy_weight=0.003,
        ),
        AblationConfig(
            name="loss_high_entropy",
            description="High entropy weight λ=0.01",
            commitment_cost=0.1,
            entropy_weight=0.01,
        ),
    ]
    return configs


def generate_contrastive_ablation() -> List[AblationConfig]:
    """Ablation over contrastive learning hyperparameters."""
    configs = []
    
    # Temperature
    for tau in [0.1, 0.3, 0.5, 0.7, 1.0]:
        configs.append(AblationConfig(
            name=f"contrast_tau{tau}",
            description=f"Contrastive temperature τ={tau}",
            temperature=tau,
        ))
    
    # Projection dimension
    for dim in [32, 64, 128, 256]:
        configs.append(AblationConfig(
            name=f"contrast_proj{dim}",
            description=f"Contrastive projection dim={dim}",
            proj_dim=dim,
        ))
    
    # Augmentation strength
    for mask_p, drop_p in [(0.05, 0.05), (0.10, 0.10), (0.15, 0.10), (0.20, 0.15), (0.30, 0.20)]:
        configs.append(AblationConfig(
            name=f"contrast_aug_m{int(mask_p*100)}_d{int(drop_p*100)}",
            description=f"Augmentation mask={mask_p}, drop={drop_p}",
            mask_prob=mask_p,
            drop_prob=drop_p,
        ))
    
    return configs


def generate_masking_ablation() -> List[AblationConfig]:
    """Ablation over masking probability."""
    configs = []
    for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        configs.append(AblationConfig(
            name=f"mask_p{int(p*100)}",
            description=f"Masking probability p={p}",
            mask_training_prob=p,
        ))
    return configs


def generate_multi_seed_configs(
    base_config: AblationConfig,
    seeds: List[int] = [42, 123, 456, 789, 1024],
) -> List[AblationConfig]:
    """Generate configs with multiple seeds for statistical significance."""
    configs = []
    for seed in seeds:
        cfg = AblationConfig(**base_config.to_dict())
        cfg.name = f"{base_config.name}_seed{seed}"
        cfg.seed = seed
        configs.append(cfg)
    return configs


def save_ablation_configs(configs: List[AblationConfig], output_dir: str):
    """Save all ablation configurations to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_configs = []
    for cfg in configs:
        cfg_dict = cfg.to_dict()
        all_configs.append(cfg_dict)
        
        # Save individual config
        cfg_path = os.path.join(output_dir, f"{cfg.name}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_dict, f, indent=2)
    
    # Save master config list
    master_path = os.path.join(output_dir, "all_ablations.json")
    with open(master_path, "w") as f:
        json.dump(all_configs, f, indent=2)
    
    print(f"Saved {len(configs)} ablation configs to {output_dir}")
    return master_path


def generate_all_ablations(output_dir: str = "configs/ablations") -> str:
    """Generate all ablation study configurations."""
    all_configs = []
    
    print("Generating ablation configurations...")
    
    studies = {
        "codebook_size": generate_codebook_size_ablation(),
        "code_dimension": generate_code_dim_ablation(),
        "kmer_size": generate_kmer_size_ablation(),
        "loss_components": generate_loss_ablation(),
        "contrastive": generate_contrastive_ablation(),
        "masking": generate_masking_ablation(),
    }
    
    for study_name, configs in studies.items():
        print(f"  {study_name}: {len(configs)} configs")
        all_configs.extend(configs)
    
    print(f"\nTotal ablation configs: {len(all_configs)}")
    
    return save_ablation_configs(all_configs, output_dir)


if __name__ == "__main__":
    generate_all_ablations()
