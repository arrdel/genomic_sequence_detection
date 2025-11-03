"""
Configuration file for genomic detection models.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for the contrastive genomic model."""
    
    # Model architecture
    input_channels: int = 5
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    embedding_dim: int = 512
    num_variant_classes: int = 10
    kernel_size: int = 7
    dropout: float = 0.1
    
    # Contrastive learning
    temperature: float = 0.07
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 20
    
    # Data
    max_length: int = 1000
    num_workers: int = 4
    augmentation_prob: float = 0.1
    
    # Paths
    data_path: str = ""
    output_dir: str = "outputs"
    checkpoint_path: str = "outputs/best_model.pth"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.temperature > 0, "temperature must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert 0 <= self.augmentation_prob <= 1, "augmentation_prob must be in [0, 1]"


@dataclass
class WastewaterConfig(ModelConfig):
    """Specialized configuration for wastewater genomic sequencing."""
    
    # Wastewater-specific parameters
    min_sequence_quality: float = 20.0
    min_coverage: int = 10
    variant_frequency_threshold: float = 0.01
    
    # Extended model capacity for complex wastewater samples
    hidden_dims: Tuple[int, ...] = (128, 256, 512)
    embedding_dim: int = 1024
    
    # Adjusted for noisy wastewater data
    augmentation_prob: float = 0.15
    temperature: float = 0.05
    
    def __post_init__(self):
        """Validate wastewater-specific configuration."""
        super().__post_init__()
        assert self.min_sequence_quality > 0, "min_sequence_quality must be positive"
        assert self.min_coverage > 0, "min_coverage must be positive"
        assert 0 <= self.variant_frequency_threshold <= 1, "variant_frequency_threshold must be in [0, 1]"


# Default configurations
DEFAULT_CONFIG = ModelConfig()
WASTEWATER_CONFIG = WastewaterConfig()


def get_config(config_type: str = "default") -> ModelConfig:
    """
    Get a configuration by type.
    
    Args:
        config_type: Type of configuration ('default' or 'wastewater')
        
    Returns:
        Configuration object
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "wastewater": WASTEWATER_CONFIG,
    }
    
    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(configs.keys())}")
    
    return configs[config_type]
