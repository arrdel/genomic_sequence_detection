"""
Example training script for contrastive genomic variant detection.
"""

import torch
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genomic_detection import ContrastiveGenomicModel, SequenceDataLoader, ContrastiveTrainer
from genomic_detection.models import NTXentLoss
from genomic_detection.utils import create_synthetic_variants, save_sequences_to_fasta


def main(args):
    """Main training function."""
    
    print("=" * 70)
    print("Contrastive Deep Learning for Variant Detection")
    print("=" * 70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create data loader
    print("\n1. Loading and preparing data...")
    data_loader = SequenceDataLoader(
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Load or create synthetic data
    if args.data_path:
        print(f"Loading sequences from {args.data_path}")
        sequences = data_loader.load_from_fasta(args.data_path)
    else:
        print("Creating synthetic data for demonstration...")
        # Create synthetic sequences
        base_sequence = "ATCG" * 250  # 1000 bp sequence
        variants = create_synthetic_variants(base_sequence, num_variants=100, mutation_rate=0.02)
        
        # Save synthetic data
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = output_dir / "synthetic_sequences.fasta"
        
        sequences_to_save = [(f"seq_{i}", seq) for i, seq in enumerate(variants)]
        save_sequences_to_fasta(sequences_to_save, str(synthetic_path))
        
        sequences = data_loader.load_from_fasta(str(synthetic_path))
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Split into train/val
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    
    # Create data loaders
    train_loader = data_loader.get_dataloader(train_sequences, shuffle=True)
    val_loader = data_loader.get_dataloader(val_sequences, shuffle=False)
    
    # Create model
    print("\n2. Initializing model...")
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128, 256),
        embedding_dim=args.embedding_dim,
        num_variant_classes=10,
        temperature=args.temperature
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    trainer = ContrastiveTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    contrastive_loss = NTXentLoss(temperature=args.temperature)
    
    # Train
    print("\n4. Starting training...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        contrastive_loss_fn=contrastive_loss,
        save_path=str(checkpoint_path)
    )
    
    # Plot training history
    from genomic_detection.utils import plot_training_history
    history_path = output_dir / "training_history.png"
    plot_training_history(trainer.history, str(history_path))
    
    print("\n" + "=" * 70)
    print(f"Training completed! Model saved to {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train contrastive model for genomic variant detection"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to FASTA file with genomic sequences"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    
    # Model arguments
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Dimension of sequence embeddings"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization"
    )
    
    args = parser.parse_args()
    main(args)
