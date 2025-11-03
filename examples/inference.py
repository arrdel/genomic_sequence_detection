"""
Example inference script for variant detection.
"""

import torch
import argparse
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genomic_detection import ContrastiveGenomicModel, SequenceDataLoader
from genomic_detection.utils import compute_embedding_similarity, visualize_embeddings


def main(args):
    """Main inference function."""
    
    print("=" * 70)
    print("Genomic Variant Detection - Inference")
    print("=" * 70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\n1. Loading model from {args.model_path}...")
    model = ContrastiveGenomicModel(
        input_channels=5,
        hidden_dims=(64, 128, 256),
        embedding_dim=args.embedding_dim,
        num_variant_classes=10
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load sequences
    print(f"\n2. Loading sequences from {args.input_path}...")
    data_loader = SequenceDataLoader(
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    sequences = data_loader.load_from_fasta(args.input_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Get embeddings and predictions
    print("\n3. Computing embeddings and predictions...")
    test_loader = data_loader.get_dataloader(sequences, shuffle=False)
    
    all_embeddings = []
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for sequences_batch, metadata in test_loader:
            sequences_batch = sequences_batch.to(device)
            
            # Get embeddings
            embeddings = model.get_embeddings(sequences_batch)
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Get variant predictions
            predictions = model.detect_variants(sequences_batch)
            predicted_classes = torch.argmax(predictions, dim=1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            
            # Store IDs
            all_ids.extend(metadata['sample_id'])
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Display results
    print("\n4. Results:")
    print("-" * 70)
    for i, (seq_id, pred) in enumerate(zip(all_ids, all_predictions)):
        print(f"Sequence {seq_id}: Variant Class {pred}")
    
    # Compute pairwise similarities if requested
    if args.compute_similarity and len(all_embeddings) > 1:
        print("\n5. Computing pairwise similarities...")
        print("-" * 70)
        for i in range(min(5, len(all_embeddings))):
            for j in range(i + 1, min(5, len(all_embeddings))):
                emb1 = torch.tensor(all_embeddings[i])
                emb2 = torch.tensor(all_embeddings[j])
                similarity = compute_embedding_similarity(emb1, emb2, metric="cosine")
                print(f"Similarity between {all_ids[i]} and {all_ids[j]}: {similarity:.4f}")
    
    # Visualize embeddings if requested
    if args.visualize:
        print("\n6. Visualizing embeddings...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create labels from predictions
        labels = [f"Class_{pred}" for pred in all_predictions]
        
        viz_path = output_dir / "embeddings_visualization.png"
        visualize_embeddings(all_embeddings, labels, str(viz_path))
    
    # Save embeddings if requested
    if args.save_embeddings:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_path = output_dir / "embeddings.npz"
        np.savez(
            embeddings_path,
            embeddings=all_embeddings,
            predictions=all_predictions,
            ids=all_ids
        )
        print(f"\nSaved embeddings to {embeddings_path}")
    
    print("\n" + "=" * 70)
    print("Inference completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with trained contrastive model"
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input FASTA file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save outputs"
    )
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--compute_similarity",
        action="store_true",
        help="Compute pairwise similarities between sequences"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize embeddings using t-SNE"
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save embeddings to file"
    )
    
    args = parser.parse_args()
    main(args)
