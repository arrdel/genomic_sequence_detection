"""
Utility functions for genomic sequence detection.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def compute_embedding_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    metric: str = "cosine"
) -> float:
    """
    Compute similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: Similarity metric ('cosine' or 'euclidean')
        
    Returns:
        Similarity score
    """
    if metric == "cosine":
        return torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()
    elif metric == "euclidean":
        return -torch.dist(embedding1, embedding2, p=2).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: List[str],
    save_path: str = "embeddings_tsne.png"
):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Array of embeddings (n_samples, embedding_dim)
        labels: List of labels for each embedding
        save_path: Path to save the visualization
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label,
            alpha=0.6,
            edgecolors='w',
            s=50
        )
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Genomic Sequence Embeddings Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved embedding visualization to {save_path}")


def evaluate_variant_detection(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str] = None
) -> Dict:
    """
    Evaluate variant detection performance.
    
    Args:
        predictions: Predicted labels
        true_labels: Ground truth labels
        class_names: Names of variant classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Classification report
    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png"
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Names of classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Variant Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "training_history.png"
):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training time
    if 'train_time' in history:
        axes[1].plot(history['train_time'], marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('Training Time per Epoch')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history to {save_path}")


def calculate_variant_statistics(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate statistics for genomic sequences.
    
    Args:
        sequences: List of DNA/RNA sequences
        
    Returns:
        Dictionary with sequence statistics
    """
    if not sequences:
        return {}
    
    lengths = [len(seq) for seq in sequences]
    
    # Nucleotide composition
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
    total_nucleotides = 0
    
    for seq in sequences:
        for nucleotide in seq.upper():
            if nucleotide in nucleotide_counts:
                nucleotide_counts[nucleotide] += 1
            total_nucleotides += 1
    
    # Calculate percentages
    nucleotide_percentages = {
        nuc: (count / total_nucleotides * 100) if total_nucleotides > 0 else 0
        for nuc, count in nucleotide_counts.items()
    }
    
    # GC content
    gc_content = (nucleotide_counts['G'] + nucleotide_counts['C']) / total_nucleotides * 100 if total_nucleotides > 0 else 0
    
    return {
        'num_sequences': len(sequences),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'gc_content': gc_content,
        'nucleotide_composition': nucleotide_percentages
    }


def create_synthetic_variants(
    base_sequence: str,
    num_variants: int = 10,
    mutation_rate: float = 0.01
) -> List[str]:
    """
    Create synthetic variants from a base sequence.
    
    Args:
        base_sequence: Original genomic sequence
        num_variants: Number of variants to create
        mutation_rate: Probability of mutation per nucleotide
        
    Returns:
        List of variant sequences
    """
    nucleotides = ['A', 'C', 'G', 'T']
    variants = []
    
    for _ in range(num_variants):
        variant = []
        for nucleotide in base_sequence.upper():
            if nucleotide in nucleotides and np.random.random() < mutation_rate:
                # Mutate to a different nucleotide
                options = [n for n in nucleotides if n != nucleotide]
                variant.append(np.random.choice(options))
            else:
                variant.append(nucleotide)
        variants.append(''.join(variant))
    
    return variants


def save_sequences_to_fasta(
    sequences: List[Tuple[str, str]],
    output_path: str
):
    """
    Save sequences to a FASTA file.
    
    Args:
        sequences: List of (id, sequence) tuples
        output_path: Path to output FASTA file
    """
    with open(output_path, 'w') as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")
    print(f"Saved {len(sequences)} sequences to {output_path}")
