"""
Training utilities for contrastive genomic models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import time
from pathlib import Path


class ContrastiveTrainer:
    """
    Trainer for contrastive learning on genomic sequences.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The contrastive genomic model
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_time': []
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        contrastive_loss_fn: Callable,
        variant_loss_fn: Optional[Callable] = None,
        variant_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            contrastive_loss_fn: Loss function for contrastive learning
            variant_loss_fn: Optional loss function for variant detection
            variant_weight: Weight for variant detection loss
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        contrastive_loss_total = 0.0
        variant_loss_total = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (sequences, metadata) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            variant_predictions, embeddings = self.model(sequences, return_embeddings=True)
            
            loss = 0.0
            
            # Contrastive loss (using pairs within batch)
            if embeddings is not None and len(embeddings) > 1:
                # Create augmented pairs
                aug_sequences = self._augment_batch(sequences)
                aug_predictions, aug_embeddings = self.model(aug_sequences, return_embeddings=True)
                
                contrastive_loss = contrastive_loss_fn(embeddings, aug_embeddings)
                loss += contrastive_loss
                contrastive_loss_total += contrastive_loss.item()
            
            # Variant detection loss (if labels available)
            if variant_loss_fn is not None and 'variant_labels' in metadata:
                variant_labels = metadata['variant_labels'].to(self.device)
                variant_loss = variant_loss_fn(variant_predictions, variant_labels)
                loss += variant_weight * variant_loss
                variant_loss_total += variant_loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'contrastive_loss': contrastive_loss_total / num_batches if num_batches > 0 else 0.0,
            'variant_loss': variant_loss_total / num_batches if num_batches > 0 else 0.0,
            'epoch_time': epoch_time
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        contrastive_loss_fn: Callable,
        variant_loss_fn: Optional[Callable] = None,
        variant_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            contrastive_loss_fn: Loss function for contrastive learning
            variant_loss_fn: Optional loss function for variant detection
            variant_weight: Weight for variant detection loss
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        contrastive_loss_total = 0.0
        variant_loss_total = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, metadata in val_loader:
                sequences = sequences.to(self.device)
                
                # Forward pass
                variant_predictions, embeddings = self.model(sequences, return_embeddings=True)
                
                loss = 0.0
                
                # Contrastive loss
                if embeddings is not None and len(embeddings) > 1:
                    aug_sequences = self._augment_batch(sequences)
                    aug_predictions, aug_embeddings = self.model(aug_sequences, return_embeddings=True)
                    
                    contrastive_loss = contrastive_loss_fn(embeddings, aug_embeddings)
                    loss += contrastive_loss
                    contrastive_loss_total += contrastive_loss.item()
                
                # Variant detection loss
                if variant_loss_fn is not None and 'variant_labels' in metadata:
                    variant_labels = metadata['variant_labels'].to(self.device)
                    variant_loss = variant_loss_fn(variant_predictions, variant_labels)
                    loss += variant_weight * variant_loss
                    variant_loss_total += variant_loss.item()
                
                total_loss += loss.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'contrastive_loss': contrastive_loss_total / num_batches if num_batches > 0 else 0.0,
            'variant_loss': variant_loss_total / num_batches if num_batches > 0 else 0.0
        }
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        contrastive_loss_fn: Optional[Callable] = None,
        variant_loss_fn: Optional[Callable] = None,
        save_path: Optional[str] = None
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            contrastive_loss_fn: Loss function for contrastive learning
            variant_loss_fn: Optional loss function for variant detection
            save_path: Path to save best model
        """
        if contrastive_loss_fn is None:
            from ..models.contrastive_model import NTXentLoss
            contrastive_loss_fn = NTXentLoss()
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(
                train_loader,
                contrastive_loss_fn,
                variant_loss_fn
            )
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Contrastive: {train_metrics['contrastive_loss']:.4f}, "
                  f"Time: {train_metrics['epoch_time']:.2f}s")
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_time'].append(train_metrics['epoch_time'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(
                    val_loader,
                    contrastive_loss_fn,
                    variant_loss_fn
                )
                
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Contrastive: {val_metrics['contrastive_loss']:.4f}")
                
                self.history['val_loss'].append(val_metrics['loss'])
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < best_val_loss and save_path is not None:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_path)
                    print(f"Saved best model to {save_path}")
        
        print("\nTraining completed!")
    
    def _augment_batch(self, sequences: torch.Tensor, noise_level: float = 0.05) -> torch.Tensor:
        """
        Augment a batch of sequences with noise.
        
        Args:
            sequences: Input sequences (batch, seq_len, channels)
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Augmented sequences
        """
        noise = torch.randn_like(sequences) * noise_level
        return sequences + noise
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {path}")
