"""
Training module for LSTM voltage forecasting system.

This module implements comprehensive training logic with early stopping,
learning rate scheduling, gradient clipping, and comprehensive monitoring.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .config import Config
from .lstm_forecaster import LSTMForecaster


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        logger.info(f"Early stopping initialized: patience={patience}, min_delta={min_delta}")
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            
        return self.early_stop


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.losses: List[float] = []
        self.batch_times: List[float] = []
        self.learning_rates: List[float] = []
        
    def update(self, loss: float, batch_time: float = 0.0, lr: float = 0.0) -> None:
        """Update metrics with new values."""
        self.losses.append(loss)
        if batch_time > 0:
            self.batch_times.append(batch_time)
        if lr > 0:
            self.learning_rates.append(lr)
    
    def get_average_loss(self) -> float:
        """Get average loss."""
        return np.mean(self.losses) if self.losses else 0.0
    
    def get_average_batch_time(self) -> float:
        """Get average batch time."""
        return np.mean(self.batch_times) if self.batch_times else 0.0
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rates[-1] if self.learning_rates else 0.0


class LSTMTrainer:
    """Main trainer class for LSTM voltage forecasting."""
    
    def __init__(
        self,
        model: LSTMForecaster,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: LSTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = config.device.torch_device
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_early_stopping()
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [], 'learning_rate': []
        }
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info("Trainer initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def _setup_optimizer(self) -> None:
        """Initialize optimizer."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        logger.info(f"Optimizer: Adam (lr={self.config.training.learning_rate}, wd={self.config.training.weight_decay})")
    
    def _setup_scheduler(self) -> None:
        """Initialize learning rate scheduler with warmup."""
        # Warmup scheduler for stable training start
        if self.config.training.warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < self.config.training.warmup_epochs:
                    # Linear warmup from warmup_start_lr to learning_rate
                    warmup_ratio = self.config.training.warmup_start_lr / self.config.training.learning_rate
                    return warmup_ratio + (1 - warmup_ratio) * epoch / self.config.training.warmup_epochs
                else:
                    return 1.0
            
            self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        else:
            self.warmup_scheduler = None
            
        # Main scheduler for learning rate decay
        self.main_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.training.scheduler_factor,
            patience=self.config.training.scheduler_patience,
            min_lr=self.config.training.scheduler_min_lr,
            verbose=True
        )
        
        self.warmup_completed = False
        logger.info(f"Scheduler initialized with {self.config.training.warmup_epochs} warmup epochs")
    
    def _setup_loss_function(self) -> None:
        """Initialize loss function."""
        self.criterion = nn.MSELoss()
        logger.info("Loss function: MSE")
    
    def _setup_early_stopping(self) -> None:
        """Initialize early stopping."""
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.early_stopping_min_delta
        )
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
            leave=False
        )
        
        for batch_idx, (sequences, targets) in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Move to device (should already be there from dataset)
            sequences = sequences.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = self.model.clip_gradients(self.config.training.grad_clip_max_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_metrics.update(loss.item(), batch_time, current_lr)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Grad': f"{grad_norm:.4f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
                raise ValueError("Training unstable: NaN/Inf loss detected")
        
        avg_loss = self.train_metrics.get_average_loss()
        avg_batch_time = self.train_metrics.get_average_batch_time()
        
        logger.info(f"Training - Avg Loss: {avg_loss:.6f}, Avg Batch Time: {avg_batch_time:.4f}s")
        
        return avg_loss
    
    def validate_epoch(self) -> float:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for sequences, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                self.val_metrics.update(loss.item())
        
        avg_loss = self.val_metrics.get_average_loss()
        logger.info(f"Validation - Avg Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def save_best_model(self) -> None:
        """Save the best model state."""
        self.best_model_state = self.model.state_dict().copy()
        
        # Save to file
        self.model.save_checkpoint(
            filepath=self.config.paths.best_model_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state={
                'main_scheduler': self.main_scheduler.state_dict(),
                'warmup_completed': self.warmup_completed,
                'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler else None
            },
            metrics={
                'best_val_loss': self.best_val_loss,
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                'epoch': self.current_epoch
            }
        )
        logger.info(f"Best model saved (val_loss: {self.best_val_loss:.6f})")
    
    def save_latest_model(self) -> None:
        """Save the latest model state."""
        self.model.save_checkpoint(
            filepath=self.config.paths.latest_model_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state={
                'main_scheduler': self.main_scheduler.state_dict(),
                'warmup_completed': self.warmup_completed,
                'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler else None
            },
            metrics={
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                'epoch': self.current_epoch
            }
        )
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.training.epochs}")
        
        # Create models directory
        Path(self.config.paths.models_dir).mkdir(exist_ok=True)
        
        # Validate model before training
        self.model.validate_shapes()
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Train and validate
                train_loss = self.train_epoch()
                val_loss = self.validate_epoch()
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Scheduler step with warmup support
                current_epoch = len(self.history['train_loss'])
                if self.warmup_scheduler and current_epoch <= self.config.training.warmup_epochs:
                    # Use warmup scheduler during warmup period
                    self.warmup_scheduler.step()
                else:
                    # Use main scheduler after warmup
                    if not self.warmup_completed:
                        logger.info(f"Warmup completed at epoch {current_epoch}. Switching to main scheduler.")
                        self.warmup_completed = True
                    self.main_scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_best_model()
                
                # Save latest model
                self.save_latest_model()
                
                # Early stopping check
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Log epoch summary
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.training.epochs} completed in {epoch_time:.2f}s - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            training_time = time.time() - training_start_time
            logger.info(f"Training completed in {training_time:.2f}s")
            
            # Load best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logger.info("Best model loaded for final evaluation")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        test_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in tqdm(test_loader, desc="Testing"):
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                test_losses.append(loss.item())
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Compute metrics
        avg_test_loss = np.mean(test_losses)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Additional metrics
        mse = torch.mean((all_predictions - all_targets) ** 2).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        metrics = {
            'test_loss': avg_test_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score.item()
        }
        
        logger.info("Test Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'total_epochs': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            'final_lr': self.history['learning_rate'][-1] if self.history['learning_rate'] else 0,
            'early_stopped': self.early_stopping.early_stop,
            'model_info': self.model.get_model_info(),
            'training_history': self.history
        }