"""
LSTM Forecaster model for voltage trace prediction.

This module implements a multi-layer LSTM architecture for forecasting
next-timestep voltage values across multiple neurons simultaneously.
"""

import logging
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

from .config import ModelConfig, DeviceConfig


logger = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """
    Enhanced multi-layer LSTM model for voltage forecasting.
    
    Architecture:
    - 3 LSTM layers with 512 hidden units each
    - Residual connections for improved gradient flow
    - Layer normalization for training stability
    - Dropout for regularization
    - Dense output layer for multi-neuron prediction
    """
    
    def __init__(self, config: ModelConfig, device_config: DeviceConfig) -> None:
        """
        Initialize LSTM forecaster.
        
        Args:
            config: Model configuration
            device_config: Device configuration
        """
        super().__init__()
        
        self.config = config
        self.device_config = device_config
        self.device = device_config.torch_device
        
        # Validate configuration
        self._validate_config()
        
        # Enhanced LSTM architecture
        self.use_residual = config.use_residual
        self.use_layer_norm = config.use_layer_norm
        self.use_attention = config.use_attention
        self.use_cnn_features = config.use_cnn_features
        
        # CNN feature extraction (optional)
        if self.use_cnn_features:
            self.cnn_features = nn.Sequential(
                # 1D convolution across neurons (spatial dimension)
                nn.Conv1d(in_channels=config.sequence_length, out_channels=config.sequence_length, 
                         kernel_size=3, padding=1, groups=1),
                nn.ReLU(),
                nn.BatchNorm1d(config.sequence_length),
                nn.Dropout1d(0.1),
                
                # Second conv layer for deeper features
                nn.Conv1d(in_channels=config.sequence_length, out_channels=config.sequence_length,
                         kernel_size=5, padding=2, groups=1),
                nn.ReLU(),
                nn.BatchNorm1d(config.sequence_length),
                nn.Dropout1d(0.1)
            )
            lstm_input_size = config.input_size  # CNN preserves neuron dimension
        else:
            lstm_input_size = config.input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bias=True
        )
        
        # Layer normalization (applied after LSTM)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Residual connection projection (if input/output sizes differ)
        if self.use_residual and config.input_size != config.hidden_size:
            self.residual_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # Attention mechanism for temporal focus
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )
            self.attention_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"LSTMForecaster initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
        logger.info(f"Model architecture: {self}")
    
    def _validate_config(self) -> None:
        """Validate model configuration parameters."""
        if self.config.input_size != self.config.output_size:
            raise ValueError(
                f"Input size ({self.config.input_size}) must equal output size ({self.config.output_size}) "
                "for voltage forecasting"
            )
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using best practices."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: zeros with forget gate bias = 1
                param.data.fill_(0)
                if 'bias_ih' in name:
                    # Set forget gate bias to 1 (LSTM has 4 gates: i, f, g, o)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
            elif 'output_layer.weight' in name:
                # Output layer: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'output_layer.bias' in name:
                # Output bias: zeros
                param.data.fill_(0)
        
        logger.info("Model weights initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM forecaster.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Validate input shape
        expected_shape = (x.shape[0], self.config.sequence_length, self.config.input_size)
        if x.shape != expected_shape:
            raise ValueError(f"Expected input shape {expected_shape}, got {x.shape}")
        
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # CNN feature extraction if enabled
        if self.use_cnn_features:
            # Apply CNN to extract spatial features across neurons
            # x shape: (batch_size, sequence_length, input_size)
            x_cnn = self.cnn_features(x)  # Same shape output
            # Add residual connection with original input
            x = x + x_cnn
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Ignore hidden states
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            # Self-attention on LSTM outputs
            attended_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out
            )
            # Residual connection + layer norm
            lstm_out = self.attention_layer_norm(lstm_out + attended_out)
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            last_output = self.layer_norm(last_output)
        
        # Residual connection if enabled
        if self.use_residual:
            # Use the last timestep of input for residual
            residual = x[:, -1, :]  # (batch_size, input_size)
            
            # Project residual if dimensions don't match
            if hasattr(self, 'residual_projection'):
                residual = self.residual_projection(residual)
            
            # Add residual connection
            last_output = last_output + residual
        
        # Apply dropout
        dropped_output = self.dropout(last_output)
        
        # Final prediction
        output = self.output_layer(dropped_output)  # (batch_size, output_size)
        
        return output
    
    def predict_sequence(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict multiple future steps (autoregressive prediction).
        
        Args:
            x: Input sequence of shape (batch_size, sequence_length, input_size)
            steps: Number of future steps to predict
            
        Returns:
            Predictions of shape (batch_size, steps, output_size)
        """
        self.eval()
        predictions = []
        current_input = x.clone()
        
        with torch.no_grad():
            for _ in range(steps):
                # Predict next step
                next_pred = self.forward(current_input)
                predictions.append(next_pred.unsqueeze(1))
                
                # Update input for next prediction
                # Remove first timestep, add prediction as last timestep
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    next_pred.unsqueeze(1)
                ], dim=1)
        
        return torch.cat(predictions, dim=1)
    
    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get LSTM hidden states and output for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (lstm_output, final_hidden, final_cell)
        """
        x = x.to(self.device)
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out, hidden, cell
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'total_parameters': self.count_parameters(),
            'lstm_parameters': sum(p.numel() for name, p in self.named_parameters() 
                                 if 'lstm' in name and p.requires_grad),
            'output_parameters': sum(p.numel() for name, p in self.named_parameters() 
                                   if 'output_layer' in name and p.requires_grad),
            'input_size': self.config.input_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'output_size': self.config.output_size,
            'dropout': self.config.dropout,
            'sequence_length': self.config.sequence_length,
            'device': str(self.device)
        }
    
    def validate_shapes(self, batch_size: int = 4) -> None:
        """
        Validate model with dummy input to ensure shapes are correct.
        
        Args:
            batch_size: Batch size for validation
        """
        logger.info("Validating model shapes...")
        
        # Create dummy input
        dummy_input = torch.randn(
            batch_size, 
            self.config.sequence_length, 
            self.config.input_size,
            device=self.device
        )
        
        self.eval()
        with torch.no_grad():
            try:
                # Test forward pass
                output = self.forward(dummy_input)
                expected_output_shape = (batch_size, self.config.output_size)
                
                if output.shape != expected_output_shape:
                    raise ValueError(f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}")
                
                # Test sequence prediction
                seq_pred = self.predict_sequence(dummy_input, steps=3)
                expected_seq_shape = (batch_size, 3, self.config.output_size)
                
                if seq_pred.shape != expected_seq_shape:
                    raise ValueError(f"Sequence prediction shape mismatch: expected {expected_seq_shape}, got {seq_pred.shape}")
                
                # Check for NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("Model output contains NaN or Inf values")
                
                logger.info(f"✓ Shape validation passed")
                logger.info(f"✓ Input shape: {dummy_input.shape}")
                logger.info(f"✓ Output shape: {output.shape}")
                logger.info(f"✓ Sequence prediction shape: {seq_pred.shape}")
                logger.info(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
                
            except Exception as e:
                logger.error(f"Shape validation failed: {e}")
                raise
    
    def get_gradient_norms(self) -> Dict[str, float]:
        """Get gradient norms for each parameter group."""
        grad_norms = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
            else:
                grad_norms[name] = 0.0
        
        return grad_norms
    
    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None, 
                       scheduler_state: Optional[Dict] = None, metrics: Optional[Dict] = None) -> None:
        """
        Save model checkpoint with full training state.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__,
            'device_config': self.device_config.__dict__,
            'model_info': self.get_model_info(),
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device_config: DeviceConfig) -> Tuple['LSTMForecaster', Dict]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device_config: Device configuration
            
        Returns:
            Tuple of (model, checkpoint_data)
        """
        logger.info(f"Loading checkpoint from {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device_config.torch_device)
        
        # Recreate model config
        model_config = ModelConfig(**checkpoint['model_config'])
        
        # Create model
        model = cls(model_config, device_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
        
        return model, checkpoint