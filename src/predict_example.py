"""
Example script showing how to load the trained CNN-LSTM model and make predictions.

This demonstrates the complete workflow for using the trained model:
1. Load the best model checkpoint
2. Load and preprocess test data
3. Make single-step predictions
4. Make multi-step autoregressive predictions
5. Visualize results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .config import Config, DeviceConfig
from .lstm_forecaster import LSTMForecaster
from .data_loader import DataLoader as VoltageDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model() -> LSTMForecaster:
    """Load the best trained model from checkpoint."""
    logger.info("Loading trained CNN-LSTM model...")
    
    # Initialize configurations
    config = Config()
    device_config = DeviceConfig()
    
    # Load the best model
    model_path = config.paths.best_model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    # Load model from checkpoint
    model, checkpoint = LSTMForecaster.load_checkpoint(model_path, device_config)
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint and 'test_r2' in checkpoint['metrics']:
        logger.info(f"Model R² score: {checkpoint['metrics']['test_r2']:.6f}")
    else:
        logger.info("Model R² score: Not available in checkpoint")
    
    return model, checkpoint


def load_test_data() -> tuple:
    """Load preprocessed test data."""
    logger.info("Loading test data...")
    
    config = Config()
    data_loader = VoltageDataLoader(config)
    
    # Load data splits
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    return train_loader, val_loader, test_loader


def make_single_predictions(model: LSTMForecaster, test_loader) -> tuple:
    """Make single-step predictions on test data."""
    logger.info("Making single-step predictions...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            # Make predictions
            y_pred = model(X_batch)
            
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
            
            # Only process first few batches for demonstration
            if batch_idx >= 10:
                break
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    
    return predictions, targets


def make_sequence_predictions(model: LSTMForecaster, test_loader, steps: int = 10) -> tuple:
    """Make multi-step autoregressive predictions."""
    logger.info(f"Making {steps}-step sequence predictions...")
    
    model.eval()
    
    # Get first batch for demonstration
    X_batch, y_batch = next(iter(test_loader))
    
    # Take first sample from batch
    input_sequence = X_batch[0:1]  # Shape: (1, seq_len, n_neurons)
    
    with torch.no_grad():
        # Make sequence predictions
        sequence_pred = model.predict_sequence(input_sequence, steps=steps)
    
    logger.info(f"Input sequence shape: {input_sequence.shape}")
    logger.info(f"Sequence predictions shape: {sequence_pred.shape}")
    
    return input_sequence.cpu().numpy(), sequence_pred.cpu().numpy()


def visualize_predictions(predictions: np.ndarray, targets: np.ndarray, 
                         neuron_idx: int = 0, n_samples: int = 100):
    """Visualize single-step predictions vs targets for a specific neuron."""
    
    # Select subset for visualization
    pred_subset = predictions[:n_samples, neuron_idx]
    target_subset = targets[:n_samples, neuron_idx]
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Time series comparison
    plt.subplot(2, 2, 1)
    plt.plot(target_subset, label='True Voltage', alpha=0.7)
    plt.plot(pred_subset, label='Predicted Voltage', alpha=0.7)
    plt.title(f'Voltage Predictions vs True Values (Neuron {neuron_idx})')
    plt.xlabel('Sample')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(target_subset, pred_subset, alpha=0.6, s=20)
    plt.plot([target_subset.min(), target_subset.max()], 
             [target_subset.min(), target_subset.max()], 'r--', alpha=0.8)
    plt.xlabel('True Voltage (mV)')
    plt.ylabel('Predicted Voltage (mV)')
    plt.title('Prediction Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = 1 - np.sum((target_subset - pred_subset)**2) / np.sum((target_subset - np.mean(target_subset))**2)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot 3: Error distribution
    plt.subplot(2, 2, 3)
    errors = pred_subset - target_subset
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (mV)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Error vs true value
    plt.subplot(2, 2, 4)
    plt.scatter(target_subset, errors, alpha=0.6, s=20)
    plt.xlabel('True Voltage (mV)')
    plt.ylabel('Prediction Error (mV)')
    plt.title('Error vs True Value')
    plt.axhline(0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_sequence_predictions(input_seq: np.ndarray, sequence_pred: np.ndarray, 
                                 neuron_idx: int = 0):
    """Visualize multi-step sequence predictions."""
    
    # Extract data for specific neuron
    input_neuron = input_seq[0, :, neuron_idx]  # Shape: (seq_len,)
    pred_neuron = sequence_pred[0, :, neuron_idx]  # Shape: (steps,)
    
    plt.figure(figsize=(12, 6))
    
    # Plot input sequence
    input_time = np.arange(len(input_neuron))
    plt.plot(input_time, input_neuron, 'b-', label='Input Sequence', linewidth=2)
    
    # Plot predicted sequence
    pred_time = np.arange(len(input_neuron), len(input_neuron) + len(pred_neuron))
    plt.plot(pred_time, pred_neuron, 'r-', label='Predicted Sequence', linewidth=2, marker='o', markersize=4)
    
    # Add vertical line to separate input from prediction
    plt.axvline(len(input_neuron) - 0.5, color='gray', linestyle='--', alpha=0.7, label='Prediction Start')
    
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (mV)')
    plt.title(f'Multi-Step Voltage Prediction (Neuron {neuron_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/sequence_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    logger.info("=== CNN-LSTM Voltage Forecasting Model Usage Demo ===")
    
    try:
        # 1. Load trained model
        model, checkpoint = load_trained_model()
        
        # 2. Load test data
        _, _, test_loader = load_test_data()
        
        # 3. Make single-step predictions
        predictions, targets = make_single_predictions(model, test_loader)
        
        # 4. Make sequence predictions
        input_seq, sequence_pred = make_sequence_predictions(model, test_loader, steps=10)
        
        # 5. Visualize results
        logger.info("Creating visualizations...")
        visualize_predictions(predictions, targets, neuron_idx=0)
        visualize_sequence_predictions(input_seq, sequence_pred, neuron_idx=0)
        
        # 6. Print summary statistics
        logger.info("\n=== Prediction Summary ===")
        
        # Calculate metrics
        mse = np.mean((predictions - targets)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Calculate R² for each neuron
        r2_per_neuron = []
        for neuron in range(predictions.shape[1]):
            y_true = targets[:, neuron]
            y_pred = predictions[:, neuron]
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res / ss_tot)
            r2_per_neuron.append(r2)
        
        overall_r2 = np.mean(r2_per_neuron)
        
        logger.info(f"Overall MSE: {mse:.6f}")
        logger.info(f"Overall RMSE: {rmse:.6f}")
        logger.info(f"Overall MAE: {mae:.6f}")
        logger.info(f"Overall R²: {overall_r2:.6f}")
        logger.info(f"Best neuron R²: {np.max(r2_per_neuron):.6f}")
        logger.info(f"Worst neuron R²: {np.min(r2_per_neuron):.6f}")
        
        logger.info("\n=== Demo completed successfully! ===")
        logger.info("Check 'src/prediction_analysis.png' and 'src/sequence_prediction.png' for visualizations")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()