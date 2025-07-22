"""
Main training script for LSTM voltage forecasting system.

This script orchestrates the complete training pipeline including
data loading, model initialization, training, and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch

from .config import Config
from .data_loader import DataLoader as VoltageDataLoader
from .lstm_forecaster import LSTMForecaster
from .trainer import LSTMTrainer


def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log', mode='w')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def validate_environment() -> None:
    """Validate the training environment and dependencies."""
    logger = logging.getLogger(__name__)
    
    # Check PyTorch installation
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check device availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    
    # Check raw data directory
    config = Config()
    raw_data_dir = Path(config.data.raw_data_dir)
    
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    # Check for at least one simulation file
    voltage_files = list(raw_data_dir.glob("run_*_voltages.dat"))
    if not voltage_files:
        raise FileNotFoundError(f"No voltage simulation files found in {raw_data_dir}")
    
    logger.info(f"Found {len(voltage_files)} simulation files in {raw_data_dir}")
    
    # Check if cache exists (optional - will be created if needed)
    cache_file = Path(config.data.cache_file)
    if cache_file.exists():
        logger.info(f"Found existing cache: {cache_file}")
    else:
        logger.info("No cache found - will process raw data on first run")
    
    logger.info("Environment validation completed")


def create_model_and_data(config: Config) -> tuple:
    """
    Create model and data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (model, train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loader = VoltageDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    # Log data statistics
    train_stats = data_loader.get_data_stats(train_loader)
    val_stats = data_loader.get_data_stats(val_loader)
    
    logger.info("Training data statistics:")
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value:.6f}")
    
    logger.info("Validation data statistics:")
    for key, value in val_stats.items():
        logger.info(f"  {key}: {value:.6f}")
    
    # Create model
    logger.info("Creating LSTM model...")
    model = LSTMForecaster(config.model, config.device)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info("Model information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    return model, train_loader, val_loader, test_loader


def main() -> Dict[str, Any]:
    """
    Main training function.
    
    Returns:
        Training results dictionary
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("LSTM Voltage Forecasting Training Pipeline")
    logger.info("="*60)
    
    try:
        # Validate environment
        validate_environment()
        
        # Load configuration
        logger.info("Loading configuration...")
        config = Config()
        config.validate_compatibility()
        
        logger.info("Configuration loaded:")
        logger.info(f"  Device: {config.device.device}")
        logger.info(f"  Model: LSTM({config.model.input_size}, {config.model.hidden_size}, {config.model.num_layers})")
        logger.info(f"  Training: {config.training.epochs} epochs, batch_size={config.training.batch_size}")
        logger.info(f"  Data splits: {config.data.train_split:.1%}/{config.data.val_split:.1%}/{config.data.test_split:.1%}")
        
        # Create model and data
        model, train_loader, val_loader, test_loader = create_model_and_data(config)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = LSTMTrainer(model, train_loader, val_loader, config)
        
        # Train model
        logger.info("Starting training process...")
        training_history = trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        
        # Log final results
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total epochs trained: {training_summary['total_epochs']}")
        logger.info(f"Best validation loss: {training_summary['best_val_loss']:.6f}")
        logger.info(f"Final training loss: {training_summary['final_train_loss']:.6f}")
        logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
        logger.info(f"Test R² Score: {test_metrics['r2_score']:.6f}")
        logger.info(f"Early stopped: {training_summary['early_stopped']}")
        
        # Save results
        results = {
            'config': config.__dict__,
            'training_history': training_history,
            'training_summary': training_summary,
            'test_metrics': test_metrics,
            'model_info': model.get_model_info()
        }
        
        # Save training results
        results_path = Path(config.paths.models_dir) / "training_results.pt"
        torch.save(results, results_path)
        logger.info(f"Training results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def run_inference_example(config: Config) -> None:
    """
    Run a simple inference example with the trained model.
    
    Args:
        config: Configuration object
    """
    logger = logging.getLogger(__name__)
    logger.info("Running inference example...")
    
    try:
        # Load trained model
        model, checkpoint = LSTMForecaster.load_checkpoint(
            config.paths.best_model_path, 
            config.device
        )
        model.eval()
        
        # Create sample input (random voltage data)
        sample_input = torch.randn(
            1,  # batch_size = 1
            config.model.sequence_length,
            config.model.input_size,
            device=config.device.torch_device
        )
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sample_input)
            
            # Also test multi-step prediction
            multi_step_pred = model.predict_sequence(sample_input, steps=5)
        
        logger.info(f"Single step prediction shape: {prediction.shape}")
        logger.info(f"Multi-step prediction shape: {multi_step_pred.shape}")
        logger.info(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
        
        logger.info("Inference example completed successfully")
        
    except Exception as e:
        logger.error(f"Inference example failed: {e}")


if __name__ == "__main__":
    """Main entry point for training script."""
    
    # Run training
    results = main()
    
    # Run inference example
    config = Config()
    if Path(config.paths.best_model_path).exists():
        run_inference_example(config)
    else:
        print("No trained model found for inference example")
    
    print("\nTraining pipeline completed successfully!")
    print(f"Best model saved to: {config.paths.best_model_path}")
    print(f"Latest model saved to: {config.paths.latest_model_path}")
    print(f"Training logs saved to: training.log")