"""
Comprehensive error analysis module for CNN-LSTM voltage forecasting model.

This module provides detailed analysis of prediction errors including:
- Per-neuron performance metrics
- Temporal error patterns  
- Error distribution characterization
- Multi-step prediction degradation analysis
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
from dataclasses import dataclass

from .config import Config
from .lstm_forecaster import LSTMForecaster
from .data_loader import DataLoader as VoltageDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ErrorMetrics:
    """Container for error metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    bias: float
    std_error: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'mse': self.mse,
            'rmse': self.rmse, 
            'mae': self.mae,
            'r2': self.r2,
            'bias': self.bias,
            'std_error': self.std_error
        }


@dataclass
class NeuronAnalysis:
    """Analysis results for individual neuron."""
    neuron_id: int
    metrics: ErrorMetrics
    predictions: np.ndarray
    targets: np.ndarray
    errors: np.ndarray
    voltage_range: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'neuron_id': self.neuron_id,
            'metrics': self.metrics.to_dict(),
            'voltage_range': self.voltage_range,
            'error_stats': {
                'mean_error': float(np.mean(self.errors)),
                'median_error': float(np.median(self.errors)),
                'error_percentiles': {
                    '5th': float(np.percentile(self.errors, 5)),
                    '25th': float(np.percentile(self.errors, 25)),
                    '75th': float(np.percentile(self.errors, 75)),
                    '95th': float(np.percentile(self.errors, 95))
                }
            }
        }


class ErrorAnalyzer:
    """Comprehensive error analysis for voltage forecasting model."""
    
    def __init__(self, config: Config):
        """
        Initialize error analyzer.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.test_loader = None
        self.results_dir = Path("src/analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("ErrorAnalyzer initialized")
    
    def load_model_and_data(self) -> None:
        """Load trained model and test data."""
        logger.info("Loading trained model and test data...")
        
        # Load model
        model_path = self.config.paths.best_model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Best model not found at {model_path}")
        
        self.model, checkpoint = LSTMForecaster.load_checkpoint(
            model_path, self.config.device
        )
        self.model.eval()
        logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
        
        # Load data
        data_loader = VoltageDataLoader(self.config)
        _, _, self.test_loader = data_loader.create_data_loaders()
        logger.info("Test data loaded successfully")
    
    def compute_error_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> ErrorMetrics:
        """
        Compute comprehensive error metrics.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            ErrorMetrics object with all computed metrics
        """
        # Basic metrics
        errors = predictions - targets
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Bias and error statistics
        bias = np.mean(errors)
        std_error = np.std(errors)
        
        return ErrorMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2,
            bias=bias, std_error=std_error
        )
    
    def analyze_per_neuron_performance(self) -> Dict[int, NeuronAnalysis]:
        """
        Analyze performance metrics for each individual neuron.
        
        Returns:
            Dictionary mapping neuron_id to NeuronAnalysis
        """
        logger.info("Analyzing per-neuron performance...")
        
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and data must be loaded first")
        
        neuron_analyses = {}
        all_predictions = []
        all_targets = []
        
        # Collect all predictions
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                y_pred = self.model(X_batch)
                all_predictions.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N, 100)
        targets = np.concatenate(all_targets, axis=0)        # Shape: (N, 100)
        
        logger.info(f"Analyzing {predictions.shape[0]} samples across {predictions.shape[1]} neurons")
        
        # Analyze each neuron
        for neuron_id in range(predictions.shape[1]):
            neuron_pred = predictions[:, neuron_id]
            neuron_target = targets[:, neuron_id]
            neuron_errors = neuron_pred - neuron_target
            
            # Compute metrics
            metrics = self.compute_error_metrics(neuron_pred, neuron_target)
            
            # Voltage range
            voltage_range = (float(np.min(neuron_target)), float(np.max(neuron_target)))
            
            # Create analysis object
            analysis = NeuronAnalysis(
                neuron_id=neuron_id,
                metrics=metrics,
                predictions=neuron_pred,
                targets=neuron_target,
                errors=neuron_errors,
                voltage_range=voltage_range
            )
            
            neuron_analyses[neuron_id] = analysis
        
        logger.info(f"Completed per-neuron analysis for {len(neuron_analyses)} neurons")
        return neuron_analyses
    
    def analyze_temporal_patterns(self, n_sequences: int = 100) -> Dict[str, Any]:
        """
        Analyze temporal error patterns within input sequences.
        
        Args:
            n_sequences: Number of sequences to analyze
            
        Returns:
            Dictionary with temporal analysis results
        """
        logger.info(f"Analyzing temporal patterns across {n_sequences} sequences...")
        
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and data must be loaded first")
        
        # Get sample sequences for temporal analysis
        sequences_analyzed = 0
        timestep_errors = []  # List of arrays, each with 150 timesteps
        
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                batch_size = X_batch.shape[0]
                
                # Analyze prediction accuracy at each timestep within sequences
                for seq_idx in range(min(batch_size, n_sequences - sequences_analyzed)):
                    sequence = X_batch[seq_idx:seq_idx+1]  # Shape: (1, 150, 100)
                    target = y_batch[seq_idx:seq_idx+1]    # Shape: (1, 100)
                    
                    # Get LSTM hidden states at each timestep
                    lstm_out, _, _ = self.model.get_hidden_states(sequence)
                    # lstm_out shape: (1, 150, hidden_size)
                    
                    # For simplicity, measure how well each timestep's representation
                    # predicts the final output by computing intermediate predictions
                    timestep_preds = []
                    for t in range(sequence.shape[1]):  # 150 timesteps
                        # Use representation up to timestep t
                        partial_sequence = sequence[:, :t+1, :]
                        if partial_sequence.shape[1] == self.config.model.sequence_length:
                            pred = self.model(partial_sequence)
                        else:
                            # Pad with zeros if sequence too short
                            pad_length = self.config.model.sequence_length - partial_sequence.shape[1]
                            padded = torch.zeros(1, self.config.model.sequence_length, 100, 
                                               device=partial_sequence.device)
                            padded[:, pad_length:, :] = partial_sequence
                            pred = self.model(padded)
                        
                        # Compute error for this partial prediction
                        error = torch.mean(torch.abs(pred - target)).item()
                        timestep_preds.append(error)
                    
                    timestep_errors.append(timestep_preds)
                    sequences_analyzed += 1
                    
                    if sequences_analyzed >= n_sequences:
                        break
                
                if sequences_analyzed >= n_sequences:
                    break
        
        # Convert to numpy array for analysis
        timestep_errors = np.array(timestep_errors)  # Shape: (n_sequences, 150)
        
        # Compute statistics across sequences
        mean_error_by_timestep = np.mean(timestep_errors, axis=0)
        std_error_by_timestep = np.std(timestep_errors, axis=0)
        
        # Find optimal sequence length (where error plateaus)
        error_improvements = np.diff(mean_error_by_timestep)
        plateau_threshold = 0.001  # Minimal improvement threshold
        optimal_length = None
        for i in range(len(error_improvements) - 10):  # Need some history
            if np.all(np.abs(error_improvements[i:i+10]) < plateau_threshold):
                optimal_length = i + 10
                break
        
        results = {
            'mean_error_by_timestep': mean_error_by_timestep.tolist(),
            'std_error_by_timestep': std_error_by_timestep.tolist(),
            'sequences_analyzed': sequences_analyzed,
            'optimal_sequence_length': optimal_length,
            'final_error': float(mean_error_by_timestep[-1]),
            'error_reduction': float(mean_error_by_timestep[0] - mean_error_by_timestep[-1]),
            'timestep_error_correlation': float(np.corrcoef(
                range(len(mean_error_by_timestep)), mean_error_by_timestep
            )[0, 1])
        }
        
        logger.info(f"Temporal analysis completed. Optimal sequence length: {optimal_length}")
        return results
    
    def analyze_multistep_degradation(self, max_steps: int = 20, n_samples: int = 50) -> Dict[str, Any]:
        """
        Analyze error accumulation in multi-step autoregressive prediction.
        
        Args:
            max_steps: Maximum prediction steps to analyze
            n_samples: Number of samples to test
            
        Returns:
            Dictionary with multi-step analysis results
        """
        logger.info(f"Analyzing multi-step prediction degradation up to {max_steps} steps...")
        
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and data must be loaded first")
        
        # Get sample sequences
        sample_sequences = []
        samples_collected = 0
        
        for X_batch, _ in self.test_loader:
            batch_size = X_batch.shape[0]
            take_samples = min(batch_size, n_samples - samples_collected)
            sample_sequences.append(X_batch[:take_samples])
            samples_collected += take_samples
            
            if samples_collected >= n_samples:
                break
        
        sample_sequences = torch.cat(sample_sequences, dim=0)[:n_samples]
        
        # Analyze degradation for each step
        step_metrics = {}
        
        self.model.eval()
        with torch.no_grad():
            for steps in range(1, max_steps + 1):
                step_predictions = []
                
                for seq_idx in range(sample_sequences.shape[0]):
                    sequence = sample_sequences[seq_idx:seq_idx+1]
                    
                    # Make multi-step prediction
                    pred_sequence = self.model.predict_sequence(sequence, steps=steps)
                    step_predictions.append(pred_sequence.cpu().numpy())
                
                # Compute step-wise error statistics
                step_predictions = np.concatenate(step_predictions, axis=0)
                
                # For analysis, we measure prediction variance as proxy for uncertainty
                step_variance = np.var(step_predictions, axis=0)  # Variance across samples
                # step_variance shape: (steps, neurons)
                mean_step_variance = np.mean(step_variance, axis=1)  # Average over neurons
                
                step_metrics[steps] = {
                    'mean_prediction_variance': mean_step_variance.tolist(),
                    'total_variance': float(np.sum(mean_step_variance)),
                    'variance_growth_rate': float(np.mean(np.diff(mean_step_variance))) if steps > 1 else 0.0
                }
        
        # Compute degradation statistics
        variances = [step_metrics[s]['total_variance'] for s in range(1, max_steps + 1)]
        degradation_rate = np.mean(np.diff(variances))
        
        results = {
            'step_metrics': step_metrics,
            'samples_analyzed': n_samples,
            'max_steps': max_steps,
            'average_degradation_rate': float(degradation_rate),
            'variance_at_step_1': float(variances[0]),
            'variance_at_max_step': float(variances[-1]),
            'total_degradation': float(variances[-1] - variances[0])
        }
        
        logger.info(f"Multi-step analysis completed. Degradation rate: {degradation_rate:.6f}")
        return results
    
    def analyze_error_distributions(self, neuron_analyses: Dict[int, NeuronAnalysis]) -> Dict[str, Any]:
        """
        Analyze error distribution characteristics across all neurons.
        
        Args:
            neuron_analyses: Per-neuron analysis results
            
        Returns:
            Dictionary with distribution analysis results
        """
        logger.info("Analyzing error distributions...")
        
        # Collect all errors
        all_errors = []
        neuron_error_stats = {}
        
        for neuron_id, analysis in neuron_analyses.items():
            errors = analysis.errors
            all_errors.extend(errors.tolist())
            
            # Statistical tests for normality and bias
            neuron_error_stats[neuron_id] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'skewness': float(self._compute_skewness(errors)),
                'kurtosis': float(self._compute_kurtosis(errors)),
                'outlier_percentage': float(self._compute_outlier_percentage(errors))
            }
        
        all_errors = np.array(all_errors)
        
        # Overall distribution statistics
        overall_stats = {
            'mean': float(np.mean(all_errors)),
            'std': float(np.std(all_errors)),
            'skewness': float(self._compute_skewness(all_errors)),
            'kurtosis': float(self._compute_kurtosis(all_errors)),
            'outlier_percentage': float(self._compute_outlier_percentage(all_errors))
        }
        
        # Error range analysis
        error_ranges = {
            'min_error': float(np.min(all_errors)),
            'max_error': float(np.max(all_errors)),
            'error_span': float(np.max(all_errors) - np.min(all_errors)),
            'percentiles': {
                '1st': float(np.percentile(all_errors, 1)),
                '5th': float(np.percentile(all_errors, 5)),
                '25th': float(np.percentile(all_errors, 25)),
                '50th': float(np.percentile(all_errors, 50)),
                '75th': float(np.percentile(all_errors, 75)),
                '95th': float(np.percentile(all_errors, 95)),
                '99th': float(np.percentile(all_errors, 99))
            }
        }
        
        results = {
            'overall_stats': overall_stats,
            'neuron_error_stats': neuron_error_stats,
            'error_ranges': error_ranges,
            'total_samples': len(all_errors),
            'neurons_analyzed': len(neuron_analyses)
        }
        
        logger.info("Error distribution analysis completed")
        return results
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def _compute_outlier_percentage(self, data: np.ndarray, threshold: float = 3.0) -> float:
        """Compute percentage of outliers using z-score threshold."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        z_scores = np.abs((data - mean) / std)
        outliers = np.sum(z_scores > threshold)
        return (outliers / len(data)) * 100
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run complete error analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive error analysis...")
        
        # Load model and data
        self.load_model_and_data()
        
        # Run all analyses
        results = {}
        
        # 1. Per-neuron analysis
        logger.info("Step 1/4: Per-neuron performance analysis")
        neuron_analyses = self.analyze_per_neuron_performance()
        results['neuron_analyses'] = {
            str(nid): analysis.to_dict() for nid, analysis in neuron_analyses.items()
        }
        
        # 2. Temporal patterns
        logger.info("Step 2/4: Temporal pattern analysis")
        results['temporal_analysis'] = self.analyze_temporal_patterns()
        
        # 3. Multi-step degradation
        logger.info("Step 3/4: Multi-step degradation analysis")
        results['multistep_analysis'] = self.analyze_multistep_degradation()
        
        # 4. Error distributions
        logger.info("Step 4/4: Error distribution analysis")
        results['distribution_analysis'] = self.analyze_error_distributions(neuron_analyses)
        
        # Save results
        self.save_analysis_results(results)
        
        logger.info("Comprehensive error analysis completed successfully")
        return results
    
    def save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to files."""
        # Save detailed results as pickle
        results_file = self.results_dir / "comprehensive_analysis.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON-serializable format
        summary_file = self.results_dir / "analysis_summary.pkl"
        summary = {
            'overall_metrics': {
                'neurons_analyzed': results['distribution_analysis']['neurons_analyzed'],
                'total_samples': results['distribution_analysis']['total_samples'],
                'best_neuron_r2': max(
                    results['neuron_analyses'][str(nid)]['metrics']['r2'] 
                    for nid in range(len(results['neuron_analyses']))
                ),
                'worst_neuron_r2': min(
                    results['neuron_analyses'][str(nid)]['metrics']['r2'] 
                    for nid in range(len(results['neuron_analyses']))
                ),
                'mean_r2': np.mean([
                    results['neuron_analyses'][str(nid)]['metrics']['r2'] 
                    for nid in range(len(results['neuron_analyses']))
                ])
            },
            'temporal_insights': {
                'optimal_sequence_length': results['temporal_analysis']['optimal_sequence_length'],
                'error_reduction': results['temporal_analysis']['error_reduction']
            },
            'multistep_insights': {
                'degradation_rate': results['multistep_analysis']['average_degradation_rate'],
                'total_degradation': results['multistep_analysis']['total_degradation']
            }
        }
        
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Analysis results saved to {self.results_dir}")


def main():
    """Main function to run error analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = Config()
    
    # Run analysis
    analyzer = ErrorAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    dist_analysis = results['distribution_analysis']
    print(f"Neurons analyzed: {dist_analysis['neurons_analyzed']}")
    print(f"Total samples: {dist_analysis['total_samples']:,}")
    
    # Best/worst neurons
    neuron_r2s = [
        results['neuron_analyses'][str(nid)]['metrics']['r2'] 
        for nid in range(dist_analysis['neurons_analyzed'])
    ]
    best_neuron = np.argmax(neuron_r2s)
    worst_neuron = np.argmin(neuron_r2s)
    
    print(f"\nBest performing neuron: {best_neuron} (R² = {neuron_r2s[best_neuron]:.6f})")
    print(f"Worst performing neuron: {worst_neuron} (R² = {neuron_r2s[worst_neuron]:.6f})")
    print(f"Mean R² across neurons: {np.mean(neuron_r2s):.6f}")
    
    # Temporal insights
    temporal = results['temporal_analysis']
    print(f"\nOptimal sequence length: {temporal['optimal_sequence_length']}")
    print(f"Error reduction with full sequence: {temporal['error_reduction']:.6f}")
    
    # Multi-step insights
    multistep = results['multistep_analysis']
    print(f"\nMulti-step degradation rate: {multistep['average_degradation_rate']:.6f}")
    print(f"Total degradation over 20 steps: {multistep['total_degradation']:.6f}")
    
    print(f"\nDetailed results saved to src/analysis_results/")
    print("="*60)


if __name__ == "__main__":
    main()