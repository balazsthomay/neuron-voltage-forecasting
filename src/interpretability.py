"""
Comprehensive interpretability analysis for CNN-LSTM voltage forecasting model.

This module provides deep insights into model behavior including:
- CNN spatial pattern analysis
- LSTM dynamics and memory mechanisms  
- Feature importance and gradient-based attribution
- Residual connection analysis
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import pickle
from dataclasses import dataclass
import copy

from .config import Config
from .lstm_forecaster import LSTMForecaster
from .data_loader import DataLoader as VoltageDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ActivationHook:
    """Container for storing layer activations."""
    name: str
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    
    def forward_hook(self, module, input, output):
        """Hook function to capture forward activations."""
        if isinstance(output, tuple):
            # For LSTM, take the output tensor, not the hidden states
            self.activations = output[0].detach().clone()
        else:
            self.activations = output.detach().clone()
    
    def backward_hook(self, module, grad_input, grad_output):
        """Hook function to capture gradients."""
        if grad_output and grad_output[0] is not None:
            if isinstance(grad_output[0], tuple):
                # For LSTM, take the gradient of the output tensor
                self.gradients = grad_output[0][0].detach().clone()
            else:
                self.gradients = grad_output[0].detach().clone()


@dataclass
class CNNAnalysis:
    """CNN spatial pattern analysis results."""
    filter_weights: torch.Tensor
    activation_maps: Dict[int, torch.Tensor]
    spatial_correlations: np.ndarray
    pattern_clusters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'filter_shapes': list(self.filter_weights.shape),
            'n_activation_maps': len(self.activation_maps),
            'spatial_correlation_shape': self.spatial_correlations.shape,
            'n_pattern_clusters': len(self.pattern_clusters)
        }


@dataclass 
class LSTMAnalysis:
    """LSTM dynamics analysis results."""
    hidden_states: torch.Tensor  # Shape: (batch, seq_len, hidden_size)
    cell_states: torch.Tensor
    gate_activations: Dict[str, torch.Tensor]  # forget, input, output gates
    attention_weights: Optional[torch.Tensor]
    memory_retention: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hidden_states_shape': list(self.hidden_states.shape),
            'cell_states_shape': list(self.cell_states.shape),
            'gate_names': list(self.gate_activations.keys()),
            'has_attention': self.attention_weights is not None,
            'memory_metrics': self.memory_retention
        }


class ModelInterpreter:
    """Comprehensive model interpretability analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize model interpreter.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.test_loader = None
        self.hooks = {}
        self.results_dir = Path("src/interpretability_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("ModelInterpreter initialized")
    
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
    
    def register_hooks(self) -> None:
        """Register forward and backward hooks for key layers."""
        logger.info("Registering activation hooks...")
        
        # CNN layer hooks
        if hasattr(self.model, 'cnn_features'):
            for i, layer in enumerate(self.model.cnn_features):
                if isinstance(layer, torch.nn.Conv1d):
                    hook = ActivationHook(f'cnn_conv_{i}')
                    layer.register_forward_hook(hook.forward_hook)
                    # Skip backward hooks to avoid warnings
                    self.hooks[f'cnn_conv_{i}'] = hook
        
        # LSTM hook
        hook = ActivationHook('lstm')
        self.model.lstm.register_forward_hook(hook.forward_hook)
        self.hooks['lstm'] = hook
        
        # Output layer hook
        hook = ActivationHook('output_layer')
        self.model.output_layer.register_forward_hook(hook.forward_hook)
        self.hooks['output_layer'] = hook
        
        logger.info(f"Registered {len(self.hooks)} activation hooks")
    
    def analyze_cnn_spatial_patterns(self, n_samples: int = 5) -> CNNAnalysis:
        """
        Analyze CNN spatial pattern detection capabilities.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            CNNAnalysis with spatial pattern insights
        """
        logger.info("Analyzing CNN spatial patterns...")
        
        if not hasattr(self.model, 'cnn_features'):
            raise ValueError("Model does not have CNN features")
        
        # Extract filter weights
        conv_layers = [layer for layer in self.model.cnn_features if isinstance(layer, torch.nn.Conv1d)]
        filter_weights = []
        for i, layer in enumerate(conv_layers):
            weights = layer.weight.detach().cpu()
            # Store as list since different layers have different kernel sizes
            filter_weights.append({
                'layer_id': i,
                'kernel_size': weights.shape[2],
                'weights': weights
            })
        
        # Use first layer's weights for main analysis (can be extended)
        main_filter_weights = filter_weights[0]['weights'] if filter_weights else torch.zeros(1, 1, 3)
        
        # Analyze activation patterns
        activation_maps = {}
        spatial_responses = []
        
        sample_count = 0
        for X_batch, _ in self.test_loader:
            if sample_count >= n_samples:
                break
                
            batch_size = min(X_batch.shape[0], n_samples - sample_count)
            X_sample = X_batch[:batch_size]
            
            # Forward pass to capture CNN activations
            with torch.no_grad():
                _ = self.model(X_sample)
            
            # Collect CNN activations from hooks
            for hook_name, hook in self.hooks.items():
                if 'cnn_conv' in hook_name and hook.activations is not None:
                    if hook_name not in activation_maps:
                        activation_maps[hook_name] = []
                    activation_maps[hook_name].append(hook.activations.cpu())
            
            # Analyze spatial correlations
            if hasattr(self.model, 'cnn_features'):
                cnn_input = X_sample
                cnn_output = self.model.cnn_features(cnn_input)
                
                # Compute correlation between input neurons and CNN output
                for b in range(batch_size):
                    input_flat = cnn_input[b].mean(dim=0).detach().cpu().numpy()  # Average over time
                    output_flat = cnn_output[b].mean(dim=0).detach().cpu().numpy()  # Average over time
                    correlation = np.corrcoef(input_flat, output_flat)[0, 1]
                    spatial_responses.append(correlation)
            
            sample_count += batch_size
        
        # Compute spatial correlation matrix
        spatial_correlations = np.array(spatial_responses).reshape(-1, 1)
        if len(spatial_responses) > 1:
            spatial_correlations = np.corrcoef(spatial_responses, rowvar=False)
        
        # Pattern clustering analysis
        pattern_clusters = self._cluster_cnn_patterns(main_filter_weights, activation_maps)
        
        analysis = CNNAnalysis(
            filter_weights=main_filter_weights,
            activation_maps=activation_maps,
            spatial_correlations=spatial_correlations,
            pattern_clusters=pattern_clusters
        )
        
        logger.info(f"CNN analysis completed with {len(filter_weights)} filter layers")
        return analysis
    
    def analyze_lstm_dynamics(self, n_sequences: int = 5) -> LSTMAnalysis:
        """
        Analyze LSTM temporal dynamics and memory mechanisms.
        
        Args:
            n_sequences: Number of sequences to analyze
            
        Returns:
            LSTMAnalysis with temporal processing insights
        """
        logger.info("Analyzing LSTM dynamics...")
        
        # Collect LSTM internal states
        hidden_states_list = []
        cell_states_list = []
        sequences_analyzed = 0
        
        for X_batch, _ in self.test_loader:
            if sequences_analyzed >= n_sequences:
                break
                
            batch_size = min(X_batch.shape[0], n_sequences - sequences_analyzed)
            X_sample = X_batch[:batch_size]
            
            with torch.no_grad():
                # Get LSTM hidden states at each timestep
                lstm_out, hidden, cell = self.model.get_hidden_states(X_sample)
                
                hidden_states_list.append(lstm_out.cpu())
                cell_states_list.append(cell.transpose(0, 1).cpu())  # Shape: (batch, layers, hidden)
            
            sequences_analyzed += batch_size
        
        # Concatenate all sequences
        hidden_states = torch.cat(hidden_states_list, dim=0)  # (total_sequences, seq_len, hidden)
        cell_states = torch.cat(cell_states_list, dim=0)      # (total_sequences, layers, hidden)
        
        # Analyze gate activations (approximate from LSTM internals)
        gate_activations = self._analyze_lstm_gates(hidden_states, cell_states)
        
        # Memory retention analysis
        memory_retention = self._analyze_memory_retention(hidden_states)
        
        # Attention analysis (if available)
        attention_weights = None
        if hasattr(self.model, 'attention'):
            attention_weights = self._analyze_attention_patterns(X_sample[:1])
        
        analysis = LSTMAnalysis(
            hidden_states=hidden_states,
            cell_states=cell_states,
            gate_activations=gate_activations,
            attention_weights=attention_weights,
            memory_retention=memory_retention
        )
        
        logger.info(f"LSTM analysis completed on {sequences_analyzed} sequences")
        return analysis
    
    def analyze_feature_importance(self, n_samples: int = 3) -> Dict[str, Any]:
        """
        Analyze feature importance using gradient-based methods.
        
        Args:
            n_samples: Number of samples for importance analysis
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info("Analyzing feature importance...")
        
        importance_maps = []
        integrated_gradients = []
        neuron_importance = np.zeros((100,))  # 100 neurons
        
        sample_count = 0
        for X_batch, y_batch in self.test_loader:
            if sample_count >= n_samples:
                break
                
            batch_size = min(X_batch.shape[0], n_samples - sample_count)
            X_sample = X_batch[:batch_size].requires_grad_(True)
            y_sample = y_batch[:batch_size]
            
            # Forward pass
            predictions = self.model(X_sample)
            
            # Compute loss for gradient calculation
            loss = F.mse_loss(predictions, y_sample)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Extract input gradients (saliency)
            if X_sample.grad is not None:
                saliency = torch.abs(X_sample.grad).mean(dim=(0, 1))  # Average over batch and time
                importance_maps.append(saliency.cpu().numpy())
                
                # Per-neuron importance
                neuron_importance += saliency.cpu().numpy()
            
            # Skip integrated gradients for now due to gradient retention issues
            # integrated_grad = self._compute_integrated_gradients(
            #     X_sample, baseline, predictions, steps=10
            # )
            # integrated_gradients.append(integrated_grad)
            
            sample_count += batch_size
        
        # Aggregate results
        importance_maps = np.array(importance_maps)
        mean_importance = np.mean(importance_maps, axis=0)
        neuron_importance /= sample_count
        
        # Temporal importance analysis
        temporal_importance = self._analyze_temporal_importance(X_sample[:1])
        
        results = {
            'saliency_maps': importance_maps,
            'mean_importance': mean_importance,
            'neuron_importance': neuron_importance,
            'integrated_gradients': [],  # Disabled for now
            'temporal_importance': temporal_importance,
            'samples_analyzed': sample_count
        }
        
        logger.info(f"Feature importance analysis completed on {sample_count} samples")
        return results
    
    def analyze_residual_connections(self, n_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze the contribution of residual connections.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with residual connection analysis
        """
        logger.info("Analyzing residual connections...")
        
        # Create model copy without residual connections for comparison
        model_no_residual = copy.deepcopy(self.model)
        model_no_residual.use_residual = False
        model_no_residual.eval()
        
        residual_contributions = []
        performance_differences = []
        
        sample_count = 0
        for X_batch, y_batch in self.test_loader:
            if sample_count >= n_samples:
                break
                
            batch_size = min(X_batch.shape[0], n_samples - sample_count)
            X_sample = X_batch[:batch_size]
            y_sample = y_batch[:batch_size]
            
            with torch.no_grad():
                # Predictions with residual connections
                pred_with_residual = self.model(X_sample)
                
                # Predictions without residual connections
                pred_without_residual = model_no_residual(X_sample)
                
                # Compute residual contribution
                residual_contribution = torch.abs(pred_with_residual - pred_without_residual)
                residual_contributions.append(residual_contribution.cpu().numpy())
                
                # Performance difference
                mse_with = F.mse_loss(pred_with_residual, y_sample).item()
                mse_without = F.mse_loss(pred_without_residual, y_sample).item()
                performance_differences.append(mse_without - mse_with)  # Positive = residual helps
            
            sample_count += batch_size
        
        # Aggregate results
        residual_contributions = np.concatenate(residual_contributions, axis=0)
        mean_residual_contribution = np.mean(residual_contributions)
        mean_performance_improvement = np.mean(performance_differences)
        
        results = {
            'residual_contributions': residual_contributions,
            'mean_residual_contribution': float(mean_residual_contribution),
            'performance_improvement': float(mean_performance_improvement),
            'samples_analyzed': sample_count,
            'improvement_percentage': float(mean_performance_improvement / np.mean([abs(x) for x in performance_differences]) * 100)
        }
        
        logger.info(f"Residual connection analysis completed")
        return results
    
    def _cluster_cnn_patterns(self, filter_weights: torch.Tensor, 
                             activation_maps: Dict[str, List[torch.Tensor]]) -> Dict[str, Any]:
        """Cluster CNN filters by similarity."""
        # Simple clustering based on filter weight correlation
        n_filters = filter_weights.shape[0]
        filter_similarities = torch.zeros((n_filters, n_filters))
        
        for i in range(n_filters):
            for j in range(n_filters):
                correlation = F.cosine_similarity(
                    filter_weights[i].flatten().unsqueeze(0),
                    filter_weights[j].flatten().unsqueeze(0)
                ).item()
                filter_similarities[i, j] = correlation
        
        # Simple clustering: group filters with high correlation
        clusters = {}
        threshold = 0.7
        cluster_id = 0
        assigned = set()
        
        for i in range(n_filters):
            if i in assigned:
                continue
                
            cluster = [i]
            for j in range(i + 1, n_filters):
                if j not in assigned and filter_similarities[i, j] > threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            if len(cluster) > 1:
                clusters[f'cluster_{cluster_id}'] = {
                    'filters': cluster,
                    'size': len(cluster),
                    'avg_similarity': float(torch.mean(filter_similarities[cluster][:, cluster]))
                }
                cluster_id += 1
            
            assigned.add(i)
        
        return clusters
    
    def _analyze_lstm_gates(self, hidden_states: torch.Tensor, 
                           cell_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze LSTM gate activations (approximated)."""
        # Since we don't have direct access to gate activations,
        # we approximate by analyzing hidden/cell state changes
        
        seq_len = hidden_states.shape[1]
        
        # Approximate forget gate: how much cell state changes
        forget_approx = torch.zeros_like(hidden_states[:, :-1, :])
        for t in range(seq_len - 1):
            if t == 0:
                forget_approx[:, t, :] = torch.sigmoid(hidden_states[:, t, :])
            else:
                # Change in hidden state as proxy for forget gate
                change = torch.abs(hidden_states[:, t+1, :] - hidden_states[:, t, :])
                forget_approx[:, t, :] = torch.sigmoid(-change)  # Less change = more forgetting
        
        # Approximate input gate: magnitude of hidden state updates
        input_approx = torch.sigmoid(torch.abs(hidden_states[:, 1:, :] - hidden_states[:, :-1, :]))
        
        # Approximate output gate: relationship between hidden and cell states
        output_approx = torch.sigmoid(hidden_states[:, :, :])
        
        return {
            'forget_gate': forget_approx,
            'input_gate': input_approx,
            'output_gate': output_approx
        }
    
    def _analyze_memory_retention(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Analyze how well LSTM retains information over time."""
        seq_len = hidden_states.shape[1]
        
        # Compute autocorrelation at different lags
        autocorrelations = []
        for lag in [1, 5, 10, 25, 50]:
            if lag < seq_len:
                h1 = hidden_states[:, :-lag, :].flatten()
                h2 = hidden_states[:, lag:, :].flatten()
                correlation = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
                autocorrelations.append((lag, correlation))
        
        # Information decay rate
        if len(autocorrelations) > 1:
            decay_rate = (autocorrelations[0][1] - autocorrelations[-1][1]) / autocorrelations[-1][0]
        else:
            decay_rate = 0.0
        
        # Memory span: lag at which correlation drops below threshold
        memory_span = seq_len
        for lag, corr in autocorrelations:
            if corr < 0.5:  # Threshold for significant memory retention
                memory_span = lag
                break
        
        return {
            'autocorrelations': dict(autocorrelations),
            'decay_rate': float(decay_rate),
            'memory_span': int(memory_span),
            'initial_correlation': float(autocorrelations[0][1]) if autocorrelations else 0.0,
            'final_correlation': float(autocorrelations[-1][1]) if autocorrelations else 0.0
        }
    
    def _analyze_attention_patterns(self, X_sample: torch.Tensor) -> Optional[torch.Tensor]:
        """Analyze attention patterns if model has attention mechanism."""
        if not hasattr(self.model, 'attention'):
            return None
        
        with torch.no_grad():
            # Get LSTM output
            lstm_out, _ = self.model.get_hidden_states(X_sample)
            
            # Apply attention
            attended_out, attention_weights = self.model.attention(
                lstm_out, lstm_out, lstm_out
            )
            
        return attention_weights
    
    def _compute_integrated_gradients(self, input_tensor: torch.Tensor, 
                                    baseline: torch.Tensor, 
                                    target: torch.Tensor,
                                    steps: int = 50) -> np.ndarray:
        """Compute integrated gradients for input attribution."""
        integrated_grads = torch.zeros_like(input_tensor)
        
        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            pred = self.model(interpolated)
            loss = F.mse_loss(pred, target)
            loss.backward()
            
            if interpolated.grad is not None:
                integrated_grads += interpolated.grad
        
        integrated_grads = integrated_grads * (input_tensor - baseline) / steps
        return integrated_grads.detach().cpu().numpy()
    
    def _analyze_temporal_importance(self, X_sample: torch.Tensor) -> Dict[str, Any]:
        """Analyze importance of different timesteps."""
        seq_len = X_sample.shape[1]
        timestep_importance = np.zeros(seq_len)
        
        X_sample = X_sample.requires_grad_(True)
        
        # Get prediction
        prediction = self.model(X_sample)
        
        # For each timestep, compute gradient magnitude
        for t in range(seq_len):
            if X_sample.grad is not None:
                X_sample.grad.zero_()
            
            # Create mask that zeros out timestep t
            mask = torch.ones_like(X_sample)
            mask[:, t, :] = 0
            
            masked_input = X_sample * mask
            masked_pred = self.model(masked_input)
            
            # Importance = difference in prediction
            importance = F.mse_loss(prediction, masked_pred).item()
            timestep_importance[t] = importance
        
        return {
            'timestep_importance': timestep_importance.tolist(),
            'most_important_timestep': int(np.argmax(timestep_importance)),
            'least_important_timestep': int(np.argmin(timestep_importance)),
            'importance_range': float(np.max(timestep_importance) - np.min(timestep_importance))
        }
    
    def run_comprehensive_interpretability(self) -> Dict[str, Any]:
        """
        Run complete interpretability analysis pipeline.
        
        Returns:
            Dictionary with all interpretability results
        """
        logger.info("Starting comprehensive interpretability analysis...")
        
        # Load model and data
        self.load_model_and_data()
        
        # Register hooks for activation capture
        self.register_hooks()
        
        # Run all analyses
        results = {}
        
        # 1. CNN spatial pattern analysis
        logger.info("Step 1/4: CNN spatial pattern analysis")
        cnn_analysis = self.analyze_cnn_spatial_patterns()
        results['cnn_analysis'] = cnn_analysis.to_dict()
        
        # 2. LSTM dynamics analysis
        logger.info("Step 2/4: LSTM dynamics analysis")
        lstm_analysis = self.analyze_lstm_dynamics()
        results['lstm_analysis'] = lstm_analysis.to_dict()
        
        # 3. Feature importance analysis
        logger.info("Step 3/4: Feature importance analysis")
        results['feature_importance'] = self.analyze_feature_importance()
        
        # 4. Residual connection analysis
        logger.info("Step 4/4: Residual connection analysis")
        results['residual_analysis'] = self.analyze_residual_connections()
        
        # Save results
        self.save_interpretability_results(results, cnn_analysis, lstm_analysis)
        
        logger.info("Comprehensive interpretability analysis completed successfully")
        return results
    
    def save_interpretability_results(self, results: Dict[str, Any], 
                                    cnn_analysis: CNNAnalysis,
                                    lstm_analysis: LSTMAnalysis) -> None:
        """Save interpretability results to files."""
        # Save detailed results as pickle
        results_file = self.results_dir / "interpretability_analysis.pkl"
        full_results = {
            'summary': results,
            'cnn_analysis': cnn_analysis,
            'lstm_analysis': lstm_analysis
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(full_results, f)
        
        # Save summary
        summary_file = self.results_dir / "interpretability_summary.pkl"
        summary = {
            'cnn_insights': {
                'n_filters_analyzed': results['cnn_analysis']['filter_shapes'][0] if results['cnn_analysis']['filter_shapes'] else 0,
                'n_pattern_clusters': results['cnn_analysis']['n_pattern_clusters'],
                'spatial_correlation_detected': len(results['cnn_analysis']['spatial_correlation_shape']) > 1 if results['cnn_analysis']['spatial_correlation_shape'] else False
            },
            'lstm_insights': {
                'memory_span': lstm_analysis.memory_retention['memory_span'],
                'memory_decay_rate': lstm_analysis.memory_retention['decay_rate'],
                'has_attention': results['lstm_analysis']['has_attention']
            },
            'feature_insights': {
                'most_important_neuron': int(np.argmax(results['feature_importance']['neuron_importance'])),
                'importance_range': float(np.max(results['feature_importance']['neuron_importance']) - np.min(results['feature_importance']['neuron_importance']))
            },
            'residual_insights': {
                'performance_improvement': results['residual_analysis']['performance_improvement'],
                'improvement_percentage': results['residual_analysis']['improvement_percentage']
            }
        }
        
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Interpretability results saved to {self.results_dir}")


def main():
    """Main function to run interpretability analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = Config()
    
    # Run analysis
    interpreter = ModelInterpreter(config)
    results = interpreter.run_comprehensive_interpretability()
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL INTERPRETABILITY ANALYSIS SUMMARY")
    print("="*70)
    
    # CNN insights
    cnn = results['cnn_analysis']
    print(f"CNN Spatial Patterns:")
    print(f"  - Filters analyzed: {cnn['filter_shapes'][0] if cnn['filter_shapes'] else 0}")
    print(f"  - Pattern clusters found: {cnn['n_pattern_clusters']}")
    print(f"  - Activation maps captured: {cnn['n_activation_maps']}")
    
    # LSTM insights  
    lstm = results['lstm_analysis']
    print(f"\nLSTM Temporal Dynamics:")
    print(f"  - Hidden state shape: {lstm['hidden_states_shape']}")
    print(f"  - Gate activations: {', '.join(lstm['gate_names'])}")
    print(f"  - Has attention mechanism: {lstm['has_attention']}")
    
    # Feature importance
    feat = results['feature_importance']
    most_important = np.argmax(feat['neuron_importance'])
    print(f"\nFeature Importance:")
    print(f"  - Most important neuron: {most_important}")
    print(f"  - Samples analyzed: {feat['samples_analyzed']}")
    print(f"  - Importance range: {np.max(feat['neuron_importance']) - np.min(feat['neuron_importance']):.6f}")
    
    # Residual connections
    res = results['residual_analysis']
    print(f"\nResidual Connection Impact:")
    print(f"  - Performance improvement: {res['performance_improvement']:.6f}")
    print(f"  - Improvement percentage: {res['improvement_percentage']:.2f}%")
    print(f"  - Mean residual contribution: {res['mean_residual_contribution']:.6f}")
    
    print(f"\nDetailed results saved to src/interpretability_results/")
    print("="*70)


if __name__ == "__main__":
    main()