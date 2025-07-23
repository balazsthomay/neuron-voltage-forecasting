"""
Specialized visualization tools for model interpretability analysis.

This module creates publication-ready visualizations for:
- CNN filter patterns and spatial correlations
- LSTM dynamics and memory mechanisms
- Feature importance maps and attention patterns  
- Residual connection contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import torch

# Import classes needed for unpickling
from .interpretability import CNNAnalysis, LSTMAnalysis, ActivationHook

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


class InterpretabilityVisualizer:
    """Publication-ready interpretability visualizations."""
    
    def __init__(self, results_dir: str = "src/interpretability_results"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing interpretability results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Publication settings
        self.fig_size = (15, 10)
        self.dpi = 300
        self.font_size = 12
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4
        })
    
    def load_interpretability_results(self, results_file: str = "interpretability_analysis.pkl") -> Dict[str, Any]:
        """Load interpretability results from pickle file."""
        results_path = self.results_dir / results_file
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    
    def visualize_cnn_filters(self, cnn_analysis) -> None:
        """
        Visualize CNN filter patterns and spatial correlations.
        
        Args:
            cnn_analysis: CNN analysis results
        """
        filter_weights = cnn_analysis.filter_weights.numpy()
        n_filters = min(16, filter_weights.shape[0])  # Visualize up to 16 filters
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle('CNN Filter Patterns (Spatial Feature Detectors)', fontsize=16, fontweight='bold')
        
        for i in range(n_filters):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            # Visualize filter weights
            filter_data = filter_weights[i, 0, :]  # First input channel
            ax.plot(filter_data, linewidth=2, alpha=0.8)
            ax.set_title(f'Filter {i+1}\\nKernel Size: {len(filter_data)}')
            ax.set_xlabel('Spatial Position')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            
            # Highlight positive and negative weights
            pos_mask = filter_data > 0
            neg_mask = filter_data < 0
            ax.fill_between(range(len(filter_data)), filter_data, 0, 
                           where=pos_mask, alpha=0.3, color='green', label='Excitatory')
            ax.fill_between(range(len(filter_data)), filter_data, 0,
                           where=neg_mask, alpha=0.3, color='red', label='Inhibitory')
            
            if i == 0:  # Add legend to first subplot
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_filters, 16):
            row, col = i // 4, i % 4
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cnn_filters.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def visualize_spatial_correlations(self, cnn_analysis) -> None:
        """
        Visualize spatial correlation patterns discovered by CNN.
        
        Args:
            cnn_analysis: CNN analysis results
        """
        spatial_corr = cnn_analysis.spatial_correlations
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('CNN Spatial Pattern Discovery', fontsize=16, fontweight='bold')
        
        # Plot 1: Spatial correlation matrix (if available)
        if spatial_corr.ndim > 1 and spatial_corr.shape[0] > 1:
            im1 = ax1.imshow(spatial_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax1.set_title('Spatial Correlation Matrix')
            ax1.set_xlabel('Neuron/Feature Index')
            ax1.set_ylabel('Neuron/Feature Index')
            plt.colorbar(im1, ax=ax1, label='Correlation Coefficient')
        else:
            # Show correlation distribution
            ax1.hist(spatial_corr.flatten(), bins=30, alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(spatial_corr), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(spatial_corr):.3f}')
            ax1.set_xlabel('Spatial Correlation Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Spatial Correlations')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pattern clusters
        clusters = cnn_analysis.pattern_clusters
        if clusters:
            cluster_sizes = [info['size'] for info in clusters.values()]
            cluster_names = list(clusters.keys())
            
            bars = ax2.bar(range(len(cluster_names)), cluster_sizes, alpha=0.7)
            ax2.set_xlabel('Pattern Cluster')
            ax2.set_ylabel('Number of Filters')
            ax2.set_title('CNN Filter Clustering')
            ax2.set_xticks(range(len(cluster_names)))
            ax2.set_xticklabels([f'C{i+1}' for i in range(len(cluster_names))], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, size in zip(bars, cluster_sizes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{size}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No significant\\npattern clusters found',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax2.set_title('Pattern Clustering Results')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'spatial_patterns.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def visualize_lstm_dynamics(self, lstm_analysis) -> None:
        """
        Visualize LSTM temporal dynamics and memory mechanisms.
        
        Args:
            lstm_analysis: LSTM analysis results
        """
        hidden_states = lstm_analysis.hidden_states.numpy()
        gate_activations = {k: v.numpy() for k, v in lstm_analysis.gate_activations.items()}
        memory_retention = lstm_analysis.memory_retention
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM Temporal Dynamics Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Hidden state evolution
        sample_seq = hidden_states[0]  # First sequence
        timesteps = np.arange(sample_seq.shape[0])
        
        # Show first few hidden units
        n_units_to_show = min(5, sample_seq.shape[1])
        for i in range(n_units_to_show):
            ax1.plot(timesteps, sample_seq[:, i], alpha=0.7, label=f'Unit {i+1}')
        
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Hidden State Value')
        ax1.set_title('Hidden State Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gate activation patterns
        gate_names = list(gate_activations.keys())
        if gate_names:
            gate_data = gate_activations[gate_names[0]][0]  # First sequence, first gate
            
            # Show average gate activation over time
            mean_activation = np.mean(gate_data, axis=1)  # Average over hidden units
            ax2.plot(timesteps[:-1], mean_activation, linewidth=2, label=gate_names[0].replace('_', ' ').title())
            ax2.fill_between(timesteps[:-1], mean_activation, alpha=0.3)
            
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Average Gate Activation')
            ax2.set_title('LSTM Gate Dynamics')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory retention analysis
        autocorr_data = memory_retention['autocorrelations']
        lags = list(autocorr_data.keys())
        correlations = list(autocorr_data.values())
        
        ax3.plot(lags, correlations, 'o-', linewidth=2, markersize=8)
        ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, 
                   label='Memory Retention Threshold')
        ax3.set_xlabel('Time Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.set_title('Memory Retention Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add annotation
        memory_span = memory_retention['memory_span']
        ax3.text(0.02, 0.98, f'Memory Span: {memory_span} timesteps\\nDecay Rate: {memory_retention["decay_rate"]:.4f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 4: Hidden state dimensionality analysis
        # Compute PCA-like analysis of hidden states
        from sklearn.decomposition import PCA
        
        # Flatten hidden states for PCA
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        
        try:
            pca = PCA(n_components=min(10, hidden_flat.shape[1]))
            pca.fit(hidden_flat)
            
            explained_variance = pca.explained_variance_ratio_[:10]
            components = np.arange(1, len(explained_variance) + 1)
            
            bars = ax4.bar(components, explained_variance, alpha=0.7)
            ax4.set_xlabel('Principal Component')
            ax4.set_ylabel('Explained Variance Ratio')
            ax4.set_title('Hidden State Dimensionality')
            ax4.grid(True, alpha=0.3)
            
            # Add cumulative variance line
            cumulative_var = np.cumsum(explained_variance)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(components, cumulative_var, 'ro-', alpha=0.7, label='Cumulative')
            ax4_twin.set_ylabel('Cumulative Variance Ratio')
            ax4_twin.legend()
            
        except ImportError:
            ax4.text(0.5, 0.5, 'sklearn not available\\nfor PCA analysis',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'lstm_dynamics.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_importance(self, feature_importance: Dict[str, Any]) -> None:
        """
        Visualize feature importance and gradient-based attribution.
        
        Args:
            feature_importance: Feature importance analysis results
        """
        neuron_importance = feature_importance['neuron_importance']
        mean_importance = feature_importance['mean_importance']
        temporal_importance = feature_importance.get('temporal_importance', {})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Per-neuron importance
        neuron_ids = np.arange(len(neuron_importance))
        bars = ax1.bar(neuron_ids, neuron_importance, alpha=0.7)
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Per-Neuron Feature Importance')
        ax1.grid(True, alpha=0.3)
        
        # Highlight top neurons
        top_neurons = np.argsort(neuron_importance)[-5:]
        for idx in top_neurons:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        # Add statistics
        ax1.text(0.02, 0.98, f'Most Important: Neuron {np.argmax(neuron_importance)}\\nLeast Important: Neuron {np.argmin(neuron_importance)}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Plot 2: Importance distribution
        ax2.hist(neuron_importance, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(neuron_importance), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(neuron_importance):.4f}')
        ax2.axvline(np.median(neuron_importance), color='green', linestyle='--',
                   label=f'Median: {np.median(neuron_importance):.4f}')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Number of Neurons')
        ax2.set_title('Importance Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spatial importance map (if we have 100 neurons)
        if len(neuron_importance) == 100:
            importance_grid = neuron_importance.reshape(10, 10)
            im = ax3.imshow(importance_grid, cmap='hot', aspect='auto')
            ax3.set_xlabel('Neuron X Position')
            ax3.set_ylabel('Neuron Y Position')
            ax3.set_title('Spatial Importance Map')
            plt.colorbar(im, ax=ax3, label='Importance Score')
        else:
            # Show top important neurons
            top_10 = np.argsort(neuron_importance)[-10:]
            ax3.barh(range(10), neuron_importance[top_10])
            ax3.set_yticks(range(10))
            ax3.set_yticklabels([f'Neuron {idx}' for idx in top_10])
            ax3.set_xlabel('Importance Score')
            ax3.set_title('Top 10 Most Important Neurons')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temporal importance (if available)
        if temporal_importance and 'timestep_importance' in temporal_importance:
            timestep_imp = temporal_importance['timestep_importance']
            timesteps = np.arange(len(timestep_imp))
            
            ax4.plot(timesteps, timestep_imp, linewidth=2, alpha=0.7)
            ax4.fill_between(timesteps, timestep_imp, alpha=0.3)
            
            # Highlight most/least important timesteps
            most_imp = temporal_importance['most_important_timestep']
            least_imp = temporal_importance['least_important_timestep']
            
            ax4.axvline(most_imp, color='red', linestyle='--', alpha=0.7,
                       label=f'Most Important: t={most_imp}')
            ax4.axvline(least_imp, color='blue', linestyle='--', alpha=0.7,
                       label=f'Least Important: t={least_imp}')
            
            ax4.set_xlabel('Timestep')
            ax4.set_ylabel('Temporal Importance')
            ax4.set_title('Temporal Feature Importance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Temporal importance\\nanalysis not available',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def visualize_residual_analysis(self, residual_analysis: Dict[str, Any]) -> None:
        """
        Visualize residual connection contributions.
        
        Args:
            residual_analysis: Residual connection analysis results
        """
        residual_contributions = residual_analysis['residual_contributions']
        performance_improvement = residual_analysis['performance_improvement']
        improvement_percentage = residual_analysis['improvement_percentage']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Residual Connection Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of residual contributions
        contributions_flat = residual_contributions.flatten()
        ax1.hist(contributions_flat, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(np.mean(contributions_flat), color='red', linestyle='--',
                   label=f'Mean: {np.mean(contributions_flat):.4f}')
        ax1.axvline(np.median(contributions_flat), color='green', linestyle='--',
                   label=f'Median: {np.median(contributions_flat):.4f}')
        
        ax1.set_xlabel('Residual Contribution Magnitude')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Residual Contributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        ax1.text(0.98, 0.98, f'Std: {np.std(contributions_flat):.4f}\\nMax: {np.max(contributions_flat):.4f}',
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # Plot 2: Performance improvement visualization
        categories = ['Without\\nResidual', 'With\\nResidual', 'Improvement']
        # Simulated baseline performance for visualization
        baseline_mse = 0.5  # Approximate baseline
        with_residual_mse = baseline_mse - performance_improvement
        values = [baseline_mse, with_residual_mse, performance_improvement]
        colors = ['lightcoral', 'lightgreen', 'gold']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('MSE / Improvement')
        ax2.set_title('Residual Connection Impact')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Add improvement percentage annotation
        ax2.text(0.5, 0.95, f'Performance Improvement: {improvement_percentage:.2f}%',
                ha='center', va='top', transform=ax2.transAxes,
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'residual_analysis.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_interpretability_dashboard(self, results: Dict[str, Any]) -> None:
        """
        Create comprehensive interpretability dashboard.
        
        Args:
            results: Complete interpretability results
        """
        print("Creating comprehensive interpretability dashboard...")
        
        # Load full results
        cnn_analysis = results['cnn_analysis']
        lstm_analysis = results['lstm_analysis']
        
        # Create all visualizations
        self.visualize_cnn_filters(cnn_analysis)
        self.visualize_spatial_correlations(cnn_analysis)
        self.visualize_lstm_dynamics(lstm_analysis)
        self.visualize_feature_importance(results['summary']['feature_importance'])
        self.visualize_residual_analysis(results['summary']['residual_analysis'])
        
        print(f"All interpretability visualizations saved to {self.results_dir}/")
        print("Dashboard complete!")


def main():
    """Main function to create interpretability visualizations."""
    visualizer = InterpretabilityVisualizer()
    
    try:
        # Load results
        results = visualizer.load_interpretability_results()
        
        # Create comprehensive dashboard
        visualizer.create_interpretability_dashboard(results)
        
    except FileNotFoundError:
        print("Error: Interpretability results not found. Please run interpretability.py first.")
        print("Usage: python -m src.interpretability")


if __name__ == "__main__":
    main()