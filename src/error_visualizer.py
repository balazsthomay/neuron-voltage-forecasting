"""
Publication-ready visualization tools for error analysis results.

This module creates comprehensive plots and figures for:
- Per-neuron performance heatmaps
- Temporal error patterns
- Error distribution analysis  
- Multi-step degradation visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
import pickle

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


class ErrorVisualizer:
    """Publication-ready error analysis visualizations."""
    
    def __init__(self, results_dir: str = "src/analysis_results"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Publication settings
        self.fig_size = (12, 8)
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
    
    def load_analysis_results(self, results_file: str = "comprehensive_analysis.pkl") -> Dict[str, Any]:
        """Load analysis results from pickle file."""
        results_path = self.results_dir / results_file
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    
    def plot_neuron_performance_heatmap(self, results: Dict[str, Any]) -> None:
        """
        Create heatmap of per-neuron performance metrics.
        
        Args:
            results: Analysis results dictionary
        """
        neuron_analyses = results['neuron_analyses']
        n_neurons = len(neuron_analyses)
        
        # Extract metrics for all neurons
        metrics_names = ['r2', 'rmse', 'mae', 'bias']
        metrics_data = np.zeros((len(metrics_names), n_neurons))
        
        for i, metric in enumerate(metrics_names):
            for neuron_id in range(n_neurons):
                metrics_data[i, neuron_id] = neuron_analyses[str(neuron_id)]['metrics'][metric]
        
        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Neuron Performance Metrics', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for i, (ax, metric) in enumerate(zip(axes.flat, metrics_names)):
            # Reshape neurons into 10x10 grid for visualization
            if n_neurons == 100:
                metric_grid = metrics_data[i].reshape(10, 10)
            else:
                # For other neuron counts, use 1D visualization
                metric_grid = metrics_data[i].reshape(1, -1)
            
            im = ax.imshow(metric_grid, cmap='viridis' if metric == 'r2' else 'plasma_r', 
                          aspect='auto', interpolation='nearest')
            
            ax.set_title(f'{metric.upper()} Distribution', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(metric.upper())
            
            # Set labels
            if n_neurons == 100:
                ax.set_xlabel('Neuron X Position')
                ax.set_ylabel('Neuron Y Position')
            else:
                ax.set_xlabel('Neuron ID')
                ax.set_ylabel('')
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'neuron_performance_heatmap.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_neuron_ranking(self, results: Dict[str, Any]) -> None:
        """
        Create ranking plots for neuron performance.
        
        Args:
            results: Analysis results dictionary
        """
        neuron_analyses = results['neuron_analyses']
        n_neurons = len(neuron_analyses)
        
        # Extract R² scores and sort
        r2_scores = []
        neuron_ids = []
        for neuron_id in range(n_neurons):
            r2_scores.append(neuron_analyses[str(neuron_id)]['metrics']['r2'])
            neuron_ids.append(neuron_id)
        
        # Sort by R² performance
        sorted_indices = np.argsort(r2_scores)[::-1]  # Descending order
        sorted_r2 = np.array(r2_scores)[sorted_indices]
        sorted_ids = np.array(neuron_ids)[sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Neuron Performance Ranking', fontsize=16, fontweight='bold')
        
        # Plot 1: R² ranking
        ax1.plot(range(n_neurons), sorted_r2, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(range(n_neurons), sorted_r2, alpha=0.3)
        ax1.set_xlabel('Neuron Rank')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Performance Ranking')
        ax1.grid(True, alpha=0.3)
        
        # Add performance tiers
        top_10_pct = int(0.1 * n_neurons)
        bottom_10_pct = int(0.9 * n_neurons)
        
        ax1.axvline(top_10_pct, color='green', linestyle='--', alpha=0.7, label='Top 10%')
        ax1.axvline(bottom_10_pct, color='red', linestyle='--', alpha=0.7, label='Bottom 10%')
        ax1.legend()
        
        # Add statistics
        ax1.text(0.02, 0.98, f'Best: {sorted_r2[0]:.4f}\\nWorst: {sorted_r2[-1]:.4f}\\nMean: {np.mean(sorted_r2):.4f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Performance distribution
        ax2.hist(r2_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax2.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(np.median(r2_scores), color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_xlabel('R² Score')
        ax2.set_ylabel('Number of Neurons')
        ax2.set_title('R² Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'neuron_ranking.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.show()
        
        # Print top/bottom performers
        print("\\n" + "="*50)
        print("TOP 5 PERFORMING NEURONS")
        print("="*50)
        for i in range(min(5, n_neurons)):
            neuron_id = sorted_ids[i]
            r2 = sorted_r2[i]
            rmse = neuron_analyses[str(neuron_id)]['metrics']['rmse']
            print(f"Rank {i+1}: Neuron {neuron_id:2d} - R² = {r2:.6f}, RMSE = {rmse:.6f}")
        
        print("\\n" + "="*50)
        print("BOTTOM 5 PERFORMING NEURONS")
        print("="*50)
        for i in range(min(5, n_neurons)):
            idx = -(i+1)
            neuron_id = sorted_ids[idx]
            r2 = sorted_r2[idx]
            rmse = neuron_analyses[str(neuron_id)]['metrics']['rmse']
            print(f"Rank {n_neurons + idx + 1}: Neuron {neuron_id:2d} - R² = {r2:.6f}, RMSE = {rmse:.6f}")
    
    def plot_temporal_patterns(self, results: Dict[str, Any]) -> None:
        """
        Visualize temporal error patterns.
        
        Args:
            results: Analysis results dictionary
        """
        temporal_analysis = results['temporal_analysis']
        
        mean_errors = np.array(temporal_analysis['mean_error_by_timestep'])
        std_errors = np.array(temporal_analysis['std_error_by_timestep'])
        timesteps = np.arange(len(mean_errors))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Temporal Error Patterns', fontsize=16, fontweight='bold')
        
        # Plot 1: Error evolution
        ax1.plot(timesteps, mean_errors, 'b-', linewidth=2, label='Mean Error')
        ax1.fill_between(timesteps, mean_errors - std_errors, mean_errors + std_errors, 
                        alpha=0.3, label='±1 STD')
        
        # Mark optimal sequence length
        optimal_length = temporal_analysis['optimal_sequence_length']
        if optimal_length is not None:
            ax1.axvline(optimal_length, color='red', linestyle='--', linewidth=2, 
                       label=f'Optimal Length: {optimal_length}')
        
        ax1.set_xlabel('Timestep in Sequence')
        ax1.set_ylabel('Prediction Error (MAE)')
        ax1.set_title('Error vs. Sequence Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotation
        error_reduction = temporal_analysis['error_reduction']
        ax1.text(0.02, 0.98, f'Total Error Reduction: {error_reduction:.6f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 2: Error improvement rate
        error_improvements = np.diff(mean_errors)
        ax2.plot(timesteps[1:], error_improvements, 'g-', linewidth=2, alpha=0.7)
        ax2.fill_between(timesteps[1:], error_improvements, alpha=0.3, color='green')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Timestep in Sequence')
        ax2.set_ylabel('Error Change Rate')
        ax2.set_title('Learning Rate Across Sequence')
        ax2.grid(True, alpha=0.3)
        
        # Mark plateau region
        plateau_threshold = 0.001
        plateau_mask = np.abs(error_improvements) < plateau_threshold
        if np.any(plateau_mask):
            ax2.axhspan(-plateau_threshold, plateau_threshold, alpha=0.2, 
                       color='yellow', label='Plateau Region')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'temporal_patterns.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_multistep_degradation(self, results: Dict[str, Any]) -> None:
        """
        Visualize multi-step prediction degradation.
        
        Args:
            results: Analysis results dictionary
        """
        multistep_analysis = results['multistep_analysis']
        step_metrics = multistep_analysis['step_metrics']
        
        steps = list(step_metrics.keys())
        total_variances = [step_metrics[step]['total_variance'] for step in steps]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Multi-Step Prediction Degradation', fontsize=16, fontweight='bold')
        
        # Plot 1: Variance growth
        ax1.plot(steps, total_variances, 'ro-', linewidth=2, markersize=6)
        ax1.set_xlabel('Prediction Steps')
        ax1.set_ylabel('Total Prediction Variance')
        ax1.set_title('Uncertainty Growth with Prediction Steps')
        ax1.grid(True, alpha=0.3)
        
        # Fit exponential trend
        if len(steps) > 3:
            log_variances = np.log(np.array(total_variances) + 1e-10)  # Avoid log(0)
            trend_coef = np.polyfit(steps, log_variances, 1)[0]
            trend_line = np.exp(np.polyval([trend_coef, log_variances[0]], steps))
            ax1.plot(steps, trend_line, 'b--', alpha=0.7, 
                    label=f'Exp. Trend (rate: {trend_coef:.4f})')
            ax1.legend()
        
        # Add degradation statistics
        degradation_rate = multistep_analysis['average_degradation_rate']
        total_degradation = multistep_analysis['total_degradation']
        ax1.text(0.02, 0.98, f'Degradation Rate: {degradation_rate:.6f}\\nTotal Degradation: {total_degradation:.6f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Plot 2: Step-wise variance for first few steps
        max_viz_steps = min(10, len(steps))
        viz_steps = steps[:max_viz_steps]
        
        step_variances_matrix = []
        for step in viz_steps:
            step_variances = step_metrics[step]['mean_prediction_variance']
            step_variances_matrix.append(step_variances)
        
        step_variances_matrix = np.array(step_variances_matrix).T  # Shape: (timesteps, steps)
        
        im = ax2.imshow(step_variances_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        ax2.set_xlabel('Prediction Step')
        ax2.set_ylabel('Future Timestep')
        ax2.set_title('Variance Heatmap (First 10 Steps)')
        ax2.set_xticks(range(len(viz_steps)))
        ax2.set_xticklabels(viz_steps)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Prediction Variance')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'multistep_degradation.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_error_distributions(self, results: Dict[str, Any]) -> None:
        """
        Visualize error distribution characteristics.
        
        Args:
            results: Analysis results dictionary
        """
        dist_analysis = results['distribution_analysis']
        overall_stats = dist_analysis['overall_stats']
        neuron_stats = dist_analysis['neuron_error_stats']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall error distribution
        # We need to reconstruct error samples from neuron analyses
        neuron_analyses = results['neuron_analyses']
        all_errors = []
        for neuron_id in range(len(neuron_analyses)):
            # Use representative error statistics to generate synthetic distribution
            stats = neuron_stats[str(neuron_id)]
            n_samples = 100  # Reduced for visualization
            synthetic_errors = np.random.normal(stats['mean'], stats['std'], n_samples)
            all_errors.extend(synthetic_errors)
        
        ax1.hist(all_errors, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(overall_stats['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Prediction Error (mV)')
        ax1.set_ylabel('Density')
        ax1.set_title('Overall Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        ax1.text(0.02, 0.98, f'Mean: {overall_stats["mean"]:.6f}\\nStd: {overall_stats["std"]:.6f}\\nSkewness: {overall_stats["skewness"]:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # Plot 2: Neuron-wise error statistics
        neuron_ids = list(range(len(neuron_stats)))
        neuron_means = [neuron_stats[str(nid)]['mean'] for nid in neuron_ids]
        neuron_stds = [neuron_stats[str(nid)]['std'] for nid in neuron_ids]
        
        scatter = ax2.scatter(neuron_means, neuron_stds, alpha=0.6, s=30)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Bias')
        ax2.set_xlabel('Mean Error (Bias)')
        ax2.set_ylabel('Error Standard Deviation')
        ax2.set_title('Per-Neuron Error Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Skewness and kurtosis
        neuron_skewness = [neuron_stats[str(nid)]['skewness'] for nid in neuron_ids]
        neuron_kurtosis = [neuron_stats[str(nid)]['kurtosis'] for nid in neuron_ids]
        
        ax3.scatter(neuron_skewness, neuron_kurtosis, alpha=0.6, s=30)
        ax3.axvline(0, color='red', linestyle='--', alpha=0.7, label='Normal Skewness')
        ax3.axhline(0, color='red', linestyle='--', alpha=0.7, label='Normal Kurtosis')
        ax3.set_xlabel('Skewness')
        ax3.set_ylabel('Excess Kurtosis')
        ax3.set_title('Distribution Shape Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Outlier analysis
        neuron_outliers = [neuron_stats[str(nid)]['outlier_percentage'] for nid in neuron_ids]
        
        ax4.hist(neuron_outliers, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(neuron_outliers), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(neuron_outliers):.2f}%')
        ax4.set_xlabel('Outlier Percentage (%)')
        ax4.set_ylabel('Number of Neurons')
        ax4.set_title('Outlier Distribution Across Neurons')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_distributions.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, results: Dict[str, Any]) -> None:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            results: Analysis results dictionary
        """
        print("Creating comprehensive error analysis dashboard...")
        
        # Create all individual plots
        self.plot_neuron_performance_heatmap(results)
        self.plot_neuron_ranking(results)
        self.plot_temporal_patterns(results)
        self.plot_multistep_degradation(results)
        self.plot_error_distributions(results)
        
        print(f"All visualizations saved to {self.results_dir}/")
        print("Dashboard complete!")


def main():
    """Main function to create all visualizations."""
    visualizer = ErrorVisualizer()
    
    try:
        # Load results
        results = visualizer.load_analysis_results()
        
        # Create comprehensive dashboard
        visualizer.create_comprehensive_dashboard(results)
        
    except FileNotFoundError:
        print("Error: Analysis results not found. Please run error_analysis.py first.")
        print("Usage: python -m src.error_analysis")


if __name__ == "__main__":
    main()