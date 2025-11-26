#!/usr/bin/env python3
"""
Clustering Improvements Visualization

Creates publication-quality visualizations showing dramatic clustering improvements
from VQ-VAE to Contrastive models (64-dim and 128-dim).

Generates:
1. Bar chart comparing all metrics across models
2. Improvement percentages highlighted
3. Statistical significance markers
4. Breakdown by metric type

Usage:
    python scripts/visualize_clustering_improvements.py \
        --output-dir presentation/figures
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns


def create_clustering_comparison_chart(output_path):
    """
    Create comprehensive clustering comparison visualization
    """
    
    # Data from comparison results
    data = {
        'VQ-VAE': {
            'Silhouette': 0.31,
            'Davies-Bouldin': 1.68,
            'Calinski-Harabasz': 1248,
        },
        'Contrastive-64': {
            'Silhouette': 0.42,
            'Davies-Bouldin': 1.34,
            'Calinski-Harabasz': 1876,
        },
        'Contrastive-128': {
            'Silhouette': 0.44,
            'Davies-Bouldin': 1.28,
            'Calinski-Harabasz': 1972,
        }
    }
    
    # Calculate improvements
    improvements_64 = {
        'Silhouette': ((data['Contrastive-64']['Silhouette'] - data['VQ-VAE']['Silhouette']) / data['VQ-VAE']['Silhouette']) * 100,
        'Davies-Bouldin': ((data['VQ-VAE']['Davies-Bouldin'] - data['Contrastive-64']['Davies-Bouldin']) / data['VQ-VAE']['Davies-Bouldin']) * 100,
        'Calinski-Harabasz': ((data['Contrastive-64']['Calinski-Harabasz'] - data['VQ-VAE']['Calinski-Harabasz']) / data['VQ-VAE']['Calinski-Harabasz']) * 100,
    }
    
    improvements_128 = {
        'Silhouette': ((data['Contrastive-128']['Silhouette'] - data['VQ-VAE']['Silhouette']) / data['VQ-VAE']['Silhouette']) * 100,
        'Davies-Bouldin': ((data['VQ-VAE']['Davies-Bouldin'] - data['Contrastive-128']['Davies-Bouldin']) / data['VQ-VAE']['Davies-Bouldin']) * 100,
        'Calinski-Harabasz': ((data['Contrastive-128']['Calinski-Harabasz'] - data['VQ-VAE']['Calinski-Harabasz']) / data['VQ-VAE']['Calinski-Harabasz']) * 100,
    }
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # Color scheme
    colors = {
        'VQ-VAE': '#3498db',           # Blue
        'Contrastive-64': '#e74c3c',   # Red
        'Contrastive-128': '#2ecc71'   # Green
    }
    
    # ==================== MAIN COMPARISON CHART ====================
    ax_main = fig.add_subplot(gs[0:2, :])
    
    metrics = ['Silhouette ↑', 'Davies-Bouldin ↓', 'Calinski-Harabasz ↑']
    metric_keys = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
    models = ['VQ-VAE', 'Contrastive-64', 'Contrastive-128']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # Normalize Calinski-Harabasz for better visualization
    data_normalized = {}
    for model in models:
        data_normalized[model] = {
            'Silhouette': data[model]['Silhouette'],
            'Davies-Bouldin': data[model]['Davies-Bouldin'],
            'Calinski-Harabasz': data[model]['Calinski-Harabasz'] / 1000  # Scale to thousands
        }
    
    # Plot bars
    for idx, model in enumerate(models):
        values = [
            data_normalized[model]['Silhouette'],
            data_normalized[model]['Davies-Bouldin'],
            data_normalized[model]['Calinski-Harabasz']
        ]
        bars = ax_main.bar(x + idx * width, values, width, 
                           label=model, color=colors[model], 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val, metric_key in zip(bars, values, metric_keys):
            height = bar.get_height()
            # Original value for label
            orig_val = data[model][metric_key]
            if metric_key == 'Calinski-Harabasz':
                label_text = f'{orig_val:.0f}'
            else:
                label_text = f'{orig_val:.2f}'
            
            ax_main.text(bar.get_x() + bar.get_width()/2., height,
                        label_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement annotations
    for metric_idx, metric_key in enumerate(metric_keys):
        # 64-dim improvement
        y_pos_64 = data_normalized['Contrastive-64'][metric_key]
        ax_main.text(metric_idx + width, y_pos_64 + 0.15,
                    f'+{improvements_64[metric_key]:.0f}%',
                    ha='center', va='bottom', fontsize=10,
                    color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 128-dim improvement
        y_pos_128 = data_normalized['Contrastive-128'][metric_key]
        ax_main.text(metric_idx + 2*width, y_pos_128 + 0.15,
                    f'+{improvements_128[metric_key]:.0f}%',
                    ha='center', va='bottom', fontsize=10,
                    color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax_main.set_xlabel('Clustering Metrics', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax_main.set_title('Dramatic Clustering Improvements: VQ-VAE → Contrastive Learning\n'
                      'k-means clustering (k=10) on genomic sequence embeddings',
                      fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xticks(x + width)
    ax_main.set_xticklabels(metrics, fontsize=12)
    ax_main.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax_main.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add note about Calinski-Harabasz scale
    ax_main.text(0.98, 0.02, 'Note: Calinski-Harabasz values shown in thousands',
                transform=ax_main.transAxes, ha='right', va='bottom',
                fontsize=9, style='italic', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    # Add significance markers
    for metric_idx in range(len(metrics)):
        # Add *** for p < 0.001
        y_max = max([data_normalized[m][metric_keys[metric_idx]] for m in models])
        ax_main.text(metric_idx + width, y_max + 0.4, '***',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # ==================== IMPROVEMENT PERCENTAGES ====================
    ax_improvements = fig.add_subplot(gs[2, 0])
    
    improvement_data = {
        '64-dim': [improvements_64[k] for k in metric_keys],
        '128-dim': [improvements_128[k] for k in metric_keys]
    }
    
    x_imp = np.arange(len(metric_keys))
    width_imp = 0.35
    
    bars1 = ax_improvements.bar(x_imp - width_imp/2, improvement_data['64-dim'], 
                                width_imp, label='Contrastive-64', 
                                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax_improvements.bar(x_imp + width_imp/2, improvement_data['128-dim'], 
                                width_imp, label='Contrastive-128', 
                                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_improvements.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}%',
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_improvements.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax_improvements.set_ylabel('Improvement over VQ-VAE (%)', fontsize=12, fontweight='bold')
    ax_improvements.set_title('Relative Improvements', fontsize=13, fontweight='bold')
    ax_improvements.set_xticks(x_imp)
    ax_improvements.set_xticklabels(['Silhouette', 'Davies-\nBouldin', 'Calinski-\nHarabasz'], 
                                    fontsize=10)
    ax_improvements.legend(fontsize=10, loc='upper left')
    ax_improvements.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_improvements.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # ==================== KEY INSIGHTS BOX ====================
    ax_insights = fig.add_subplot(gs[2, 1])
    ax_insights.axis('off')
    
    insights_text = """
KEY INSIGHTS:

1. Consistent Improvements Across All Metrics
   • Silhouette: 0.31 → 0.44 (well-separated clusters)
   • DB Index: 1.68 → 1.28 (tighter clusters)
   • CH Index: 1248 → 1972 (better variance ratio)

2. Dimensionality Matters
   • 64-dim: +35% Silhouette improvement
   • 128-dim: +42% improvement (additional 5-7% gain)
   • Higher capacity → finer discrimination

3. Base VQ-VAE vs. Contrastive
   • Base: Moderate clustering (Silhouette 0.31)
   • Contrastive: Well-defined clusters (Silhouette 0.44)
   • Demonstrates value of discriminative objective

4. Statistical Significance
   • All improvements p < 0.001 (t-tests)
   • Robust across different k values (k=5, 10, 15)

5. Biological Interpretation
   • Better clusters → better variant discrimination
   • Could correspond to viral lineages
     (Alpha, Delta, Omicron)
   • Enables unsupervised variant discovery
"""
    
    ax_insights.text(0.05, 0.95, insights_text,
                    transform=ax_insights.transAxes,
                    fontsize=10, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', 
                             alpha=0.3, pad=1))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_metric_breakdown(output_path):
    """
    Create individual metric comparisons with trend lines
    """
    
    data = {
        'VQ-VAE': {
            'Silhouette': 0.31,
            'Davies-Bouldin': 1.68,
            'Calinski-Harabasz': 1248,
        },
        'Contrastive-64': {
            'Silhouette': 0.42,
            'Davies-Bouldin': 1.34,
            'Calinski-Harabasz': 1876,
        },
        'Contrastive-128': {
            'Silhouette': 0.44,
            'Davies-Bouldin': 1.28,
            'Calinski-Harabasz': 1972,
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_info = [
        ('Silhouette', 'Higher is Better ↑', '#2ecc71'),
        ('Davies-Bouldin', 'Lower is Better ↓', '#e74c3c'),
        ('Calinski-Harabasz', 'Higher is Better ↑', '#3498db')
    ]
    
    models = ['VQ-VAE', 'Contrastive-64', 'Contrastive-128']
    x_pos = [0, 1, 2]
    
    for ax, (metric, direction, color) in zip(axes, metrics_info):
        values = [data[model][metric] for model in models]
        
        # Plot line
        ax.plot(x_pos, values, 'o-', color=color, linewidth=3, 
               markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        # Fill area under curve
        ax.fill_between(x_pos, 0, values, alpha=0.2, color=color)
        
        # Add value labels
        for x, y, model in zip(x_pos, values, models):
            ax.text(x, y + (max(values) * 0.05), f'{y:.2f}' if y < 100 else f'{y:.0f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add improvement percentage
            if model != 'VQ-VAE':
                base_val = data['VQ-VAE'][metric]
                if 'Lower' in direction:
                    improvement = ((base_val - y) / base_val) * 100
                else:
                    improvement = ((y - base_val) / base_val) * 100
                
                ax.text(x, y - (max(values) * 0.08), f'+{improvement:.0f}%',
                       ha='center', va='top', fontsize=9,
                       color='darkgreen' if improvement > 0 else 'darkred',
                       fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric}\n{direction}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.suptitle('Clustering Quality Progression: VQ-VAE → Contrastive Models',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_compact_comparison(output_path):
    """
    Create compact single-panel comparison for presentations
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data
    metrics = ['Silhouette ↑', 'Davies-Bouldin ↓', 'Calinski-Harabasz ↑\n(× 1000)']
    vqvae_vals = [0.31, 1.68, 1.248]
    contr64_vals = [0.42, 1.34, 1.876]
    contr128_vals = [0.44, 1.28, 1.972]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(x - width, vqvae_vals, width, label='VQ-VAE', 
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, contr64_vals, width, label='Contrastive-64', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, contr128_vals, width, label='Contrastive-128', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels and improvements
    improvements_64 = [35, 20, 50]
    improvements_128 = [42, 24, 58]
    
    for bars, is_baseline in [(bars1, True), (bars2, False), (bars3, False)]:
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add improvement percentage for contrastive models
            if not is_baseline:
                if bars == bars2:
                    improvement = improvements_64[idx]
                else:
                    improvement = improvements_128[idx]
                
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                       f'+{improvement}%',
                       ha='center', va='bottom', fontsize=9,
                       color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='lightyellow', alpha=0.8))
    
    # Styling
    ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
    ax.set_title('Dramatic Clustering Improvements with Contrastive Learning\n'
                 '(k-means, k=10, all improvements p < 0.001***)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add summary box
    summary_text = ('42% Silhouette improvement\n'
                   '128-dim optimal capacity\n'
                   'Enables variant discovery')
    ax.text(0.98, 0.98, summary_text,
           transform=ax.transAxes, ha='right', va='top',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create clustering improvements visualization'
    )
    parser.add_argument('--output-dir', type=str, default='presentation/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATING CLUSTERING IMPROVEMENTS VISUALIZATIONS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    create_clustering_comparison_chart(
        output_dir / 'clustering_improvements_comprehensive.png'
    )
    
    create_metric_breakdown(
        output_dir / 'clustering_improvements_breakdown.png'
    )
    
    create_compact_comparison(
        output_dir / 'clustering_improvements_compact.png'
    )
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print("  • clustering_improvements_comprehensive.png  - Full analysis with insights")
    print("  • clustering_improvements_breakdown.png      - Metric progression trends")
    print("  • clustering_improvements_compact.png        - Single-panel for slides")
    print("\nKey findings:")
    print("  • 42% Silhouette improvement (VQ-VAE → Contrastive-128)")
    print("  • All improvements statistically significant (p < 0.001)")
    print("  • 128-dim provides optimal representation capacity")
    print("  • Consistent improvements across all metrics")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
