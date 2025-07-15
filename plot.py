#!/usr/bin/env python3
"""
Script per generare plot di ottimizzazione iperparametri per paper scientifico
Autore: [Il tuo nome]
Data: 2025-07-10
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
import argparse
from pathlib import Path

# Configurazione stile per paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configurazione font per pubblicazione
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_optimization_data(file_path):
    """Carica i dati dal file JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Converti in DataFrame per facilità di manipolazione
    all_trials = []
    pareto_trials_numbers = [trial['trial_number'] for trial in data['pareto_trials']]
    
    for trial in data['all_trials']:
        if trial['state'] == 'COMPLETE':
            trial_data = {
                'trial_number': trial['trial_number'],
                'lr': trial['params']['lr'],
                'loss_cam_weight': trial['params']['loss_cam_weight'],
                'variance_weight': trial['params']['variance_weight'],
                'accuracy': trial['accuracy'],
                'mse': trial['mse'],
                'is_pareto': trial['trial_number'] in pareto_trials_numbers
            }
            all_trials.append(trial_data)
    
    return pd.DataFrame(all_trials), data

def create_scatter_plots(df, output_dir):
    """Crea scatter plots per accuracy e MSE"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
    
    # Colori per Pareto e non-Pareto
    colors = ['#1f77b4' if not pareto else '#ff7f0e' for pareto in df['is_pareto']]
    sizes = [100 if pareto else 50 for pareto in df['is_pareto']]
    
    # Plot 1: Accuracy vs Learning Rate
    axes[0,0].scatter(df['lr'], df['accuracy'], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,0].set_xlabel('Learning Rate')
    axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,0].set_xscale('log')
    axes[0,0].set_title('Accuracy vs Learning Rate')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Loss Weight
    axes[0,1].scatter(df['loss_cam_weight'], df['accuracy'], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,1].set_xlabel('Loss Weight')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_title('Accuracy vs Loss Weight')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: MSE vs Learning Rate
    axes[1,0].scatter(df['lr'], df['mse'], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1,0].set_xlabel('Learning Rate')
    axes[1,0].set_ylabel('MSE')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('MSE vs Learning Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: MSE vs Loss Weight
    axes[1,1].scatter(df['loss_cam_weight'], df['mse'], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1,1].set_xlabel('Loss Weight')
    axes[1,1].set_ylabel('MSE')
    axes[1,1].set_yscale('log')
    axes[1,1].set_title('MSE vs Loss Weight')
    axes[1,1].grid(True, alpha=0.3)
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Regular trials'),
        Patch(facecolor='#ff7f0e', label='Pareto-optimal trials')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/scatter_plots.pdf', bbox_inches='tight')
    plt.show()

def create_pareto_front(df, output_dir):
    """Crea il plot del fronte di Pareto"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Tutti i trial
    regular_trials = df[~df['is_pareto']]
    pareto_trials = df[df['is_pareto']]
    
    # Plot tutti i trial
    ax.scatter(regular_trials['mse'], regular_trials['accuracy'], 
              c='lightblue', s=50, alpha=0.6, label='Regular trials', edgecolors='black', linewidth=0.5)
    
    # Plot trial Pareto
    ax.scatter(pareto_trials['mse'], pareto_trials['accuracy'], 
              c='red', s=100, alpha=0.8, label='Pareto-optimal trials', 
              edgecolors='darkred', linewidth=1, marker='s')
    
    # Linea del fronte di Pareto
    pareto_sorted = pareto_trials.sort_values('mse')
    ax.plot(pareto_sorted['mse'], pareto_sorted['accuracy'], 
           'r--', linewidth=2, alpha=0.7, label='Pareto front')
    
    ax.set_xlabel('MSE')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Pareto Front: Accuracy vs MSE Trade-off', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_front.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pareto_front.pdf', bbox_inches='tight')
    plt.show()

def create_heatmap(df, output_dir):
    """Crea heatmap dell'accuracy nello spazio dei parametri"""
    # Crea una griglia per l'interpolazione
    lr_range = np.logspace(np.log10(df['lr'].min()), np.log10(df['lr'].max()), 50)
    loss_weight_range = np.linspace(df['loss_cam_weight'].min(), df['loss_cam_weight'].max(), 50)
    
    # Griglia per interpolazione
    lr_grid, loss_weight_grid = np.meshgrid(lr_range, loss_weight_range)
    
    # Interpolazione dei dati
    points = np.column_stack((df['lr'], df['loss_cam_weight']))
    accuracy_grid = griddata(points, df['accuracy'], (lr_grid, loss_weight_grid), method='cubic')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap Accuracy
    im1 = ax1.contourf(lr_grid, loss_weight_grid, accuracy_grid, levels=20, cmap='viridis')
    ax1.scatter(df['lr'], df['loss_cam_weight'], c='white', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.scatter(df[df['is_pareto']]['lr'], df[df['is_pareto']]['loss_cam_weight'], 
               c='red', s=100, alpha=1, edgecolors='darkred', linewidth=1, marker='s')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Loss Weight')
    ax1.set_xscale('log')
    ax1.set_title('Accuracy Heatmap', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Accuracy (%)')
    
    # Heatmap MSE
    mse_grid = griddata(points, df['mse'], (lr_grid, loss_weight_grid), method='cubic')
    im2 = ax2.contourf(lr_grid, loss_weight_grid, mse_grid, levels=20, cmap='viridis_r')
    ax2.scatter(df['lr'], df['loss_cam_weight'], c='white', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.scatter(df[df['is_pareto']]['lr'], df[df['is_pareto']]['loss_cam_weight'], 
               c='red', s=100, alpha=1, edgecolors='darkred', linewidth=1, marker='s')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Loss Weight')
    ax2.set_xscale('log')
    ax2.set_title('MSE Heatmap', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='MSE')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmaps.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/heatmaps.pdf', bbox_inches='tight')
    plt.show()

def create_3d_surface(df, output_dir):
    """Crea superficie 3D dell'accuracy"""
    fig = plt.figure(figsize=(15, 6))
    
    # Preparazione griglia
    lr_range = np.logspace(np.log10(df['lr'].min()), np.log10(df['lr'].max()), 30)
    loss_weight_range = np.linspace(df['loss_cam_weight'].min(), df['loss_cam_weight'].max(), 30)
    lr_grid, loss_weight_grid = np.meshgrid(lr_range, loss_weight_range)
    
    points = np.column_stack((df['lr'], df['loss_cam_weight']))
    accuracy_grid = griddata(points, df['accuracy'], (lr_grid, loss_weight_grid), method='cubic')
    
    # Plot 3D Accuracy
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(np.log10(lr_grid), loss_weight_grid, accuracy_grid, 
                            cmap='viridis', alpha=0.8)
    ax1.scatter(np.log10(df['lr']), df['loss_cam_weight'], df['accuracy'], 
               c='red', s=50, alpha=1)
    ax1.set_xlabel('log10(Learning Rate)')
    ax1.set_ylabel('Loss Weight')
    ax1.set_zlabel('Accuracy (%)')
    ax1.set_title('3D Accuracy Surface')
    
    # Plot 3D MSE
    ax2 = fig.add_subplot(122, projection='3d')
    mse_grid = griddata(points, df['mse'], (lr_grid, loss_weight_grid), method='cubic')
    surf2 = ax2.plot_surface(np.log10(lr_grid), loss_weight_grid, mse_grid, 
                            cmap='viridis_r', alpha=0.8)
    ax2.scatter(np.log10(df['lr']), df['loss_cam_weight'], df['mse'], 
               c='red', s=50, alpha=1)
    ax2.set_xlabel('log10(Learning Rate)')
    ax2.set_ylabel('Loss Weight')
    ax2.set_zlabel('MSE')
    ax2.set_title('3D MSE Surface')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3d_surfaces.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/3d_surfaces.pdf', bbox_inches='tight')
    plt.show()

def create_correlation_analysis(df, output_dir):
    """Crea analisi di correlazione"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Matrice di correlazione
    corr_matrix = df[['lr', 'loss_cam_weight', 'variance_weight', 'accuracy', 'mse']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0,0], fmt='.3f', square=True)
    axes[0,0].set_title('Correlation Matrix')
    
    # Distribuzione accuracy
    axes[0,1].hist(df['accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].axvline(df['accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {df["accuracy"].mean():.2f}')
    axes[0,1].set_xlabel('Accuracy (%)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Accuracy Distribution')
    axes[0,1].legend()
    
    # Distribuzione MSE
    axes[1,0].hist(df['mse'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,0].axvline(df['mse'].mean(), color='red', linestyle='--', label=f'Mean: {df["mse"].mean():.4f}')
    axes[1,0].set_xlabel('MSE')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('MSE Distribution')
    axes[1,0].legend()
    
    # Box plot per variance weight
    variance_groups = df.groupby('variance_weight')
    group_data = [group['accuracy'].values for name, group in variance_groups]
    group_labels = [f'VW={name}' for name, group in variance_groups]
    
    axes[1,1].boxplot(group_data, labels=group_labels)
    axes[1,1].set_xlabel('Variance Weight')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].set_title('Accuracy vs Variance Weight')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/correlation_analysis.pdf', bbox_inches='tight')
    plt.show()

def create_summary_table(df, data, output_dir):
    """Crea tabella riassuntiva"""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"Total trials: {data['total_trials']}")
    print(f"Completed trials: {data['completed_trials']}")
    print(f"Pareto-optimal trials: {data['pareto_trials_count']}")
    print(f"Success rate: {data['completed_trials']/data['total_trials']*100:.1f}%")
    
    print("\nPerformance Statistics:")
    print(f"Max Accuracy: {df['accuracy'].max():.2f}% (Trial {df.loc[df['accuracy'].idxmax(), 'trial_number']})")
    print(f"Min MSE: {df['mse'].min():.6f} (Trial {df.loc[df['mse'].idxmin(), 'trial_number']})")
    print(f"Mean Accuracy: {df['accuracy'].mean():.2f} ± {df['accuracy'].std():.2f}%")
    print(f"Mean MSE: {df['mse'].mean():.6f} ± {df['mse'].std():.6f}")
    
    print("\nBest Pareto-optimal trials:")
    pareto_df = df[df['is_pareto']].sort_values('accuracy', ascending=False)
    for i, (_, row) in enumerate(pareto_df.head(3).iterrows()):
        print(f"{i+1}. Trial {row['trial_number']}: "
              f"Acc={row['accuracy']:.2f}%, MSE={row['mse']:.6f}, "
              f"LR={row['lr']:.2e}, LW={row['loss_cam_weight']:.3f}")
    
    print("\nOptimal parameter ranges (from Pareto trials):")
    print(f"Learning Rate: {pareto_df['lr'].min():.2e} - {pareto_df['lr'].max():.2e}")
    print(f"Loss Weight: {pareto_df['loss_cam_weight'].min():.3f} - {pareto_df['loss_cam_weight'].max():.3f}")
    
    # Salva tabella su file
    with open(f'{output_dir}/summary_table.txt', 'w') as f:
        f.write(f"Optimization Results Summary\n")
        f.write(f"Total trials: {data['total_trials']}\n")
        f.write(f"Completed trials: {data['completed_trials']}\n")
        f.write(f"Pareto-optimal trials: {data['pareto_trials_count']}\n")
        f.write(f"Max Accuracy: {df['accuracy'].max():.2f}%\n")
        f.write(f"Min MSE: {df['mse'].min():.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate optimization plots for scientific paper')
    parser.add_argument('--input_file', help='Input JSON file with optimization results')
    parser.add_argument('--output_dir', '-o', default='plots', help='Output directory for plots')

    args = parser.parse_args()

    root = "work/project/" if "work/project/" not in args.input_file else ""


    input_file = root + args.input_file

    if args.output_dir == 'plots':
        output_dir = args.input_file.replace('.json', '_plots')
        output_dir = root + output_dir
 
    else:
        output_dir = args.output_dir

    print(f"Input file: {args.input_file}, Output directory: {output_dir}")


    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input file: {input_file}")
    if not Path(input_file).is_file():
        print(f"Error: Input file '{input_file}' does not exist.")
        print("oss.listdir() output:")
        print("\n".join(str(p) for p in Path('.').iterdir()))
        return

    print(f"Loading data from {input_file}...")
    df, data = load_optimization_data(input_file)
    
    print(f"Generating plots in {output_dir}...")
    
    # Genera tutti i plot
    create_scatter_plots(df, output_dir)
    create_pareto_front(df, output_dir)
    create_heatmap(df, output_dir)
    create_3d_surface(df, output_dir)
    create_correlation_analysis(df, output_dir)
    create_summary_table(df, data, output_dir)
    
    print(f"\nAll plots saved in {output_dir}/")
    print("Files generated:")
    for file in output_dir.glob("*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()