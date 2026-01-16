#!/usr/bin/env python
"""
Metrics Comparison Tool

Compare Q_Y vs alpha plots from multiple experiment runs.

HOW TO USE:
    1. Edit METRICS_PATHS below to add paths to your metrics.csv files
    2. Run: python compare_metrics.py
    
    Or use CLI: python compare_metrics.py path/to/file1.csv path/to/file2.csv
"""

from pathlib import Path

# ============================================================================
# ★★★ EDIT HERE: Add paths to metrics.csv files to compare ★★★
# ============================================================================

METRICS_PATHS = [
    "/Users/password-is-0000/Projects/Sparse-Matrix-Factorization/terao_gamp_gaussian/F_random/results/20260114_120550_gamp_F_random_1000x10_alpha0.5-5.0/metrics.csv",
    "/Users/password-is-0000/Projects/Sparse-Matrix-Factorization/terao_gd/results/20260114_121146_agd_1000x10_alpha0.5-5.0/metrics.csv",
]

# Optional: Custom labels for each curve (leave empty for auto-labels from config.yaml)
CUSTOM_LABELS = [
    "F_random",
    "AGD",
]

# Output settings
OUTPUT_PATH = "comparison_result.png"  # Set to "comparison.png" to save, or None to skip
PLOT_TITLE = "Q_Y vs Alpha Comparison"

# ============================================================================
# Configuration (advanced settings)
# ============================================================================

import argparse
import sys
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

COLORS = [
    '#1E88E5',  # Blue
    '#E53935',  # Red
    '#43A047',  # Green
    '#FB8C00',  # Orange
    '#8E24AA',  # Purple
    '#00ACC1',  # Cyan
    '#FFB300',  # Amber
    '#6D4C41',  # Brown
]

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']


# ============================================================================
# Data Loading
# ============================================================================

def load_metrics(csv_path: Path) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Load metrics.csv and optionally config.yaml from the same directory.
    
    Returns:
        df: DataFrame with columns [alpha, Q_Y_mean, Q_Y_std, ...]
        config: Configuration dict or None
    """
    df = pd.read_csv(csv_path)
    
    # Try to load config
    config_path = csv_path.parent / "config.yaml"
    config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    return df, config


def get_label_from_config(config: Optional[dict], csv_path: Path) -> str:
    """Generate a label from config or path."""
    if config:
        algo = config.get('algorithm', '')
        n1 = config.get('N1', '')
        m = config.get('M', '')
        onsager = config.get('onsager_correction', False)
        
        label = algo
        if n1 and m:
            label += f" (N={n1}, M={m})"
        if onsager:
            label += " +Onsager"
        return label
    
    # Fallback: use directory name
    return csv_path.parent.name


def find_metrics_in_dir(dir_path: Path, pattern: str = "**/metrics.csv") -> List[Path]:
    """Find all metrics.csv files in a directory."""
    return sorted(dir_path.glob(pattern))


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(
    csv_paths: List[Path],
    labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Q_Y vs Alpha Comparison",
    show_error_bars: bool = True,
    use_sem: bool = True,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot Q_Y vs alpha comparison from multiple metrics.csv files.
    
    Args:
        csv_paths: List of paths to metrics.csv files
        labels: Optional custom labels for each curve
        output_path: Optional path to save the figure
        title: Plot title
        show_error_bars: Whether to show error bars
        use_sem: Use SEM (std/sqrt(n)) instead of std for error bars
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, csv_path in enumerate(csv_paths):
        df, config = load_metrics(csv_path)
        
        # Get label
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = get_label_from_config(config, csv_path)
        
        # Extract data
        alphas = df['alpha'].values
        qy_means = df['Q_Y_mean'].values
        qy_stds = df['Q_Y_std'].values
        
        # Compute error bars
        if show_error_bars:
            # Try to get number of replicas from config or count qy_replica columns
            n_replicas = 10  # default
            if config and 'num_replicas' in config:
                n_replicas = config['num_replicas']
            else:
                replica_cols = [c for c in df.columns if c.startswith('qy_replica_')]
                if replica_cols:
                    n_replicas = len(replica_cols)
            
            if use_sem:
                yerr = qy_stds / np.sqrt(n_replicas)
            else:
                yerr = qy_stds
        else:
            yerr = None
        
        # Plot
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        
        ax.errorbar(
            alphas, qy_means, yerr=yerr,
            fmt=f'{marker}-', color=color, markersize=6, linewidth=2,
            capsize=4, capthick=1.5, elinewidth=1.5,
            label=label
        )
    
    # Style
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
    
    plt.show(block=False)
    plt.pause(0.1)  # Allow window to render
    
    return fig, ax


def interactive_mode():
    """Interactive mode to select metrics files."""
    print("=" * 60)
    print("Metrics Comparison Tool - Interactive Mode")
    print("=" * 60)
    print()
    
    # Find common results directories
    repo_root = Path(__file__).parent
    search_dirs = [
        repo_root / "terao_gamp_gaussian",
        repo_root / "Terao",
        repo_root / "terao_gd",
    ]
    
    all_metrics = []
    for search_dir in search_dirs:
        if search_dir.exists():
            all_metrics.extend(find_metrics_in_dir(search_dir))
    
    if not all_metrics:
        print("No metrics.csv files found in common directories.")
        return
    
    print(f"Found {len(all_metrics)} metrics files:\n")
    for i, path in enumerate(all_metrics):
        rel_path = path.relative_to(repo_root)
        print(f"  [{i+1}] {rel_path}")
    
    print()
    print("Enter the numbers of files to compare (comma-separated), or 'all':")
    selection = input("> ").strip()
    
    if selection.lower() == 'all':
        selected = all_metrics
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected = [all_metrics[i] for i in indices]
        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
            return
    
    if not selected:
        print("No files selected.")
        return
    
    print(f"\nSelected {len(selected)} files for comparison.")
    
    # Ask for output path
    print("\nEnter output path for plot (or press Enter to skip saving):")
    output_str = input("> ").strip()
    output_path = Path(output_str) if output_str else None
    
    # Plot
    plot_comparison(selected, output_path=output_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Q_Y vs alpha plots from multiple metrics.csv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare specific files
    python compare_metrics.py file1.csv file2.csv file3.csv
    
    # Compare all metrics in a directory
    python compare_metrics.py --dir terao_gamp_gaussian/F_random/results
    
    # Interactive mode
    python compare_metrics.py --interactive
    
    # Custom labels and output
    python compare_metrics.py f1.csv f2.csv --labels "Run A" "Run B" -o comparison.png
"""
    )
    
    parser.add_argument(
        'files', nargs='*', type=Path,
        help='Paths to metrics.csv files to compare'
    )
    parser.add_argument(
        '--dir', '-d', type=Path,
        help='Directory to search for metrics.csv files'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--labels', '-l', nargs='*',
        help='Custom labels for each curve'
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        help='Output path for the plot'
    )
    parser.add_argument(
        '--title', '-t', type=str, default='Q_Y vs Alpha Comparison',
        help='Plot title'
    )
    parser.add_argument(
        '--no-error', action='store_true',
        help='Disable error bars'
    )
    parser.add_argument(
        '--std', action='store_true',
        help='Use std instead of SEM for error bars'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    # First priority: use METRICS_PATHS from top of file if defined
    if METRICS_PATHS and not args.files and not args.dir:
        csv_paths = [Path(p) for p in METRICS_PATHS]
        labels_to_use = CUSTOM_LABELS if CUSTOM_LABELS else None
        output_to_use = Path(OUTPUT_PATH) if OUTPUT_PATH else None
        title_to_use = PLOT_TITLE
    else:
        csv_paths = list(args.files)
        labels_to_use = args.labels
        output_to_use = args.output
        title_to_use = args.title
        
        if args.dir:
            csv_paths.extend(find_metrics_in_dir(args.dir))
    
    if not csv_paths:
        print("No metrics files specified.")
        print("\nOption 1: Edit METRICS_PATHS at the top of this file")
        print("Option 2: Use CLI: python compare_metrics.py file1.csv file2.csv")
        print("Option 3: Use --interactive for guided mode")
        return
    
    # Validate paths
    valid_paths = []
    for p in csv_paths:
        if p.exists():
            valid_paths.append(p)
        else:
            print(f"Warning: File not found: {p}")
    
    if not valid_paths:
        print("No valid files found.")
        return
    
    print(f"Comparing {len(valid_paths)} metrics files...")
    
    plot_comparison(
        valid_paths,
        labels=labels_to_use,
        output_path=output_to_use,
        title=title_to_use,
        show_error_bars=not args.no_error,
        use_sem=not args.std,
    )


if __name__ == "__main__":
    main()
