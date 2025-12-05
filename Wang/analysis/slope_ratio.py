"""
Pre-transition Slope Analysis Tool

Analyzes the slope of Q_Y in the linear growth region before phase transition
for different (N, M) configurations, and fits the relationship between slope and N/M.

Usage:
    1. Modify the configuration parameters below
    2. python slope_ratio.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import datetime


# =============================================================================
# Configuration
# =============================================================================

# JSON results file path (relative to project root)
# Standard data example: "results/size_scaling/fixed_M100_.../multi_size_results_steps10000.json"
# Ortho teacher example: "results/orthogonal_teacher/.../ortho_results_steps5000.json"
JSON_PATH = "results/orthogonal_teacher/1000x100_2000x100_3000x100_4000x100/ortho_results_steps5000.json"

# Alpha range for linear fitting
# Set to None for auto-detection, or specify values manually
ALPHA_START = 1    # e.g., 0.5
ALPHA_END = 1.5    # e.g., 1.5

# Output directory (None = same directory as input file)
OUTPUT_DIR = None

# Analysis mode:
#   "both"           - Analyze both Q_Y and Q_Y_unobserved
#   "Q_Y"            - Analyze Q_Y only
#   "Q_Y_unobserved" - Analyze Q_Y_unobserved only
#   "all"            - Analyze all 4 metrics (including Q_Y_ortho, Q_Y_ortho_unobserved)
#   "ortho"          - Analyze only orthogonal teacher's 2 metrics
ANALYSIS_MODE = "both"

# Whether data is orthogonal teacher comparison (auto-detect)
# Set True/False to force, None for auto-detection
IS_ORTHO_COMPARISON = None


# =============================================================================
# Data Loading & Parsing
# =============================================================================

def load_results(json_path: str) -> dict:
    """Load JSON results file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_config_key(key: str) -> Tuple[int, int]:
    """Parse config key string, e.g., '200x50' -> (200, 50)."""
    parts = key.split('x')
    return int(parts[0]), int(parts[1])


def extract_qy_data(results: dict, config_key: str, alpha_values: List[float],
                    metric: str = "Q_Y") -> np.ndarray:
    """Extract Q_Y or Q_Y_unobserved data for a given configuration.

    Args:
        results: Results dictionary from JSON
        config_key: Configuration key (e.g., '200x50')
        alpha_values: List of alpha values
        metric: Metric name ("Q_Y" or "Q_Y_unobserved")

    Returns:
        Array of metric values for each alpha
    """
    config_results = results[config_key]
    metric_key = f"{metric}_mean"
    qy = []

    for alpha in alpha_values:
        alpha_str = str(float(alpha))
        if alpha_str in config_results:
            # Prefer specified metric, fall back to Q_Y_mean if not available
            val = config_results[alpha_str].get(
                metric_key,
                config_results[alpha_str].get('Q_Y_mean', 0)
            )
            qy.append(val)
        else:
            # Try alternative formats
            for k in config_results.keys():
                if abs(float(k) - alpha) < 1e-6:
                    val = config_results[k].get(
                        metric_key,
                        config_results[k].get('Q_Y_mean', 0)
                    )
                    qy.append(val)
                    break

    return np.array(qy)


def detect_ortho_data(data: dict) -> bool:
    """Detect whether data contains orthogonal teacher metrics."""
    results = data.get('results', {})
    if not results:
        return False
    first_config = list(results.values())[0]
    first_alpha = list(first_config.values())[0]
    return 'Q_Y_ortho_mean' in first_alpha


def get_metric_display_info(metric: str) -> Tuple[str, str, str]:
    """Get display label, title, and file suffix for a metric.

    Returns:
        Tuple of (latex_label, title, file_suffix)
    """
    metric_info = {
        "Q_Y": (r'$Q_Y$', 'Q_Y (Standard)', ''),
        "Q_Y_unobserved": (r'$Q_Y^{unobs}$', 'Q_Y_unobserved (Standard)', '_unobserved'),
        "Q_Y_ortho": (r'$Q_Y^{ortho}$', 'Q_Y (Orthogonal)', '_ortho'),
        "Q_Y_ortho_unobserved": (r'$Q_Y^{ortho,unobs}$', 'Q_Y_unobserved (Orthogonal)', '_ortho_unobserved'),
    }
    return metric_info.get(metric, (metric, metric, f'_{metric}'))


# =============================================================================
# Phase Transition Detection
# =============================================================================

def detect_transition_point(alpha_values: np.ndarray,
                            qy_values: np.ndarray) -> Tuple[float, dict]:
    """Detect phase transition point using gradient peak method.

    Method:
        1. Compute first derivative (gradient) of Q_Y
        2. Find position of maximum gradient = transition point
        3. Return transition region characteristics

    Args:
        alpha_values: Array of alpha values
        qy_values: Array of Q_Y values

    Returns:
        Tuple of (transition_alpha, info_dict)
    """
    # Smoothing with simple moving average
    window = 3
    if len(qy_values) > window * 2:
        qy_smooth = np.convolve(qy_values, np.ones(window)/window, mode='valid')
        alpha_smooth = alpha_values[window//2:-(window//2)]
    else:
        qy_smooth = qy_values
        alpha_smooth = alpha_values

    # Compute gradient
    gradient = np.gradient(qy_smooth, alpha_smooth)

    # Find maximum gradient point = transition point
    max_grad_idx = np.argmax(gradient)
    transition_alpha = alpha_smooth[max_grad_idx]
    max_gradient = gradient[max_grad_idx]

    # Compute transition width (region where gradient > half maximum)
    half_max = max_gradient / 2
    above_half = gradient > half_max

    if np.any(above_half):
        first_idx = np.argmax(above_half)
        last_idx = len(above_half) - 1 - np.argmax(above_half[::-1])
        transition_width = alpha_smooth[last_idx] - alpha_smooth[first_idx]
        transition_start = alpha_smooth[first_idx]
        transition_end = alpha_smooth[last_idx]
    else:
        transition_width = 0.2
        transition_start = transition_alpha - 0.1
        transition_end = transition_alpha + 0.1

    info = {
        'transition_alpha': transition_alpha,
        'max_gradient': max_gradient,
        'transition_width': transition_width,
        'transition_start': transition_start,
        'transition_end': transition_end,
        'gradient_curve': (alpha_smooth, gradient)
    }

    return transition_alpha, info


def detect_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                         min_alpha: float = 0.3) -> Tuple[float, float, dict]:
    """Auto-detect linear growth region before phase transition.

    Strategy:
        1. Find transition point using gradient peak method
        2. Start: First point where Q_Y > 0.02 and alpha >= min_alpha
        3. End: 0.1 alpha units before transition start

    Args:
        alpha_values: Array of alpha values
        qy_values: Array of Q_Y values
        min_alpha: Minimum alpha to consider

    Returns:
        Tuple of (alpha_start, alpha_end, phase_info)
    """
    # Detect transition point
    transition_alpha, phase_info = detect_transition_point(alpha_values, qy_values)

    # Find start: first point where Q_Y is significantly above zero
    start_idx = 0
    for i, qy in enumerate(qy_values):
        if qy > 0.02 and alpha_values[i] >= min_alpha:
            start_idx = i
            break

    # End: 0.1 alpha units before transition start
    end_alpha = phase_info['transition_start'] - 0.1

    # Find corresponding index
    end_idx = len(alpha_values) - 1
    for i, alpha in enumerate(alpha_values):
        if alpha > end_alpha:
            end_idx = max(start_idx + 2, i - 1)  # Ensure at least 3 points
            break

    return alpha_values[start_idx], alpha_values[end_idx], phase_info


# =============================================================================
# Linear Fitting
# =============================================================================

def fit_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                      alpha_start: float, alpha_end: float) -> Tuple[float, float, float, float]:
    """Perform linear fit on specified alpha range.

    Args:
        alpha_values: Array of alpha values
        qy_values: Array of Q_Y values
        alpha_start: Start of fitting range
        alpha_end: End of fitting range

    Returns:
        Tuple of (slope, intercept, R², slope_std_error)
    """
    # Select data in specified range
    mask = (alpha_values >= alpha_start) & (alpha_values <= alpha_end)
    x = alpha_values[mask]
    y = qy_values[mask]

    if len(x) < 3:
        print(f"  Warning: Too few data points ({len(x)}), cannot fit reliably")
        return 0.0, 0.0, 0.0, 0.0

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return slope, intercept, r_value**2, std_err


# =============================================================================
# Slope vs Ratio Analysis
# =============================================================================

def fit_slope_vs_ratio(ratios_NM: np.ndarray, slopes: np.ndarray) -> Dict:
    """Fit relationship between slope and M/N ratio.

    Input ratios_NM is N/M, internally converted to M/N for analysis.

    Models:
        1. Linear through origin: slope = a * (M/N)  [Physically most reasonable]
        2. Linear with intercept: slope = a * (M/N) + b
        3. Power law: slope = a * (M/N)^b
        4. Quadratic: slope = a*(M/N)² + b*(M/N) + c
        5. Quadratic through origin: slope = a*(M/N)² + b*(M/N)
        6. Square root through origin: slope = a*√(M/N)

    Args:
        ratios_NM: Array of N/M ratios
        slopes: Array of slopes

    Returns:
        Dictionary of fitting results for each model
    """
    results = {}
    ratios = 1.0 / ratios_NM  # Convert to M/N

    # Model 1: Linear through origin (most important)
    # Least squares: a = sum(x*y) / sum(x^2)
    try:
        a_origin = np.sum(ratios * slopes) / np.sum(ratios ** 2)
        y_pred = a_origin * ratios
        ss_res = np.sum((slopes - y_pred) ** 2)
        ss_tot = np.sum(slopes ** 2)  # Different R² calculation for origin-constrained
        r2_origin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['linear_origin'] = {
            'a': a_origin, 'R2': r2_origin,
            'formula': f'slope = {a_origin:.6f} * (M/N)'
        }
    except Exception:
        results['linear_origin'] = None

    # Model 2: Linear with intercept
    try:
        slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(ratios, slopes)
        results['linear'] = {
            'a': slope_lin, 'b': intercept_lin, 'R2': r_lin**2,
            'formula': f'slope = {slope_lin:.6f} * (M/N) + {intercept_lin:.6f}'
        }
    except Exception:
        results['linear'] = None

    # Model 3: Power law
    try:
        def power_law(x, a, b):
            return a * np.power(x, b)
        popt, _ = curve_fit(power_law, ratios, slopes, p0=[1.0, 1.0], maxfev=5000)
        y_pred = power_law(ratios, *popt)
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum((slopes - np.mean(slopes))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['power_law'] = {
            'a': popt[0], 'b': popt[1], 'R2': r2,
            'formula': f'slope = {popt[0]:.6f} * (M/N)^{popt[1]:.4f}'
        }
    except Exception:
        results['power_law'] = None

    # Model 4: Quadratic polynomial
    try:
        coeffs = np.polyfit(ratios, slopes, 2)
        y_pred = np.polyval(coeffs, ratios)
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum((slopes - np.mean(slopes))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['quadratic'] = {
            'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2], 'R2': r2,
            'formula': f'slope = {coeffs[0]:.4f}×(M/N)² + {coeffs[1]:.4f}×(M/N) + {coeffs[2]:.4f}'
        }
    except Exception:
        results['quadratic'] = None

    # Model 5: Quadratic through origin
    try:
        # Least squares: [a, b] = (X^T X)^{-1} X^T y
        X = np.vstack([ratios**2, ratios]).T
        coeffs, _, _, _ = np.linalg.lstsq(X, slopes, rcond=None)
        y_pred = coeffs[0]*ratios**2 + coeffs[1]*ratios
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum(slopes**2)  # Use sum(y²) for origin-constrained
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['quadratic_origin'] = {
            'a': coeffs[0], 'b': coeffs[1], 'R2': r2,
            'formula': f'slope = {coeffs[0]:.4f}×(M/N)² + {coeffs[1]:.4f}×(M/N)'
        }
    except Exception:
        results['quadratic_origin'] = None

    # Model 6: Square root through origin
    try:
        sqrt_ratios = np.sqrt(ratios)
        a_sqrt = np.sum(sqrt_ratios * slopes) / np.sum(sqrt_ratios**2)
        y_pred = a_sqrt * sqrt_ratios
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum(slopes**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['sqrt_origin'] = {
            'a': a_sqrt, 'R2': r2,
            'formula': f'slope = {a_sqrt:.4f}×√(M/N)'
        }
    except Exception:
        results['sqrt_origin'] = None

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_analysis(alpha_values: np.ndarray, all_qy: Dict[str, np.ndarray],
                  linear_fits: Dict[str, Dict], slope_analysis: Dict,
                  save_dir: Path, metric: str = "Q_Y"):
    """Generate analysis plots (2 figures).

    Figure 1: Q_Y curves with linear fits
    Figure 2: Slope vs M/N with multiple model fits
    """
    metric_label, metric_title, file_suffix = get_metric_display_info(metric)

    # Color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999']

    # -------------------------------------------------------------------------
    # Figure 1: Q_Y Curves with Linear Fits
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    for i, (config_key, qy) in enumerate(all_qy.items()):
        N, M = parse_config_key(config_key)
        ratio = N / M
        color = colors[i % len(colors)]

        # Original data
        ax1.plot(alpha_values, qy, 'o-', color=color, markersize=4,
                 linewidth=1.5, label=f'N={N}, M={M} (N/M={ratio:.0f})')

        # Linear fit line
        if config_key in linear_fits:
            fit = linear_fits[config_key]
            x_fit = np.linspace(fit['alpha_start'], fit['alpha_end'], 50)
            y_fit = fit['slope'] * x_fit + fit['intercept']
            ax1.plot(x_fit, y_fit, '--', color=color, linewidth=2, alpha=0.7)

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel(metric_label, fontsize=14)
    ax1.set_title(f'{metric_title} vs Alpha (with Linear Fits)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    save_path1 = save_dir / f'qy{file_suffix}_with_linear_fits.png'
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"  Curve plot saved: {save_path1}")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Slope vs M/N
    # -------------------------------------------------------------------------
    ratios_MN = []  # M/N
    slopes_list = []
    slope_errs = []
    labels = []

    for config_key, fit in linear_fits.items():
        N, M = parse_config_key(config_key)
        ratios_MN.append(M / N)
        slopes_list.append(fit['slope'])
        slope_errs.append(fit['slope_err'])
        labels.append(f'N={N}')

    ratios_MN = np.array(ratios_MN)
    slopes_arr = np.array(slopes_list)
    slope_errs = np.array(slope_errs)

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Data points
    ax2.errorbar(ratios_MN, slopes_arr, yerr=slope_errs, fmt='o', markersize=10,
                 capsize=5, color='#2563eb', label='Measured slopes')

    # Add labels
    for r, s, lbl in zip(ratios_MN, slopes_arr, labels):
        ax2.annotate(lbl, (r, s), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Fitted curves (show all models)
    x_fit = np.linspace(0.001, max(ratios_MN) * 1.1, 100)
    fit_styles = [
        ('linear_origin', '#dc2626', '-', 2.5),       # Red solid
        ('quadratic_origin', '#16a34a', '--', 2.0),   # Green dashed
        ('power_law', '#9333ea', '-.', 2.0),          # Purple dash-dot
        ('sqrt_origin', '#ea580c', ':', 2.0),         # Orange dotted
    ]

    for model_key, color, linestyle, lw in fit_styles:
        if slope_analysis and slope_analysis.get(model_key):
            m = slope_analysis[model_key]
            if model_key == 'linear_origin':
                y_fit = m['a'] * x_fit
            elif model_key == 'quadratic_origin':
                y_fit = m['a'] * x_fit**2 + m['b'] * x_fit
            elif model_key == 'power_law':
                y_fit = m['a'] * np.power(x_fit, m['b'])
            elif model_key == 'sqrt_origin':
                y_fit = m['a'] * np.sqrt(x_fit)
            else:
                continue
            ax2.plot(x_fit, y_fit, linestyle=linestyle, color=color, linewidth=lw,
                     label=f"{m['formula']}, R²={m['R2']:.4f}")

    ax2.set_xlabel('M/N', fontsize=14)
    ax2.set_ylabel(f'Slope (d{metric_title}/dα)', fontsize=14)
    ax2.set_title(f'Pre-Transition Slope vs M/N ({metric_title})', fontsize=14, fontweight='bold')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    save_path2 = save_dir / f'slope_vs_ratio{file_suffix}.png'
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"  Slope plot saved: {save_path2}")
    plt.close(fig2)


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(configs: list, linear_fits: dict, slope_analysis: dict,
                    best_model: str, metric: str = "Q_Y") -> str:
    """Generate natural language analysis report.

    Args:
        configs: List of (config_key, N, M, ratio) tuples
        linear_fits: Dictionary of linear fit results
        slope_analysis: Dictionary of slope vs ratio analysis results
        best_model: Key of best fitting model
        metric: Metric name

    Returns:
        Markdown report string
    """
    _, metric_title, _ = get_metric_display_info(metric)
    is_ortho = "ortho" in metric.lower()

    lines = []
    lines.append(f"# Pre-Transition Slope Analysis Report - {metric_title}\n")

    # Data overview
    lines.append("## Data Overview\n")
    lines.append(f"This analysis includes **{len(configs)}** different (N, M) configurations:\n")
    for config_key, N, M, ratio in configs:
        fit = linear_fits[config_key]
        lines.append(f"- N={N}, M={M} (N/M={ratio:.1f}): slope = {fit['slope']:.4f}, "
                     f"transition point α_c ≈ {fit['transition_alpha']:.2f}")
    lines.append("")

    # Key findings
    lines.append("## Key Findings\n")

    # Model comparison table
    lines.append("### Fitting Model Comparison\n")
    lines.append("| Model | Formula | R² |")
    lines.append("|-------|---------|-----|")

    model_order = ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin', 'linear', 'quadratic']
    model_names = {
        'linear_origin': 'Linear (origin)',
        'quadratic_origin': 'Quadratic (origin)',
        'power_law': 'Power law',
        'sqrt_origin': 'Square root',
        'linear': 'Linear (intercept)',
        'quadratic': 'Quadratic',
    }

    best_r2 = 0
    best_model_key = None
    for model_key in model_order:
        if slope_analysis.get(model_key):
            m = slope_analysis[model_key]
            lines.append(f"| {model_names[model_key]} | `{m['formula']}` | {m['R2']:.4f} |")
            if m['R2'] > best_r2:
                best_r2 = m['R2']
                best_model_key = model_key

    lines.append("")
    if best_model_key:
        lines.append(f"**Best fitting model**: {model_names[best_model_key]} (R² = {best_r2:.4f})\n")

    # Analyze linear through origin fit
    if slope_analysis.get('linear_origin'):
        lo = slope_analysis['linear_origin']
        lines.append("### Linear Fit Through Origin\n")
        lines.append(f"**Formula**: `slope = {lo['a']:.4f} × (M/N)`\n")
        lines.append(f"**R² = {lo['R2']:.4f}**\n")

        if lo['R2'] > 0.95:
            lines.append("Excellent fit. Data strongly supports the hypothesis that "
                         "**slope is proportional to M/N**.")
        elif lo['R2'] > 0.85:
            lines.append("Good fit. Data generally supports the hypothesis that slope is proportional to M/N.")
        else:
            lines.append("Moderate fit. Other factors may be involved.")
        lines.append("")

    # Physical interpretation
    lines.append("## Physical Interpretation\n")

    if is_ortho:
        lines.append("**Orthogonal teacher model** enforces W^T W = I_M via QR decomposition, "
                     "eliminating finite-size fluctuations.\n")
        lines.append("- Standard teacher's Gram matrix G = (M/N) W^T W = I + Δ, "
                     "where Δ is O(1/√N) random fluctuation")
        lines.append("- Orthogonal teacher forces Δ = 0, so linear offset in low-α region should be smaller")
        lines.append("")
        lines.append("If orthogonal teacher's slope coefficient is significantly smaller than standard teacher, "
                     "it indicates that **finite-size fluctuations are the main cause of elevated Q_Y in low-α region**.")
    else:
        lines.append("The slope in the pre-transition linear growth region reflects learning efficiency "
                     "at low observation density:\n")
        lines.append("- **Slope ∝ M/N** means: when hidden dimension M is larger relative to observation "
                     "dimension N, Q_Y grows faster per unit increase in observation density α")
        lines.append("- This is intuitive: larger M/N means higher information redundancy, "
                     "making it easier to recover the signal from partial observations")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion\n")

    if slope_analysis.get('linear_origin') and slope_analysis['linear_origin']['R2'] > 0.9:
        a = slope_analysis['linear_origin']['a']
        metric_short = "Q_Y" if "unobs" not in metric else "Q_Y_unobs"
        lines.append("Data supports the following empirical formula:\n")
        lines.append("```")
        lines.append(f"d{metric_short}/dα ≈ {a:.2f} × (M/N)    (pre-transition linear region)")
        lines.append("```\n")
        lines.append(f"That is, {metric_title} growth rate before transition is proportional to M/N, "
                     f"with coefficient approximately **{a:.2f}**.")
    else:
        lines.append(f"Best fitting model is **{best_model}**, but more data points may be needed "
                     "to confirm the relationship.")

    return "\n".join(lines)


def generate_summary_report(all_results: dict, output_dir: Path, is_ortho: bool) -> str:
    """Generate comprehensive comparison report across all metrics.

    Args:
        all_results: Dictionary of analysis results for each metric
        output_dir: Output directory
        is_ortho: Whether data contains orthogonal teacher metrics

    Returns:
        Markdown report string
    """
    lines = []
    lines.append("# Slope Analysis Summary Report\n")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary table
    lines.append("## Slope Coefficient Summary\n")
    lines.append("| Metric | Coefficient a | R² | Physical Meaning |")
    lines.append("|--------|---------------|-----|------------------|")

    for metric, res in all_results.items():
        lo = res.get('slope_vs_ratio', {}).get('linear_origin', {})
        a = lo.get('a', 0)
        r2 = lo.get('R2', 0)
        _, title, _ = get_metric_display_info(metric)

        if "ortho" in metric:
            meaning = "Orthogonal teacher (no finite-size fluctuation)"
        elif "unobs" in metric:
            meaning = "Unobserved positions (generalization)"
        else:
            meaning = "All positions (including training data)"

        lines.append(f"| {title} | {a:.4f} | {r2:.4f} | {meaning} |")

    lines.append("")

    # Multi-model comparison (using first metric as example)
    first_metric = list(all_results.keys())[0] if all_results else None
    if first_metric:
        slope_vs_ratio = all_results[first_metric].get('slope_vs_ratio', {})
        if slope_vs_ratio:
            lines.append("## Fitting Model Comparison (First Metric)\n")
            lines.append("| Model | Formula | R² |")
            lines.append("|-------|---------|-----|")

            model_order = ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin']
            model_names = {
                'linear_origin': 'Linear (origin)',
                'quadratic_origin': 'Quadratic (origin)',
                'power_law': 'Power law',
                'sqrt_origin': 'Square root',
            }

            for model_key in model_order:
                if slope_vs_ratio.get(model_key):
                    m = slope_vs_ratio[model_key]
                    lines.append(f"| {model_names[model_key]} | `{m['formula']}` | {m['R2']:.4f} |")

            lines.append("")

    # Comparison analysis
    if is_ortho:
        lines.append("## Standard vs Orthogonal Teacher Comparison\n")

        std_qy = all_results.get('Q_Y', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        ortho_qy = all_results.get('Q_Y_ortho', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if std_qy > 0 and ortho_qy > 0:
            reduction = (std_qy - ortho_qy) / std_qy * 100
            lines.append(f"- Standard teacher Q_Y slope coefficient: **{std_qy:.4f}**")
            lines.append(f"- Orthogonal teacher Q_Y slope coefficient: **{ortho_qy:.4f}**")
            lines.append(f"- Slope reduction: **{reduction:.1f}%**")
            lines.append("")

            if reduction > 50:
                lines.append("**Conclusion**: Orthogonal teacher significantly reduces pre-transition slope, "
                             "indicating that **finite-size fluctuations are the main cause of elevated Q_Y "
                             "in low-α region**.")
            elif reduction > 20:
                lines.append("**Conclusion**: Orthogonal teacher reduces pre-transition slope. "
                             "Finite-size fluctuations have some effect on low-α region.")
            else:
                lines.append("**Conclusion**: Orthogonal teacher has minimal effect on slope. "
                             "Q_Y offset in low-α region may have other causes.")
        lines.append("")

        # Unobserved comparison
        std_unobs = all_results.get('Q_Y_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        ortho_unobs = all_results.get('Q_Y_ortho_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if std_unobs > 0 and ortho_unobs > 0:
            lines.append("### Generalization (Unobserved) Comparison\n")
            reduction_unobs = (std_unobs - ortho_unobs) / std_unobs * 100
            lines.append(f"- Standard teacher Q_Y_unobserved slope coefficient: **{std_unobs:.4f}**")
            lines.append(f"- Orthogonal teacher Q_Y_ortho_unobserved slope coefficient: **{ortho_unobs:.4f}**")
            lines.append(f"- Slope reduction: **{reduction_unobs:.1f}%**")
    else:
        # Non-orthogonal data comparison
        qy = all_results.get('Q_Y', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        qy_unobs = all_results.get('Q_Y_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if qy > 0 and qy_unobs > 0:
            lines.append("## Q_Y vs Q_Y_unobserved Comparison\n")
            ratio = qy_unobs / qy
            lines.append(f"- Q_Y slope coefficient: **{qy:.4f}**")
            lines.append(f"- Q_Y_unobserved slope coefficient: **{qy_unobs:.4f}**")
            lines.append(f"- Ratio: **{ratio:.4f}**")

    return "\n".join(lines)


# =============================================================================
# Single Metric Analysis
# =============================================================================

def run_single_analysis(data: dict, configs: list, alpha_values: np.ndarray,
                        output_dir: Path, metric: str = "Q_Y") -> dict:
    """Run complete analysis for a single metric.

    Args:
        data: Loaded JSON data
        configs: List of (config_key, N, M, ratio) tuples
        alpha_values: Array of alpha values
        output_dir: Output directory
        metric: Metric name ("Q_Y", "Q_Y_unobserved", "Q_Y_ortho", or "Q_Y_ortho_unobserved")

    Returns:
        Analysis results dictionary
    """
    _, metric_title, file_suffix = get_metric_display_info(metric)
    results = data['results']

    print(f"\n{'─' * 60}")
    print(f"  Analyzing metric: {metric_title}")
    print(f"{'─' * 60}")

    # Store all data and fit results
    all_qy = {}
    linear_fits = {}

    for config_key, N, M, ratio in configs:
        print(f"\n[N={N}, M={M}, N/M={ratio:.1f}]")

        # Extract data
        qy = extract_qy_data(results, config_key, alpha_values, metric=metric)
        all_qy[config_key] = qy

        # Detect linear region and transition point
        if ALPHA_START is not None and ALPHA_END is not None:
            alpha_start, alpha_end = ALPHA_START, ALPHA_END
            _, phase_info = detect_transition_point(alpha_values, qy)
            print(f"  Using specified linear region: [{alpha_start:.2f}, {alpha_end:.2f}]")
        else:
            alpha_start, alpha_end, phase_info = detect_linear_region(alpha_values, qy)
            print(f"  Auto-detected linear region: [{alpha_start:.2f}, {alpha_end:.2f}]")

        # Print transition point info
        print(f"  Detected transition point: α_c = {phase_info['transition_alpha']:.3f}")
        print(f"  Transition region: [{phase_info['transition_start']:.2f}, {phase_info['transition_end']:.2f}]")
        print(f"  Maximum gradient: {phase_info['max_gradient']:.4f}")

        # Linear fitting
        slope, intercept, r2, slope_err = fit_linear_region(
            alpha_values, qy, alpha_start, alpha_end
        )

        print(f"  Slope: {slope:.6f} ± {slope_err:.6f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  R²: {r2:.6f}")

        linear_fits[config_key] = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'slope_err': slope_err,
            'alpha_start': alpha_start,
            'alpha_end': alpha_end,
            'transition_alpha': phase_info['transition_alpha'],
            'transition_start': phase_info['transition_start'],
            'transition_end': phase_info['transition_end'],
            'max_gradient': phase_info['max_gradient']
        }

    # Analyze slope vs N/M relationship
    print(f"\n  Slope vs M/N relationship fitting:")

    ratios = np.array([c[3] for c in configs])
    slopes = np.array([linear_fits[c[0]]['slope'] for c in configs])

    slope_analysis = fit_slope_vs_ratio(ratios, slopes)

    # Display linear through origin result
    if slope_analysis.get('linear_origin'):
        result = slope_analysis['linear_origin']
        print(f"    slope = {result['a']:.4f} × (M/N), R² = {result['R2']:.4f}")

    # Find best model
    best_model = 'linear_origin'
    best_r2 = slope_analysis.get('linear_origin', {}).get('R2', 0)

    # Generate plots
    plot_analysis(alpha_values, all_qy, linear_fits, slope_analysis, output_dir, metric=metric)

    # Save analysis results
    linear_fits_clean = {}
    for k, v in linear_fits.items():
        linear_fits_clean[k] = {kk: vv for kk, vv in v.items() if kk != 'gradient_curve'}

    analysis_results = {
        'metric': metric,
        'configs': [{'key': c[0], 'N': c[1], 'M': c[2], 'ratio': c[3]} for c in configs],
        'linear_fits': linear_fits_clean,
        'slope_vs_ratio': {k: v for k, v in slope_analysis.items() if v is not None},
        'best_model': best_model
    }

    analysis_path = output_dir / f'slope_analysis_results{file_suffix}.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nAnalysis results saved: {analysis_path}")

    # Generate natural language report
    report = generate_report(configs, linear_fits, slope_analysis, best_model, metric=metric)
    report_path = output_dir / f'analysis_report{file_suffix}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Analysis report saved: {report_path}")

    return analysis_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for slope analysis."""
    json_path = Path(JSON_PATH)

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        print("Please check JSON_PATH configuration")
        return

    print("=" * 70)
    print("  Pre-Transition Slope Analysis")
    print("=" * 70)
    print(f"Input file: {json_path.name}")

    data = load_results(json_path)

    # Auto-detect orthogonal teacher data
    is_ortho = IS_ORTHO_COMPARISON if IS_ORTHO_COMPARISON is not None else detect_ortho_data(data)
    print(f"Data type: {'Orthogonal teacher comparison' if is_ortho else 'Standard data'}")

    # Extract configurations
    alpha_values = np.array(data['alpha_values'])
    results = data['results']
    config_keys = list(results.keys())

    print(f"Alpha range: {alpha_values[0]:.2f} - {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    print(f"Number of configurations: {len(config_keys)}")

    # Parse all configurations
    configs = []
    for key in config_keys:
        N, M = parse_config_key(key)
        configs.append((key, N, M, N/M))

    # Sort by N/M
    configs.sort(key=lambda x: x[3])

    for config_key, N, M, ratio in configs:
        print(f"  - N={N}, M={M} (N/M={ratio:.0f})")

    # Output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = json_path.parent / 'slope_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which metrics to analyze based on ANALYSIS_MODE
    if ANALYSIS_MODE == "all" and is_ortho:
        metrics_to_analyze = ["Q_Y", "Q_Y_unobserved", "Q_Y_ortho", "Q_Y_ortho_unobserved"]
    elif ANALYSIS_MODE == "ortho" and is_ortho:
        metrics_to_analyze = ["Q_Y_ortho", "Q_Y_ortho_unobserved"]
    elif ANALYSIS_MODE == "both":
        if is_ortho:
            metrics_to_analyze = ["Q_Y", "Q_Y_unobserved", "Q_Y_ortho", "Q_Y_ortho_unobserved"]
        else:
            metrics_to_analyze = ["Q_Y", "Q_Y_unobserved"]
    elif ANALYSIS_MODE == "Q_Y":
        metrics_to_analyze = ["Q_Y"]
    elif ANALYSIS_MODE == "Q_Y_unobserved":
        metrics_to_analyze = ["Q_Y_unobserved"]
    else:
        print(f"Unknown analysis mode: {ANALYSIS_MODE}")
        return

    print(f"Metrics to analyze: {', '.join(metrics_to_analyze)}")

    all_results = {}
    for metric in metrics_to_analyze:
        all_results[metric] = run_single_analysis(data, configs, alpha_values, output_dir, metric=metric)

    # Generate summary report
    print(f"\n{'=' * 70}")
    print("  Generating Summary Report")
    print(f"{'=' * 70}")

    summary = generate_summary_report(all_results, output_dir, is_ortho)
    summary_path = output_dir / 'summary_report.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary report saved: {summary_path}")

    # Print summary
    print("\n" + "─" * 60)
    print("  Slope Coefficients (Linear through origin: slope = a × M/N)")
    print("─" * 60)
    for metric, res in all_results.items():
        lo = res.get('slope_vs_ratio', {}).get('linear_origin', {})
        _, title, _ = get_metric_display_info(metric)
        print(f"  {title:30s}: a = {lo.get('a', 0):.4f}, R² = {lo.get('R2', 0):.4f}")

    # Print multi-model comparison (first metric)
    first_metric = list(all_results.keys())[0] if all_results else None
    if first_metric:
        slope_vs_ratio = all_results[first_metric].get('slope_vs_ratio', {})
        if slope_vs_ratio:
            print("\n" + "─" * 60)
            print(f"  Multi-Model Comparison ({first_metric})")
            print("─" * 60)
            model_names = {
                'linear_origin': 'Linear (origin)',
                'quadratic_origin': 'Quadratic (origin)',
                'power_law': 'Power law',
                'sqrt_origin': 'Square root',
            }
            for model_key in ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin']:
                if slope_vs_ratio.get(model_key):
                    m = slope_vs_ratio[model_key]
                    print(f"  {model_names[model_key]:18s}: R² = {m['R2']:.4f}  {m['formula']}")

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete! Results directory: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
