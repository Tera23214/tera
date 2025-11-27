"""
BiG-AMP Multi-Size Comparison

Runs BiG-AMP training for multiple (N, M) configurations and plots
results on the same figure for comparison.

Features:
1. Accept multiple (N, M) pairs (assumes N1=N2=N)
2. Plot Q_Y from different sizes on one figure
3. Plot Q_W', Q_X' from different sizes on another figure
4. Automatic color assignment for different configurations

Usage:
    # Default configurations
    python bigamp_multi_size.py

    # Custom configurations (comma-separated N:M pairs)
    python bigamp_multi_size.py --sizes "200:50,400:100,800:200"

    # With custom steps
    python bigamp_multi_size.py --sizes "200:50,400:100" --steps 500
"""

from pathlib import Path
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# Default Parameters
# ============================================================
# Default (N, M) configurations to compare
DEFAULT_SIZES = [
    (500, 50),
    (1000, 50),
    (1500, 50),
    (2000,50),
]

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 3
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 2000

SAMPLES_PER_ALPHA = 10
RESAMPLE_MASK_EACH_TRIAL = True  # True: each trial gets different mask, False: all trials share same mask
SEED = 42

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Precision settings
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Device Info
# ============================================================
@dataclass
class DeviceInfo:
    device_type: str
    available_memory_gb: float
    device_name: str


def get_device_info() -> DeviceInfo:
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_type='cuda',
            available_memory_gb=props.total_memory / (1024**3),
            device_name=props.name
        )
    elif DEVICE.type == 'mps':
        return DeviceInfo(
            device_type='mps',
            available_memory_gb=32.0,
            device_name='Apple Silicon'
        )
    else:
        return DeviceInfo(
            device_type='cpu',
            available_memory_gb=64.0,
            device_name='CPU'
        )


DEVICE_INFO = get_device_info()


# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_teacher(N1, N2, M, device, seed=42):
    """Create teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def sample_mask(N1, N2, M, alpha, device, seed=None):
    """Generate random observation mask"""
    if seed is not None:
        torch.manual_seed(seed)
    c = alpha * M
    prob = min(c / N2, 1.0)
    mask = (torch.rand((N1, N2), device=device) < prob).float()
    return mask, c


# ============================================================
# Evaluation Metrics
# ============================================================
@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    """Compute Gram matrix overlap using cosine similarity"""
    if use_left:
        G_A = A @ A.T
        G_B = B @ B.T
    else:
        G_A = A.T @ A
        G_B = B.T @ B

    G_A_flat = G_A.flatten()
    G_B_flat = G_B.flatten()

    dot = (G_A_flat * G_B_flat).sum()
    norm_A = G_A_flat.norm()
    norm_B = G_B_flat.norm()

    return float(dot / (norm_A * norm_B + 1e-12))


@torch.no_grad()
def gram_overlap_zero_to_one(A, B, use_left=True):
    """
    Compute normalized Gram overlap in [0, 1] range with baseline correction.
    """
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)
    return float(max(0.0, min(1.0, qc)))


# ============================================================
# Memory Management
# ============================================================
def estimate_memory_per_alpha(N1, N2, M, S, dtype_bytes=4):
    """Estimate GPU memory needed per alpha value"""
    student_params = 2 * (S * N1 * M + S * M * N2)
    intermediate = 16 * S * N1 * N2
    total_elements = student_params + intermediate
    return total_elements * dtype_bytes / (1024**3)


def calculate_smart_parallelism(N1, N2, M, S, num_alphas):
    """Calculate optimal parallelism based on memory"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    usable_mem = available * 0.85 - teacher_mem

    if mem_per_batch_alpha <= 0:
        return num_alphas

    max_parallel = max(1, min(int(usable_mem / mem_per_batch_alpha), num_alphas))
    return max_parallel


# ============================================================
# BiG-AMP Training
# ============================================================
def train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, alpha_values, steps, S, M,
                          damping=0.5, noise_var=1e-6):
    """BiG-AMP training with parallel alpha processing"""
    device = Wt.device
    N1 = Wt.shape[0]
    N2 = Xt.shape[1]
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    w_hat = torch.randn((num_alphas, S, N1, M), device=device) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_teacher_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for step in tqdm(range(steps), desc="BiG-AMP Training", leave=False, mininterval=1.0):
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher_exp - z_hat) * A_all
        s = residual / V

        tau_W = (alpha_scale ** 2) * torch.matmul(A_all / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher_exp - z_hat2) * A_all
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A_all / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


def train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, M, seed,
                        damping=0.5, noise_var=1e-6, resample_mask=False):
    """Memory-optimized BiG-AMP training for single alpha"""
    device = Wt.device
    N1 = Wt.shape[0]
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    if resample_mask and S > 1:
        # Generate S different masks (S, N1, N2)
        A = torch.zeros((S, N1, N2), device=device)
        for s in range(S):
            mask_seed = seed + s * 10000
            A_s, _ = sample_mask(N1, N2, M, alpha, device, seed=mask_seed)
            A[s] = A_s
    else:
        # Generate one mask, broadcast to all trials (1, N1, N2)
        A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
        A = A.unsqueeze(0)

    torch.manual_seed(seed + 10000)
    w_hat = torch.randn((S, N1, M), device=device) * scale
    x_hat = torch.randn((S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher - z_hat) * A
        s = residual / V

        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher - z_hat2) * A
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, alpha_values, S):
    """Evaluate metrics for all alphas"""
    results = {}

    for a_idx, alpha in enumerate(alpha_values):
        trial_results = []

        for s in range(S):
            W_s = W[a_idx, s] if W.dim() == 4 else W[s]
            X_s = X[a_idx, s] if X.dim() == 4 else X[s]

            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

            Yp = W_s @ X_s
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))

            trial_results.append({
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime, 'Q_Y': Q_Y
            })

        metrics = {}
        for key in trial_results[0].keys():
            vals = [r[key] for r in trial_results]
            metrics[f'{key}_mean'] = float(np.mean(vals))
            metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        results[float(alpha)] = metrics

    return results


@torch.no_grad()
def evaluate_single(W, X, Wt, Xt, Y_teacher, S):
    """Evaluate single alpha result"""
    trial_results = []

    for s in range(S):
        W_s, X_s = W[s], X[s]
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                   (Y_teacher.norm() * Yp.norm() + 1e-12))

        trial_results.append({
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime, 'Q_Y': Q_Y
        })

    metrics = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return metrics


# ============================================================
# Training for Single Configuration
# ============================================================
def run_single_config(N, M, alpha_values, steps, S, damping, noise_var, seed):
    """Run training for a single (N, M) configuration"""
    N1 = N2 = N
    set_seed(seed)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=seed)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    max_parallel = calculate_smart_parallelism(N1, N2, M, S, num_alphas)

    print(f"\n  Training N={N}, M={M}")
    print(f"  Parallelism: {max_parallel} alphas")

    all_results = {}

    if max_parallel >= 2:
        # Parallel mode
        for batch_start in range(0, num_alphas, max_parallel):
            batch_end = min(batch_start + max_parallel, num_alphas)
            batch_alphas = alpha_values[batch_start:batch_end]
            batch_size = len(batch_alphas)

            if RESAMPLE_MASK_EACH_TRIAL:
                # Generate S different masks for each alpha (num_alphas, S, N1, N2)
                A_all = torch.zeros((batch_size, S, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    for s in range(S):
                        mask_seed = seed + int(alpha * 1000) + s * 10000
                        A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                        A_all[i, s] = A
            else:
                # Generate one mask per alpha, broadcast to all trials (num_alphas, 1, N1, N2)
                A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    mask_seed = seed + int(alpha * 1000)
                    A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                    A_all[i, 0] = A

            W, X = train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, batch_alphas, steps, S, M,
                                          damping=damping, noise_var=noise_var)
            batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, S)
            all_results.update(batch_results)

            del A_all, W, X
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # Sequential mode
        for alpha in alpha_values:
            alpha_seed = seed + int(alpha * 1000)
            W, X = train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, M, alpha_seed,
                                        damping=damping, noise_var=noise_var,
                                        resample_mask=RESAMPLE_MASK_EACH_TRIAL)
            metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, S)
            all_results[float(alpha)] = metrics

            del W, X
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    return all_results


# ============================================================
# Visualization
# ============================================================
def get_color_palette(n_colors):
    """Get a list of distinct colors for plotting"""
    # High-contrast color palette
    colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
    ]
    if n_colors <= len(colors):
        return colors[:n_colors]
    # If need more colors, cycle through
    return [colors[i % len(colors)] for i in range(n_colors)]


def plot_qy_comparison(all_results, sizes, alpha_values, save_path, steps):
    """
    Plot Q_Y comparison across different (N, M) configurations
    """
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_mu = np.array([results[a]['Q_Y_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_mu, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Q_Y Comparison Across Different Matrix Sizes\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qwx_comparison(all_results, sizes, alpha_values, save_path, steps):
    """
    Plot Q_W' and Q_X' comparison across different (N, M) configurations
    """
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Q_W' comparison
    ax1 = axes[0]
    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qW_mu = np.array([results[a]['Q_W_prime_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax1.plot(aL, qW_mu, marker='o', linewidth=2, markersize=4,
                 color=colors[i], label=label)

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel(r"$Q_W'$", fontsize=14)
    ax1.set_title(r"$Q_W'$ Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=10, loc='lower right')

    # Right: Q_X' comparison
    ax2 = axes[1]
    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qX_mu = np.array([results[a]['Q_X_prime_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax2.plot(aL, qX_mu, marker='s', linewidth=2, markersize=4,
                 color=colors[i], label=label)

    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax2.set_ylabel(r"$Q_X'$", fontsize=14)
    ax2.set_title(r"$Q_X'$ Comparison", fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=10, loc='lower right')

    plt.suptitle(f"Gram Overlap Comparison Across Different Matrix Sizes\n(BiG-AMP, {steps} steps)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_W'/Q_X' comparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_sizes(sizes_str: str) -> List[Tuple[int, int]]:
    """Parse sizes string like '200:50,400:100' into list of (N, M) tuples"""
    sizes = []
    for pair in sizes_str.split(','):
        pair = pair.strip()
        if ':' in pair:
            n, m = pair.split(':')
            sizes.append((int(n.strip()), int(m.strip())))
        else:
            raise ValueError(f"Invalid size format: {pair}. Expected 'N:M'")
    return sizes


def main():
    parser = argparse.ArgumentParser(description='BiG-AMP Multi-Size Comparison')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated N:M pairs, e.g., "200:50,400:100,800:200"')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Samples per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
    parser.add_argument('--damping', type=float, default=DAMPING, help='BiG-AMP damping factor')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Parse sizes
    if args.sizes:
        sizes = parse_sizes(args.sizes)
    else:
        sizes = DEFAULT_SIZES

    # Create alpha values
    alpha_values = list(np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, args.alpha_step))

    # Output directory - use Result_compareNM for multi-size comparison results
    if args.output_dir:
        result_dir = Path(args.output_dir)
    else:
        # Use sizes string for directory name
        sizes_str = "_".join([f"{n}x{m}" for n, m in sizes])
        result_dir = Path(__file__).parent / "Result_compareNM" / sizes_str
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BiG-AMP MULTI-SIZE COMPARISON")
    print("=" * 70)
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Configurations: {sizes}")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    print(f"Steps: {args.steps}, Samples: {args.samples}")
    print(f"Resample mask each trial: {RESAMPLE_MASK_EACH_TRIAL}")
    print(f"Output: {result_dir}")
    print("=" * 70)

    # Run training for each configuration
    all_results = {}
    total_start = time.time()

    for i, (N, M) in enumerate(sizes):
        print(f"\n[{i+1}/{len(sizes)}] Configuration: N={N}, M={M}")
        config_start = time.time()

        results = run_single_config(N, M, alpha_values, args.steps, args.samples,
                                     args.damping, NOISE_VAR, SEED)
        all_results[(N, M)] = results

        config_time = time.time() - config_start
        print(f"  Completed in {config_time:.1f}s")

    total_time = time.time() - total_start

    # Save results
    results_data = {
        'config': {
            'sizes': [[n, m] for n, m in sizes],
            'steps': args.steps,
            'samples_per_alpha': args.samples,
            'resample_mask_each_trial': RESAMPLE_MASK_EACH_TRIAL,
            'damping': args.damping,
            'noise_var': NOISE_VAR,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {f"{n}x{m}": {str(k): v for k, v in results.items()}
                    for (n, m), results in all_results.items()}
    }

    results_path = result_dir / f'multi_size_results_steps{args.steps}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot 1: Q_Y comparison
    plot_path1 = result_dir / f'multi_size_qy_steps{args.steps}.png'
    plot_qy_comparison(all_results, sizes, alpha_values, plot_path1, args.steps)

    # Plot 2: Q_W' and Q_X' comparison
    plot_path2 = result_dir / f'multi_size_qwx_steps{args.steps}.png'
    plot_qwx_comparison(all_results, sizes, alpha_values, plot_path2, args.steps)

    # Summary
    print(f"\n{'='*70}")
    print("MULTI-SIZE COMPARISON COMPLETED")
    print(f"{'='*70}")
    print(f"Configurations tested: {len(sizes)}")
    for N, M in sizes:
        print(f"  - N={N}, M={M}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Plot 1 (Q_Y): {plot_path1}")
    print(f"Plot 2 (Q_W'/Q_X'): {plot_path2}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
