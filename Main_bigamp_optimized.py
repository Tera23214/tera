"""
BiG-AMP Optimized - Final Production Version

Features:
1. BiG-AMP (Bilinear Generalized Approximate Message Passing) algorithm
2. Smart memory control with 3 modes:
   - parallel: Standard batched processing (N <= 10000)
   - optimized: No mask pre-storage, sequential alpha (N <= 20000)
   - extreme: FP16 storage + chunked computation (N <= 25000)
3. Intelligent parallelism based on available GPU memory
4. Complete evaluation metrics (Q_Y, Q_W, Q_X, Q_W', Q_X', Gen_Error)

Usage:
    python Main_bigamp_optimized.py                          # Default settings
    python Main_bigamp_optimized.py --n1 20000 --m 141       # Large matrix
    python Main_bigamp_optimized.py --memory-mode extreme    # Force extreme mode
    python Main_bigamp_optimized.py --steps 500              # Custom step count
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
from typing import Optional, Literal

# ============================================================
# Default Parameters
# ============================================================
N1 = 10000
N2 = 10000
M = 100

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 2
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-6
MAX_STEPS = 200

SAMPLES_PER_ALPHA = 5
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

# Result directory
RESULT_DIR = Path(__file__).parent / "Result" / f"{N1}_{N2}_{M}"


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
            available_memory_gb=32.0,  # Approximate for Apple Silicon
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

    # Expected number of observations per row/column
    c = alpha * M

    # Generate mask with probability c/N2 for each entry
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
    """Compute normalized Gram overlap in [0, 1] range"""
    cosine = gram_overlap_cosine(A, B, use_left)
    return (cosine + 1.0) / 2.0


# ============================================================
# Memory Management
# ============================================================
def estimate_memory_per_alpha(N1, N2, M, S, dtype_bytes=4):
    """Estimate GPU memory needed per alpha value for BiG-AMP"""
    # BiG-AMP needs more memory than AGD due to variance tensors
    # w_hat, x_hat: S × N1 × M + S × M × N2
    # w_var, x_var: S × N1 × M + S × M × N2
    # Intermediate: Y_student (S × N1 × N2), residual, etc.

    student_params = 2 * (S * N1 * M + S * M * N2)  # w_hat, x_hat, w_var, x_var
    intermediate = 4 * S * N1 * N2  # Y, residual, s, V
    gradients = 2 * S * N1 * N2  # tau_W, tau_X computations

    total_elements = student_params + intermediate + gradients
    return total_elements * dtype_bytes / (1024**3)


def select_memory_mode(N1, N2, M, S, num_alphas, mode_override='auto'):
    """Select optimal memory mode based on matrix size"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    effective_available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    # Memory estimates
    masks_mem = num_alphas * N1 * N2 * 4 / (1024**3)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)

    print(f"\n[Memory Mode Selection]")
    print(f"  Matrix: {N1}x{N2}, M={M}, S={S}")
    print(f"  Available: {effective_available:.1f} GB")
    print(f"  All masks: {masks_mem:.2f} GB")
    print(f"  Per-alpha: {per_alpha_mem:.2f} GB")

    if mode_override != 'auto':
        print(f"  Mode override: {mode_override}")
        return mode_override

    # Auto-select
    total_parallel_mem = masks_mem + teacher_mem + per_alpha_mem * min(num_alphas, 5)

    if total_parallel_mem < effective_available * 0.85:
        mode = "parallel"
        print(f"  Selected: parallel (batched processing)")
    elif per_alpha_mem < effective_available * 0.7:
        mode = "optimized"
        print(f"  Selected: optimized (sequential, no mask pre-storage)")
    else:
        mode = "extreme"
        print(f"  Selected: extreme (FP16 + sequential)")

    return mode


def calculate_smart_parallelism(N1, N2, M, S, num_alphas):
    """Calculate optimal parallelism based on memory"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    mem_per_alpha = estimate_memory_per_alpha(N1, N2, M, S)

    if mem_per_alpha <= 0:
        return num_alphas

    max_parallel = max(1, min(int(available / mem_per_alpha), num_alphas))
    return max_parallel


# ============================================================
# BiG-AMP Training - Parallel Mode
# ============================================================
def train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, alpha_values, steps, S,
                          damping=0.5, noise_var=1e-6):
    """BiG-AMP training with parallel alpha processing"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    # Initialize estimates
    w_hat = torch.randn((num_alphas, S, N1, M), device=device) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_teacher_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for step in tqdm(range(steps), desc="BiG-AMP Training", leave=False, mininterval=1.0):
        # Forward pass
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher_exp - z_hat) * A_all
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.matmul(A_all / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        # Update X
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


# ============================================================
# BiG-AMP Training - Memory Optimized (Sequential)
# ============================================================
def train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, seed,
                        damping=0.5, noise_var=1e-6, use_fp16=False):
    """Memory-optimized BiG-AMP training for single alpha"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    # Generate mask on-demand
    A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)

    # Initialize
    torch.manual_seed(seed + 10000)
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    w_hat = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
    x_hat = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)
    w_var = (torch.ones_like(w_hat) * (1.0 / M))
    x_var = (torch.ones_like(x_hat) * (1.0 / M))

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
        w_f, x_f = w_hat.float(), x_hat.float()
        w_v, x_v = w_var.float(), x_var.float()

        # Forward
        z_hat = alpha_scale * torch.matmul(w_f, x_f)
        w_sq = w_f ** 2
        x_sq = x_f ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_v) + torch.matmul(w_v, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher - z_hat) * A
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_f.transpose(-2, -1))
        w_hat_new = w_f + w_var_new * r_W
        w_f = damping * w_f + (1 - damping) * w_hat_new
        w_v = torch.clamp(damping * w_v + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        w_hat = w_f.to(storage_dtype)
        w_var = w_v.to(storage_dtype)

        # Update X
        z_hat2 = alpha_scale * torch.matmul(w_f, x_f)
        w_sq2 = w_f ** 2
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_v) + torch.matmul(w_v, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher - z_hat2) * A
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_f.transpose(-2, -1), s2)
        x_hat_new = x_f + x_var_new * r_X
        x_f = damping * x_f + (1 - damping) * x_hat_new
        x_v = torch.clamp(damping * x_v + (1 - damping) * x_var_new, min=1e-8, max=1.0)

        x_hat = x_f.to(storage_dtype)
        x_var = x_v.to(storage_dtype)

    return w_hat.float(), x_hat.float()


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, alpha_values, S):
    """Evaluate metrics for all alphas"""
    results = {}
    num_alphas = len(alpha_values)

    for a_idx, alpha in enumerate(alpha_values):
        trial_results = []

        for s in range(S):
            W_s = W[a_idx, s] if W.dim() == 4 else W[s]
            X_s = X[a_idx, s] if X.dim() == 4 else X[s]

            Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
            Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

            Yp = W_s @ X_s
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))
            gen_error = float(torch.mean((Y_teacher - Yp) ** 2))

            trial_results.append({
                'Q_W': Q_W, 'Q_X': Q_X,
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                'Q_Y': Q_Y, 'Gen_Error': gen_error
            })

        # Aggregate
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
        Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
        Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                   (Y_teacher.norm() * Yp.norm() + 1e-12))
        gen_error = float(torch.mean((Y_teacher - Yp) ** 2))

        trial_results.append({
            'Q_W': Q_W, 'Q_X': Q_X,
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Gen_Error': gen_error
        })

    metrics = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return metrics


# ============================================================
# Main Training Functions
# ============================================================
def run_parallel_mode(alpha_values, steps, S):
    """Run with parallel alpha processing"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    max_parallel = calculate_smart_parallelism(N1, N2, M, S, num_alphas)

    print(f"\n{'='*70}")
    print(f"BiG-AMP TRAINING - PARALLEL MODE")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"Parallelism: {max_parallel} alphas")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, num_alphas, max_parallel):
        batch_end = min(batch_start + max_parallel, num_alphas)
        batch_alphas = alpha_values[batch_start:batch_end]
        batch_size = len(batch_alphas)

        print(f"[Batch {batch_start//max_parallel + 1}] Alpha {batch_alphas[0]:.2f} - {batch_alphas[-1]:.2f}")

        # Pre-generate masks
        A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
        for i, alpha in enumerate(batch_alphas):
            mask_seed = SEED + int(alpha * 1000)
            A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
            A_all[i, 0] = A

        # Train
        W, X = train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, batch_alphas, steps, S,
                                      damping=DAMPING, noise_var=NOISE_VAR)

        # Evaluate
        batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, S)
        all_results.update(batch_results)

        del A_all, W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time


def run_sequential_mode(alpha_values, steps, S, use_fp16=False):
    """Run with sequential alpha processing (memory optimized)"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    mode_name = "EXTREME" if use_fp16 else "OPTIMIZED"

    print(f"\n{'='*70}")
    print(f"BiG-AMP TRAINING - {mode_name} MODE (Sequential)")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"FP16 storage: {use_fp16}")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    for i, alpha in enumerate(alpha_values):
        alpha_seed = SEED + int(alpha * 1000)
        print(f"[{i+1}/{num_alphas}] Alpha = {alpha:.2f}")

        W, X = train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, alpha_seed,
                                    damping=DAMPING, noise_var=NOISE_VAR, use_fp16=use_fp16)

        metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, S)
        all_results[float(alpha)] = metrics

        del W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time


# ============================================================
# Visualization
# ============================================================
def plot_results(results, alpha_values, save_path):
    """Generate result plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    MAIN_COLOR = '#2563eb'
    SECONDARY_COLOR = '#dc2626'

    # Plot 1: Q_Y
    ax1 = axes[0, 0]
    qy_mean = [results[a]['Q_Y_mean'] for a in alpha_values]
    qy_std = [results[a]['Q_Y_std'] for a in alpha_values]
    ax1.errorbar(alpha_values, qy_mean, yerr=qy_std, fmt='o-', color=MAIN_COLOR,
                 capsize=3, markersize=6, linewidth=2, label='Q_Y')
    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax1.set_ylabel('Q_Y', fontsize=12)
    ax1.set_title('Y Overlap (Q_Y)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Q_W' and Q_X'
    ax2 = axes[0, 1]
    qw_mean = [results[a]['Q_W_prime_mean'] for a in alpha_values]
    qx_mean = [results[a]['Q_X_prime_mean'] for a in alpha_values]
    ax2.plot(alpha_values, qw_mean, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2, label="Q_W'")
    ax2.plot(alpha_values, qx_mean, 's-', color=SECONDARY_COLOR, markersize=6, linewidth=2, label="Q_X'")
    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax2.set_ylabel("Q' (normalized)", fontsize=12)
    ax2.set_title("Normalized Gram Overlaps", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Plot 3: Generalization Error
    ax3 = axes[1, 0]
    ge_mean = [results[a]['Gen_Error_mean'] for a in alpha_values]
    ax3.semilogy(alpha_values, ge_mean, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2)
    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax3.set_ylabel('Generalization Error (log)', fontsize=12)
    ax3.set_title('Generalization Error', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Q_W and Q_X (cosine)
    ax4 = axes[1, 1]
    qw_cos = [results[a]['Q_W_mean'] for a in alpha_values]
    qx_cos = [results[a]['Q_X_mean'] for a in alpha_values]
    ax4.plot(alpha_values, qw_cos, 'o-', color=MAIN_COLOR, markersize=6, linewidth=2, label='Q_W')
    ax4.plot(alpha_values, qx_cos, 's-', color=SECONDARY_COLOR, markersize=6, linewidth=2, label='Q_X')
    ax4.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax4.set_ylabel('Q (cosine)', fontsize=12)
    ax4.set_title('Gram Overlaps (Cosine)', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'BiG-AMP Results: {N1}×{N2}, M={M}, Steps={MAX_STEPS}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    global N1, N2, M, ALPHA_TILDE_STEP, MAX_STEPS, SAMPLES_PER_ALPHA, RESULT_DIR

    parser = argparse.ArgumentParser(description='BiG-AMP Optimized Training')
    parser.add_argument('--n1', type=int, default=N1, help='Matrix N1 dimension')
    parser.add_argument('--n2', type=int, default=None, help='Matrix N2 dimension (default: same as N1)')
    parser.add_argument('--m', type=int, default=M, help='Latent dimension M')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Samples per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
    parser.add_argument('--memory-mode', type=str, default='auto',
                        choices=['auto', 'parallel', 'optimized', 'extreme'],
                        help='Memory mode')
    parser.add_argument('--damping', type=float, default=DAMPING, help='BiG-AMP damping factor')
    args = parser.parse_args()

    # Apply args
    N1 = args.n1
    N2 = args.n2 if args.n2 else args.n1
    M = args.m
    MAX_STEPS = args.steps
    SAMPLES_PER_ALPHA = args.samples
    ALPHA_TILDE_STEP = args.alpha_step

    RESULT_DIR = Path(__file__).parent / "Result" / f"{N1}_{N2}_{M}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, ALPHA_TILDE_STEP)

    # Select mode
    mode = select_memory_mode(N1, N2, M, SAMPLES_PER_ALPHA, len(alpha_values), args.memory_mode)

    # Run training
    if mode == 'parallel':
        results, total_time = run_parallel_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA)
    elif mode == 'optimized':
        results, total_time = run_sequential_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA, use_fp16=False)
    else:  # extreme
        results, total_time = run_sequential_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA, use_fp16=True)

    # Save results
    results_data = {
        'config': {
            'N1': N1, 'N2': N2, 'M': M,
            'steps': MAX_STEPS,
            'samples_per_alpha': SAMPLES_PER_ALPHA,
            'damping': DAMPING,
            'noise_var': NOISE_VAR,
            'mode': mode,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {str(k): v for k, v in results.items()}
    }

    results_path = RESULT_DIR / f'bigamp_results_steps{MAX_STEPS}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {results_path}")

    # Plot
    plot_path = RESULT_DIR / f'bigamp_results_steps{MAX_STEPS}.png'
    plot_results(results, [float(a) for a in alpha_values], plot_path)

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Mode: {mode}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Plot: {plot_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
