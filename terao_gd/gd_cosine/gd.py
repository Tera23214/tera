#!/usr/bin/env python
"""
Alternating Gradient Descent (AGD) for sparse matrix factorization with
dense teacher-student overlap evaluation.

This variant aligns the experimental setup with ``gd_cosine_minibatch``:
- teacher / noise are shared across the whole run
- graph is shared per alpha
- replica-to-replica variation comes only from student initialization

Optimization remains full-batch alternating gradient descent, and the reported
loss is the per-edge value ``M * sum((Y - Y_pred)^2) / C``.
"""

#%%

import sys
import math
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path(to get smf modules)
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 =  200   # Number of rows
N2 = 200   # Number of columns  
M = 50     # Rank (hidden dimension)

ALPHA_START = 0
ALPHA_STOP = 4
ALPHA_STEP = 0.2

MAX_STEPS = 100000
LR_BASE = 0.3   # Base learning rate (calibrated for N=1000)
LR = LR_BASE / math.sqrt(N1 * N2 * M)  # Auto-scale: 0.01 for N=1000, ~0.001 for N=3000
NOISE_VAR = 0.0
SHARED_SEED = 1
STUDENT_SEED_BASE = 100
NUM_REPLICAS = 5   # Number of replicas per alpha
CONVERGENCE_THRESHOLD = 1e-5  # Early stopping threshold for loss_per_edge change
INIT_EPSILON: float | None = 1.0

# ============================================================================
# AGD Helper Functions
# ============================================================================


def maybe_torch_compile(func):
    try:
        return torch.compile(mode="reduce-overhead")(func)
    except RuntimeError:
        return func


def nan_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(arr[finite]))


def nan_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.std(arr[finite]))


def validate_init_epsilon(init_epsilon: float | None) -> float | None:
    if init_epsilon is None:
        return None
    epsilon = float(init_epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("init_epsilon must satisfy 0 <= epsilon <= 1.")
    return epsilon


def initialize_student_factors(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    seed: int,
    init_epsilon: float | None,
    random_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 2000)
    epsilon = validate_init_epsilon(init_epsilon)
    if epsilon is None:
        return (
            torch.randn_like(W_teacher) * float(random_scale),
            torch.randn_like(X_teacher) * float(random_scale),
        )

    noise_scale = math.sqrt(max(epsilon - epsilon ** 2, 0.0))
    W_hat = epsilon * W_teacher + noise_scale * torch.randn_like(W_teacher)
    X_hat = epsilon * X_teacher + noise_scale * torch.randn_like(X_teacher)
    return W_hat, X_hat


def compute_predictions(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    M: int,                # Rank for 1/√M scaling
) -> torch.Tensor:
    """
    Compute predictions Y_pred for observed entries.
    
    Y_pred[c] = (1/√M) * sum_mu W[i_c, mu] * X[mu, j_c]
    
    The 1/√M scaling ensures proper normalization: E[Y²] ~ O(1).
    """
    W_sel = W[i_idx.long(), :]       # (C, M)観測された行列の抽出
    X_sel = X[:, j_idx.long()].T     # (C, M)観測された行列のを抽出してから転置
    
    Y_pred = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,)
    return Y_pred


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """
    Compute the optimization loss: L = M * sum((Y - Y_pred)^2)
    
    The M factor compensates for 1/√M scaling in Y, keeping gradient scale unchanged.
    """
    return M * ((Y - Y_pred) ** 2).sum()


def compute_loss_per_edge(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """
    Compute reported loss normalized by the number of observed edges.

    This keeps the optimization loss unchanged while reporting a per-edge value:
    L_report = (M * sum((Y - Y_pred)^2)) / C
    """
    num_edges = max(Y.numel(), 1)
    return compute_loss(Y, Y_pred, M) / num_edges


@maybe_torch_compile
def agd_step_W(
    W: torch.Tensor,   # (N1, M)
    X: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,   # (C,)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Gradient descent step for W (fixing X).
    
    Gradient: dL/dW[i,mu] = 2 * sum_{c: i_c=i} (Y_pred[c] - Y[c]) * X[mu, j_c]
    """
    N1, M = W.shape
    
    # Compute predictions and residuals
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * X[mu, j_c]
    X_sel = X[:, j_idx.long()].T     # (C, M)
    # Gradient includes M factor from loss and 1/√M from Y, net effect: √M factor
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * X_sel  # (C, M)
    
    # Scatter-add gradients to W
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    # Update W
    W_new = W - lr * grad_W
    return W_new


@maybe_torch_compile
def agd_step_X(
    W: torch.Tensor,   # (N1, M)
    X: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,   # (C,)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Gradient descent step for X (fixing W).
    
    Gradient: dL/dX[mu,j] = 2 * sum_{c: j_c=j} (Y_pred[c] - Y[c]) * W[i_c, mu]
    """
    M, N2 = X.shape
    
    # Compute predictions and residuals
    N1 = W.shape[0]  # Get N1 for M parameter
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * W[i_c, mu]
    W_sel = W[i_idx.long(), :]       # (C, M)
    # Gradient includes M factor from loss and 1/√M from Y, net effect: √M factor
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * W_sel  # (C, M)
    
    # Scatter-add gradients to X
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    
    # Update X
    X_new = X - lr * grad_X
    return X_new


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor so that mean square equals 1.
    
    E[x^2] = 1  =>  x_new = x / sqrt(mean(x^2))
    """
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_y_cosine_similarity(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> float:
    """
    Compute cosine similarity between Y_teacher = W_teacher X_teacher and
    Y_student = W_student X_student without materializing the dense Y matrices.
    """
    cross_w = W_teacher.T @ W_student
    cross_x = X_student @ X_teacher.T
    inner = torch.trace(cross_w @ cross_x)

    teacher_norm_sq = torch.trace((W_teacher.T @ W_teacher) @ (X_teacher @ X_teacher.T))
    student_norm_sq = torch.trace((W_student.T @ W_student) @ (X_student @ X_student.T))
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return (inner / denom).item()


ORDER_PARAMETER_KEYS = [
    "m_overlap_Y",
    "m_overlap_W",
    "m_overlap_X",
    "Q_Y_teacher",
]


def compute_order_parameters(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> dict[str, float]:
    """
    Compute dense order parameters defined in CLAUDE.md.

    X tensors use the AGD convention (M, N2), so the X overlap sums over
    the second axis as the column index j.
    """
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    normalizer_y = float(N1 * N2 * M)

    w_overlap_by_mu = torch.sum(W_teacher * W_hat, dim=0)
    x_overlap_by_mu = torch.sum(X_teacher * X_hat, dim=1)
    m_overlap_Y = torch.sum(w_overlap_by_mu * x_overlap_by_mu) / normalizer_y

    w_teacher_sq_by_mu = torch.sum(W_teacher ** 2, dim=0)
    x_teacher_sq_by_mu = torch.sum(X_teacher ** 2, dim=1)
    Q_Y_teacher = torch.sum(w_teacher_sq_by_mu * x_teacher_sq_by_mu) / normalizer_y

    return {
        "m_overlap_Y": float(m_overlap_Y.item()),
        "m_overlap_W": float(torch.mean(W_teacher * W_hat).item()),
        "m_overlap_X": float(torch.mean(X_teacher * X_hat).item()),
        "Q_Y_teacher": float(Q_Y_teacher.item()),
    }


def prepare_global_shared_data(
    device: torch.device,
    seed: int = 1,
    N1: int = N1,
    N2: int = N2,
    M: int = M,
    noise_var: float = NOISE_VAR,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare teacher matrices and a full-grid noise field once for the whole run.
    """
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)

    torch.manual_seed(seed)
    noise_full = torch.randn((N1, N2), device=device, dtype=torch.float32)
    noise_full = noise_full * math.sqrt(noise_var)

    return {
        "seed": seed,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "noise_full": noise_full,
    }


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N1: int = N1,
    N2: int = N2,
    M: int = M,
    noise_var: float = NOISE_VAR,
    global_data: dict[str, torch.Tensor | float | int] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare graph and observed values once for a single alpha.

    Teacher and noise are shared across the whole simulation. The returned
    alpha-specific tensors are shared across replicas, so replica-to-replica
    variation comes only from student initialization.
    """
    if global_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
        )

    graph = RandomGraph()
    i_idx, j_idx, num_observed = graph.generate(N1, N2, M, alpha, device, seed)
    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    noise_full = global_data["noise_full"]

    if num_observed == 0:
        return {
            "alpha": alpha,
            "num_observed": 0,
            "i_idx": i_idx,
            "j_idx": j_idx,
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "Y_clean": torch.empty(0, dtype=torch.float32, device=device),
            "Y_train": torch.empty(0, dtype=torch.float32, device=device),
        }

    Y_clean = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    Y_train = Y_clean + noise_full[i_idx.long(), j_idx.long()]

    return {
        "alpha": alpha,
        "num_observed": num_observed,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "Y_clean": Y_clean,
        "Y_train": Y_train,
    }


def train_single_replica(
    alpha: float | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    N1: int = N1,
    N2: int = N2,
    M: int = M,
    max_steps: int = MAX_STEPS,
    lr: float = LR,
    noise_var: float = NOISE_VAR,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    shared_data: dict[str, torch.Tensor | float | int] | None = None,
    return_order_parameters: bool = False,
    init_epsilon: float | None = INIT_EPSILON,
):
    """
    Train a single replica using full-batch alternating gradient descent.

    If ``shared_data`` is provided, graph / teacher / noisy observations are
    reused. Otherwise teacher/noise are generated once and graph is generated
    for the requested alpha. The ``seed`` argument controls only the student
    initialization.
    """
    if shared_data is None:
        if alpha is None:
            raise ValueError("alpha must be provided when shared_data is None.")
        if device is None:
            raise ValueError("device must be provided when shared_data is None.")
        global_data = prepare_global_shared_data(
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
        )
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
            global_data=global_data,
        )

    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    Y_train = shared_data["Y_train"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    num_observed = int(shared_data["num_observed"])

    W_hat, X_hat = initialize_student_factors(
        W_teacher,
        X_teacher,
        seed=seed,
        init_epsilon=init_epsilon,
        random_scale=0.01,
    )

    if num_observed == 0:
        order_parameters = compute_order_parameters(
            W_hat, X_hat, W_teacher, X_teacher
        )
        order_parameters["convergence"] = float("nan")
        if return_order_parameters:
            return order_parameters, 0.0, 0
        cosine_similarity = compute_y_cosine_similarity(
            W_hat, X_hat, W_teacher, X_teacher
        )
        return cosine_similarity, 0.0, 0

    Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
    previous_loss = float(compute_loss_per_edge(Y_train, Y_pred, M).item())
    final_loss = previous_loss
    final_convergence = float("nan")
    steps_taken = max_steps

    for step in range(1, max_steps + 1):
        W_hat = agd_step_W(W_hat, X_hat, Y_train, i_idx, j_idx, lr)
        X_hat = agd_step_X(W_hat, X_hat, Y_train, i_idx, j_idx, lr)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)

        Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
        final_loss = float(compute_loss_per_edge(Y_train, Y_pred, M).item())
        final_convergence = abs(final_loss - previous_loss)
        previous_loss = final_loss

        if final_convergence < convergence_threshold:
            steps_taken = step
            break

    order_parameters = compute_order_parameters(W_hat, X_hat, W_teacher, X_teacher)
    order_parameters["convergence"] = final_convergence
    if return_order_parameters:
        return order_parameters, final_loss, steps_taken

    cosine_similarity = compute_y_cosine_similarity(
        W_hat, X_hat, W_teacher, X_teacher
    )
    return cosine_similarity, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    INIT_EPSILON = validate_init_epsilon(INIT_EPSILON)

    print("=" * 60)
    print("Alternating Gradient Descent (AGD) - Matrix Factorization")
    print("GPU Accelerated with Multiple Replicas")
    print("=" * 60)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        print("Using: CPU")
    
    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Steps: {MAX_STEPS}, LR={LR}")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print(f"Teacher / graph / noise seed: {SHARED_SEED}")
    print(f"Student seed rule: {STUDENT_SEED_BASE} + replica_id")
    print(
        "Student init:",
        (
            "0.01 * N(0, 1) random initialization"
            if INIT_EPSILON is None
            else (
                "epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1), "
                f"epsilon={INIT_EPSILON}"
            )
        ),
    )
    print("Shared across run: teacher / noisy field")
    print("Shared per alpha: graph")
    print("Replica-specific: student initialization only")
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_agd_m_overlap_Y_{N1}x{M}_alpha{ALPHA_START}-{ALPHA_STOP}_"
        f"initeps{INIT_EPSILON if INIT_EPSILON is not None else 'random'}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_m_overlap_y_alpha_sweep',
        'N1': N1,
        'N2': N2,
        'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'max_steps': MAX_STEPS,
        'lr': LR,
        'lr_base': LR_BASE,
        'noise_var': NOISE_VAR,
        'teacher_seed': SHARED_SEED,
        'graph_seed': SHARED_SEED,
        'noise_seed': SHARED_SEED,
        'student_seed_base': STUDENT_SEED_BASE,
        'student_init_mode': (
            'random_gaussian' if INIT_EPSILON is None else 'correlated_gaussian'
        ),
        'student_init_formula': (
            '0.01 * N(0, 1)'
            if INIT_EPSILON is None
            else 'epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1)'
        ),
        'student_init_epsilon': INIT_EPSILON,
        'num_replicas': NUM_REPLICAS,
        'convergence_threshold': CONVERGENCE_THRESHOLD,
        'device': str(device),
        'evaluation_metric': 'dense_teacher_student_overlap_order_parameter',
        'order_parameters': list(ORDER_PARAMETER_KEYS),
        'loss_definition': (
            '(M / |E_obs|) * sum_{(i,j) in E_obs} '
            '(Y_obs[i,j] - Y_hat[i,j])^2'
        ),
        'convergence_definition': 'abs(loss_per_edge_t - loss_per_edge_{t-1})',
        'unavailable_order_parameters': {
            'Q_W': 'GD does not maintain a posterior variance estimate v_W.',
            'Q_X': 'GD does not maintain a posterior variance estimate v_X.',
        },
        'shared_teacher_noise_global': True,
        'shared_graph_per_alpha': True,
        'replica_variation': 'student_initialization_only',
        'early_stop_metric': 'convergence',
        'early_stop_rule': 'stop when convergence < convergence_threshold',
        'output_files': [
            'config.yaml',
            'alpha_summary.csv',
            'plots/m_overlap_Y_vs_alpha.png',
        ],
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {}
    
    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
    global_data = prepare_global_shared_data(
        device=device,
        seed=SHARED_SEED,
        N1=N1,
        N2=N2,
        M=M,
        noise_var=NOISE_VAR,
    )

    for alpha in alphas:
        order_parameter_values = {key: [] for key in ORDER_PARAMETER_KEYS}
        convergence_values = []
        loss_values = []
        steps_values = []
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=SHARED_SEED,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=NOISE_VAR,
            global_data=global_data,
        )
        for replica_id in range(NUM_REPLICAS):
            seed = STUDENT_SEED_BASE + replica_id
            t0 = time.time()
            order_parameters, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
                device=device,
                seed=seed,
                N1=N1,
                N2=N2,
                M=M,
                max_steps=MAX_STEPS,
                lr=LR,
                noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                shared_data=shared_data,
                return_order_parameters=True,
                init_epsilon=INIT_EPSILON,
            )
            dt = time.time() - t0
            for key in ORDER_PARAMETER_KEYS:
                order_parameter_values[key].append(order_parameters[key])
            convergence_values.append(order_parameters["convergence"])
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1
            convergence_text = (
                "nan"
                if math.isnan(order_parameters["convergence"])
                else f"{order_parameters['convergence']:.2e}"
            )
            print(
                f"α={alpha:.2f}, replica {replica_id+1}/{NUM_REPLICAS}: "
                f"m_overlap_Y={order_parameters['m_overlap_Y']:.4f}, Loss/edge={final_loss:.2e}, "
                f"convergence={convergence_text}, Steps={steps_taken} "
                f"({dt:.1f}s) [{completed}/{total_tasks}]"
            )
        
        alpha_result = {
            'loss_mean': np.mean(loss_values),
            'loss_std': np.std(loss_values),
            'loss_values': loss_values,
            'convergence_mean': nan_mean(convergence_values),
            'convergence_std': nan_std(convergence_values),
            'convergence_values': convergence_values,
            'steps_mean': np.mean(steps_values),
            'steps_values': steps_values,
        }
        for key in ORDER_PARAMETER_KEYS:
            values = order_parameter_values[key]
            alpha_result[f'{key}_mean'] = np.mean(values)
            alpha_result[f'{key}_std'] = np.std(values)
            alpha_result[f'{key}_values'] = values
        results[alpha] = alpha_result
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'m_overlap_Y':^20} | {'Loss/edge':^20} | {'Steps':>8}")
    print("-" * 60)
    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(
            f"{alpha:6.2f} | "
            f"{r['m_overlap_Y_mean']:8.4f} ± {r['m_overlap_Y_std']:<8.4f} | "
            f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | "
            f"{r['steps_mean']:8.0f}"
        )
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot m_overlap_Y vs alpha with error bars
    print("\nGenerating plots...")
    
    alphas_list = sorted(results.keys())
    m_overlap_y_means = [results[a]['m_overlap_Y_mean'] for a in alphas_list]
    m_overlap_y_stds = [results[a]['m_overlap_Y_std'] for a in alphas_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(alphas_list, m_overlap_y_means, yerr=m_overlap_y_stds,
                fmt='o-', color='#1E88E5', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5)
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel("m_overlap_Y", fontsize=14)
    ax.set_title(f'Phase Transition (AGD)\n({N1}×{N2}, M={M}, {MAX_STEPS} steps, {NUM_REPLICAS} replicas)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create plots subdirectory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save results to CSV
    csv_path = results_dir / "alpha_summary.csv"
    with open(csv_path, 'w') as f:
        # Header
        header = "alpha"
        for key in ORDER_PARAMETER_KEYS:
            header += f",{key}_mean,{key}_std"
        header += (
            ",loss_per_edge_mean,loss_per_edge_std,"
            "convergence_mean,convergence_std,steps_mean"
        )
        for i in range(NUM_REPLICAS):
            for key in ORDER_PARAMETER_KEYS:
                header += f",{key}_replica_{i + 1}"
            header += (
                f",loss_per_edge_replica_{i + 1},"
                f"convergence_replica_{i + 1},steps_replica_{i + 1}"
            )
        f.write(header + "\n")
        
        # Data
        for alpha in alphas_list:
            r = results[alpha]
            line = f"{alpha}"
            for key in ORDER_PARAMETER_KEYS:
                line += f",{r[f'{key}_mean']},{r[f'{key}_std']}"
            line += (
                f",{r['loss_mean']},{r['loss_std']},"
                f"{r['convergence_mean']},{r['convergence_std']},"
                f"{r['steps_mean']}"
            )
            for replica_idx in range(NUM_REPLICAS):
                for key in ORDER_PARAMETER_KEYS:
                    line += f",{r[f'{key}_values'][replica_idx]}"
                line += (
                    f",{r['loss_values'][replica_idx]},"
                    f"{r['convergence_values'][replica_idx]},"
                    f"{r['steps_values'][replica_idx]}"
                )
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")

    # Save plot
    plot_path = plots_dir / "m_overlap_Y_vs_alpha.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close(fig)
    print(f"\nResults saved to: {results_dir}")
    
    print("Done!")


# %%
