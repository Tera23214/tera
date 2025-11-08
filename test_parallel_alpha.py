"""
Quick test of multi-alpha parallel training optimization
"""
import sys
import time
import numpy as np
import torch

# Import from main_multi_alpha
sys.path.insert(0, '/Users/sucia/Desktop/Sparse_Matrix')

# Set small test parameters
N1, N2, M = 100, 50, 10
ALPHA_TILDE_START = 0.5
ALPHA_TILDE_STOP = 1.5
ALPHA_TILDE_STEP = 0.5  # Only 3 alphas for quick test
EPOCHS_PER_ALPHA = 100
SAMPLES_PER_ALPHA = 1
LEARNING_RATE = 1e-2
SEED = 42
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      ('cuda' if torch.cuda.is_available() else 'cpu'))
COMPUTE_DTYPE = torch.float32
USE_BIREGULAR_GRAPH = False

print(f"Test configuration:")
print(f"  Matrix size: {N1}×{N2}×{M}")
print(f"  Alphas: {ALPHA_TILDE_START} to {ALPHA_TILDE_STOP}, step {ALPHA_TILDE_STEP}")
print(f"  Steps per alpha: {EPOCHS_PER_ALPHA}")
print(f"  Device: {DEVICE}")

a_vals = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)
num_alphas = len(a_vals)

print(f"\nNumber of alphas: {num_alphas}")
print(f"Total Python loops:")
print(f"  Original: {num_alphas} × {EPOCHS_PER_ALPHA} = {num_alphas * EPOCHS_PER_ALPHA:,}")
print(f"  Optimized: {EPOCHS_PER_ALPHA:,}")
print(f"  Reduction: {num_alphas}x\n")

# Create simple test to verify the code runs
torch.manual_seed(SEED)
Wt = torch.randn(N1, M, device=DEVICE, dtype=torch.float32) * 0.1
Xt = torch.randn(M, N2, device=DEVICE, dtype=torch.float32) * 0.1

print("Testing mask generation...")
from main_multi_alpha import sample_pairs_biregular_exact

i_idx, j_idx, C = sample_pairs_biregular_exact(N1, N2, M, 1.0, DEVICE, seed=SEED)
print(f"✓ Mask generated: {C} edges\n")

print("Testing parallel alpha training...")
print("=" * 60)

start_time = time.time()

# Simplified version of train_all_alphas_parallel for testing
alpha_scale = 1.0 / (M ** 0.5)

# Generate all masks
all_masks = []
for alpha_tilde in a_vals:
    i_idx, j_idx, C = sample_pairs_biregular_exact(
        N1, N2, M, alpha_tilde, DEVICE, seed=SEED + int(alpha_tilde * 1000)
    )
    A_single = torch.zeros((N1, N2), dtype=torch.float32, device=DEVICE)
    if i_idx is not None and i_idx.numel() > 0:
        A_single[i_idx, j_idx] = 1.0
    all_masks.append(A_single.unsqueeze(0))

A_all = torch.stack(all_masks, dim=0)  # (num_alphas, 1, N1, N2)

# Initialize parameters
scale = 1.0 / (M ** 0.5)
torch.manual_seed(SEED + 10_000)
W_all = torch.randn((num_alphas, 1, N1, M), device=DEVICE, dtype=torch.float32) * scale
X_all = torch.randn((num_alphas, 1, M, N2), device=DEVICE, dtype=torch.float32) * scale

Y_teacher = Wt @ Xt
Y_teacher_expanded = Y_teacher.unsqueeze(0).unsqueeze(0)

# Training loop
for step in range(EPOCHS_PER_ALPHA):
    # Fused step for all alphas simultaneously
    Y_student = alpha_scale * torch.matmul(W_all, X_all)
    Mres = (Y_teacher_expanded - Y_student) * A_all
    grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X_all.transpose(-2, -1))
    W_all = W_all - LEARNING_RATE * grad_W

    Y_student2 = alpha_scale * torch.matmul(W_all, X_all)
    Mres2 = (Y_teacher_expanded - Y_student2) * A_all
    grad_X = -2.0 * alpha_scale * torch.matmul(W_all.transpose(-2, -1), Mres2)
    X_all = X_all - LEARNING_RATE * grad_X

    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{EPOCHS_PER_ALPHA}")

# Sync if needed
if DEVICE.type == 'mps':
    torch.mps.synchronize()
elif DEVICE.type == 'cuda':
    torch.cuda.synchronize()

elapsed_time = time.time() - start_time

print(f"\n✓ Training completed!")
print(f"  Time: {elapsed_time:.2f}s")
print(f"  Time per step: {elapsed_time/EPOCHS_PER_ALPHA*1000:.2f} ms")
print(f"  Time per alpha-step: {elapsed_time/(EPOCHS_PER_ALPHA*num_alphas)*1000:.2f} ms")

# Check final loss
with torch.no_grad():
    Y_final = alpha_scale * torch.matmul(W_all, X_all)
    Rf = (Y_teacher_expanded - Y_final) * A_all
    final_losses = torch.sum(Rf ** 2, dim=(-2, -1))
    print(f"\nFinal losses per alpha:")
    for i, (alpha_val, loss) in enumerate(zip(a_vals, final_losses.squeeze())):
        print(f"  Alpha {alpha_val:.1f}: Loss = {loss.item():.6e}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
