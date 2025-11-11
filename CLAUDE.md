# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research implementation of Teacher-Student Masked Matrix Factorization with GPU acceleration. The code simulates learning sparse matrix factorization where a student model tries to recover a teacher model's matrix factorization (Y = W × X) from partial observations.

## Running the Code

### Main Entry Points

Two main implementations are available:

1. **Sequential Alpha Training** ([Main.py](Main.py)):
   ```bash
   python Main.py
   ```
   - Trains each alpha value sequentially
   - Supports early stopping for convergence detection
   - Uses kernel fusion optimization

2. **Parallel Alpha Training** ([Main_multi_alpha.py](Main_multi_alpha.py)) - RECOMMENDED:
   ```bash
   python Main_multi_alpha.py
   ```
   - Trains all alpha values simultaneously in parallel
   - Dramatically faster (reduces Python loop overhead by N_alpha times)
   - Uses same optimizations as sequential version

### Key Configuration Parameters

All parameters are configured via global variables at the top of each script:

**Matrix Dimensions:**
- `N1`, `N2`: Matrix dimensions (teacher matrix is N1×N2)
- `M`: Latent dimension (rank of factorization)

**Training Configuration:**
- `ALPHA_TILDE_START`, `ALPHA_TILDE_STOP`, `ALPHA_TILDE_STEP`: Range of alpha values (sparsity levels)
- `EPOCHS_PER_ALPHA`: Training steps per alpha value
- `LEARNING_RATE`: Student model learning rate
- `SAMPLES_PER_ALPHA`: Number of independent trials per alpha

**Graph Generation:**
- `USE_BIREGULAR_GRAPH`: Choose graph generation method
  - `True`: Dinic max-flow algorithm for bi-regular graphs (uniform degree distribution)
  - `False`: Pure random GPU-based generation (faster, supports any N1≠N2)
- `RESAMPLE_MASK_EACH_TRIAL`: Whether to generate different masks for each trial

**Early Stopping (Main.py only):**
- `USE_EARLY_STOP`: Enable/disable early stopping
- `TARGET_LOSS_THRESHOLD`: Absolute loss threshold for stopping
- `RELATIVE_CHANGE_THRESHOLD`: Relative change threshold for convergence
- `EARLY_STOP_CHECK_INTERVAL`: Steps between convergence checks
- `EARLY_STOP_PATIENCE`: Number of checks showing no change before stopping

**Device & Performance:**
- `DEVICE`: Automatically selects MPS (Apple Silicon), CUDA, or CPU
- `USE_BF16`: Enable BF16 mixed precision (CUDA only)
- TF32 acceleration is automatically enabled on CUDA devices

## Code Architecture

### Core Algorithm Flow

1. **Teacher Model Creation** ([create_teacher_dense](Main.py#L109-L129)):
   - Generates ground truth W_true (N1×M) and X_true (M×N2)
   - Always uses FP32 for precision

2. **Graph/Mask Generation** (two methods available):
   - **Bi-regular** ([sample_pairs_biregular_exact](Main.py#L164-L323)): Uses Dinic max-flow algorithm to generate graphs with uniform degree distribution. Randomness is introduced by shuffling edge addition order.
   - **Random** ([sample_pairs_random_gpu](Main.py#L125-L162)): Pure GPU-based random sampling using `torch.randperm`. Faster and supports any N1≠N2.

3. **Student Model Training**:
   - **Sequential version** ([train_batched_trials_agd](Main.py#L519-L689)): Trains S trials for one alpha value
   - **Parallel version** ([train_all_alphas_parallel](Main_multi_alpha.py#L698-L859)): Trains all alphas simultaneously

4. **Evaluation**:
   - Gram overlap metrics (Q_W, Q_X) - measures how well student recovers teacher subspaces
   - Y overlap (Q_Y) - rotationally invariant measure
   - Generalization error - MSE on entire matrix
   - m² metric for theoretical verification

### Performance Optimizations

The code implements several GPU optimizations:

1. **Kernel Fusion** ([fused_training_step](Main.py#L453-L486)):
   - Reduces GPU kernel launches from ~18 to ~6 per training step
   - Compatible with MPS and CUDA
   - Can be compiled with `torch.compile` on PyTorch 2.0+

2. **Parallel Alpha Training** (Main_multi_alpha.py only):
   - Batches all alpha values together: shape becomes (num_alphas, S, N1, M/N2)
   - Instead of: 21 alphas × 200k steps = 4.2M Python loops
   - We do: 200k steps with all 21 alphas batched = 200k loops
   - Reduces CPU overhead by N_alpha times

3. **Mixed Precision** (CUDA only):
   - BF16 for forward/backward passes (2x speedup, 50% memory reduction)
   - FP32 for parameter storage and final evaluation
   - Controlled by `USE_BF16` flag

4. **TF32 Acceleration** (CUDA only):
   - Automatically enabled for matrix multiplication
   - ~8x speedup for matmul operations with minimal precision impact

### Training Loop Implementation

The core training implements alternating gradient descent:
- **Step 1**: Compute gradient w.r.t. W using current (W, X), update W
- **Step 2**: Recompute with updated W, compute gradient w.r.t. X, update X

Key implementation details:
- Teacher parameters always in FP32 for ground truth accuracy
- Student parameters stored in FP32, converted to COMPUTE_DTYPE during training
- Uses `torch.autocast` for mixed precision support
- Device synchronization before collecting final results

### Output Format

Results are saved to `Result/{N1}_{N2}_{M}/` with filename format:
```
{GraphType}_{Resample}_{EarlyStop}_{KeyParam}_batch{S}.png
```

Where:
- GraphType: `BiReg` or `Rand`
- Resample: `Resample` or `NoResample`
- EarlyStop: `ET` (enabled) or `EF` (disabled)
- KeyParam: `Loss{threshold}` if early stop enabled, else `Epoch{count}`

The plot combines:
- Q_Y metrics (rotationally invariant overlap)
- Q_W' and Q_X' metrics (zero-to-one normalized Gram overlaps)
- Parameter table with configuration

## Development Workflow

When modifying parameters:
1. Edit global variables at the top of the script
2. For quick testing, reduce `N1`, `N2`, `M` and `EPOCHS_PER_ALPHA`
3. Run the script to verify changes work

When debugging performance:
- Check if `torch.compile` is enabled (requires PyTorch 2.0+)
- Verify BF16 is being used on CUDA (check console output)
- Ensure device synchronization happens before collecting results
- For MPS, note that BF16 is not used (uses FP32)

When adding new metrics:
- Compute in the evaluation phase (after training loop) using FP32
- Add to results dictionary in both sequential and parallel versions
- Update `display_results` and `plot_results` functions accordingly
