# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research implementation of Teacher-Student Masked Matrix Factorization with GPU acceleration. The code simulates learning sparse matrix factorization where a student model tries to recover a teacher model's matrix factorization (Y = W × X) from partial observations.

## Running the Code

### Main Entry Points

Three main implementations are available:

1. **Sequential Alpha Training** ([Main.py](Main.py)):
   ```bash
   python Main.py
   ```
   - Trains each alpha value sequentially
   - Supports early stopping for convergence detection
   - Uses kernel fusion optimization

2. **Parallel Alpha Training** ([Main_multi_alpha.py](Main_multi_alpha.py)):
   ```bash
   python Main_multi_alpha.py
   ```
   - Trains all alpha values simultaneously in parallel
   - Dramatically faster (reduces Python loop overhead by N_alpha times)
   - Uses SGD with fixed learning rate

3. **Optimized Parallel Training** (In `optimization_tests/step1_adam_scheduler/`):
   ```bash
   python optimization_tests/step1_adam_scheduler/program/Main_step1_adam_scheduler.py
   ```
   - **10x faster convergence**: Manual Adam optimizer + Cosine Annealing LR scheduler
   - Same parallel alpha training architecture as Main_multi_alpha.py
   - Achieves baseline 20k epoch quality in just 2k epochs
   - Validated and ready for production (see VALIDATION_REPORT.md)

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
   - **Sequential version** ([train_batched_trials_agd](Main.py#L500)): Trains S trials for one alpha value
   - **Parallel version** ([train_all_alphas_parallel](Main_multi_alpha.py)): Trains all alphas simultaneously

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

## File Organization

**Repository Structure:**

```
Sparse-Matrix/
├── Main.py                              # Sequential alpha training (baseline)
├── Main_multi_alpha.py                  # Parallel alpha training (SGD)
├── CLAUDE.md                            # Project documentation
├── Result/                              # Production results
│   └── {N1}_{N2}_{M}/
│       ├── *.json                       # Result data files
│       └── *.png                        # Generated plots
└── optimization_tests/                  # Testing & optimization experiments
    ├── phase_transition_analyzer.py     # Phase transition detection (Mode 1-2)
    ├── compare_with_phase_check.py      # Enhanced validation with phase check
    ├── VALIDATION_REPORT.md             # Step1 validation report
    ├── utils/
    │   └── log_parser.py                # Training log parser (corrected)
    ├── baseline/                        # Baseline comparison programs
    ├── step1_adam_scheduler/            # Step 1: Adam + LR scheduler (10x speedup)
    ├── step2_adaptive_sampling/         # Step 2: Adaptive alpha sampling
    ├── step3_precise_analysis/          # Step 3: Precise phase analysis (Mode 3)
    │   ├── phase_transition_analyzer_v2.py
    │   ├── run_precise_analysis.py
    │   └── README.md
    └── Result/                          # Test results (separate from production)
        └── {N1}_{N2}_{M}/
```

**Important File Organization Rules:**

1. **Production code** (Main*.py) → Root directory
2. **Test/experimental code** → `optimization_tests/` directory
3. **Results separation**:
   - Production results: `Result/`
   - Test results: `optimization_tests/Result/`
4. **Naming conventions**:
   - Main programs: `Main_*.py`
   - Test scripts: `test_*.py`
   - Utility tools: descriptive names (e.g., `phase_transition_analyzer.py`)

**Never mix production and test code in the same directory!**

## Optimizer Configurations

### Adam + Cosine Annealing LR (Optimized Version)

The optimized version uses manual Adam implementation to preserve alternating gradient descent:

```python
# Hyperparameters (optimization_tests/step1_adam_scheduler/program/Main_step1_adam_scheduler.py)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
LEARNING_RATE = 1e-2            # Initial LR
LR_SCHEDULER_ETA_MIN = 1e-6     # Minimum LR
```

**Why manual Adam?**
- PyTorch's `torch.optim.Adam` expects all parameters in one step
- Our algorithm requires alternating updates (W then X)
- Manual implementation maintains separate momentum/velocity states for W and X

**Performance Impact:**
- 10x faster convergence vs baseline SGD
- Baseline 20k epochs ≈ Optimized 2k epochs
- Suitable for large-scale experiments (300k → 30k epochs)

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

When running tests:
- Always use CUDA for speed (avoid CPU tests)
- **Increase `SAMPLES_PER_ALPHA` for tests** (e.g., 20-50 instead of 5)
  - Larger batch sizes give more accurate statistics
  - Crucial for validation and comparison tests
- Place test scripts in `optimization_tests/` directory
- Use descriptive names (`test_*.py` for test scripts)

## Optimization Roadmap

This section documents the systematic optimization process to achieve ~30x speedup through optimizer improvements and intelligent alpha sampling.

### Complete Optimization Path

```
Baseline (SGD) → Step1 (Adam+LR) → Step2 (Adaptive Sampling) → Production
   ↓                ↓                   ↓                         ↓
  20k epochs      10x faster          +3x sampling            Deploy
                                       ≈ 30x total
```

**Note**: Original "Step1 (Adam with fixed LR)" was removed after proving ineffective. Adam alone (without LR scheduling) cannot converge in sparse constraint problems due to fixed high learning rate causing oscillation near minimum. Even 5M epochs failed (Q_Y negative). This validates that **LR scheduling is essential** for Adam to work in this problem domain.

### Step-by-Step Progress

**✅ Step1: Adam + Learning Rate Scheduler** (`optimization_tests/step1_adam_scheduler/`)
- **Goal**: Add adaptive optimization (Adam) with LR scheduling to accelerate convergence
- **Implementation**:
  - Manual Adam (β1=0.9, β2=0.999, ε=1e-8) for alternating gradient descent
  - Cosine Annealing LR: 1e-2 → 1e-6 over training
- **Result**: **10x speedup** - 2k epochs (Step1) ≈ 20k epochs (Baseline)
- **Validation**: Epoch sweep test (2k, 4k, 8k, 12k, 16k, 20k) + phase transition consistency
- **Status**: ✅ Completed and validated (VALIDATION_REPORT.md)

**🚧 Step2: Adaptive Sampling** (`optimization_tests/step2_adaptive_sampling/`)
- **Goal**: Intelligent alpha sampling based on phase transition detection
- **Implementation**: Two-stage training (coarse scan → smart alphas → fine training)
- **Expected**: **3x additional speedup** (106 points vs 301 uniform points)
- **Validation**: Dual validation (point-by-point + phase transition accuracy)
- **Status**: 🚧 In Progress

**✅ Step3: Precise Phase Analysis** (`optimization_tests/step3_precise_analysis/`)
- **Goal**: Research-grade accurate phase transition point detection
- **Implementation**: Multi-round training with increasing epochs, thermodynamic limit extrapolation
- **Use Case**: Scientific analysis, precise α_c determination
- **Status**: ✅ Implemented (see `optimization_tests/step3_precise_analysis/README.md` for usage)

### PhaseTransitionAnalyzer Architecture

The `PhaseTransitionAnalyzer` class provides three distinct functional modes:

#### **Functional Mode 1: Basic Analysis** (Post-processing)
**Purpose**: Analyze existing training results

**Input**:
- Alpha list + training metrics (overlap data)

**Output**:
- Phase transition detection (α_c, gradient, consistency)
- Anomaly detection (deceleration, decline, volatility)
- Region classification (phase/anomaly/stable)
- Text report + 4-panel visualization

**Use Cases**:
- Analyzing completed experiments
- Generating publication-quality figures
- Understanding phase transition behavior

**Key Methods**:
```python
analyzer = PhaseTransitionAnalyzer(alphas, metrics)
phase = analyzer.detect_phase_transition_enhanced()
report = analyzer.generate_report()
analyzer.plot_full_analysis(save_path="analysis.png")
```

#### **Functional Mode 2: Simple Adaptive Analysis** (Acceleration)
**Purpose**: Intelligent alpha sampling for computational efficiency

**Workflow**:
1. **Coarse scan**: Train with uniform sparse alphas (e.g., Δα=0.1, low epochs)
2. **Analyze**: Use Mode 1 to detect phase transition and anomaly regions
3. **Generate smart alphas**: Dense sampling (Δα=0.01) in phase region, sparse (Δα=0.5) in stable regions
4. **Fine training**: Train with smart alphas using high epochs

**Expected Speedup**: ~3x (106 points vs 301 uniform Δα=0.01)

**Configuration**:
```python
def simple_adaptive_analysis(
    train_callback,       # Main training function
    coarse_epochs=2000,   # Coarse scan epochs (Adam-optimized)
    coarse_step=0.1,      # Coarse scan step
    phase_step=0.02,      # Phase region density
    stable_step=0.2,      # Stable region density
    fine_epochs=20000,    # Fine training epochs
    sensitivity='medium'   # Anomaly detection sensitivity
):
```

**Sensitivity Levels**:
- `'strict'`: Only phase transition region (severity_threshold=3.0)
- `'medium'`: Phase + significant anomalies (severity_threshold=2.0)
- `'loose'`: All detected anomalies (severity_threshold=0.5)

**Use Cases**:
- Production runs with unknown parameters
- Validation tests (对拍) with reduced computational cost
- Exploratory parameter sweeps

#### **Functional Mode 3: Precise Phase Analysis** (Research)
**Purpose**: Determine thermodynamic limit of phase transition point

**Workflow**:
1. **Initial localization**: Run Mode 2 to roughly locate α_c
2. **Focused refinement**: Narrow range (α_c ± 0.5 → α_c ± 0.05)
3. **Epoch escalation**: Train with increasing epochs [20k, 50k, 100k, 200k, ...]
4. **Convergence detection**: Stop when |Δα_c| < 0.001 for 3 consecutive rounds
5. **Extrapolation**: Fit α_c vs 1/epoch → extrapolate to epoch → ∞

**Convergence Criteria** (all must be met):
- Phase center stability: |α_c(current) - α_c(prev)| < 0.001
- Gradient stability: |grad(current) - grad(prev)| / grad(prev) < 1%
- Consecutive stability: 3 rounds without significant change

**Output**:
- α_c in thermodynamic limit (epoch → ∞)
- Confidence interval
- Training history with convergence curve

**Use Cases**:
- Research publications
- Theoretical validation
- Precise critical exponent determination

### Validation Methodology (对拍)

**"对拍" (duìpāi)** is a systematic comparison testing approach ensuring new implementations match baseline behavior.

#### Validation Workflow for Step2

**Phase 1: Generate Smart Alphas**
```python
# Step2 runs Mode 2
coarse_results = train(coarse_alphas, coarse_epochs)
analyzer = PhaseTransitionAnalyzer(coarse_alphas, coarse_results)
smart_alphas = analyzer.get_all_adaptive_alphas()  # e.g., 106 points
```

**Phase 2: Fair Comparison**
```python
# Train both with SAME alphas
step2_results = train_optimized(smart_alphas, fine_epochs)
baseline_results = train_baseline(smart_alphas, fine_epochs)
```

**Phase 3: Dual Validation**

**Validation 1: Point-by-Point Comparison**
- Compare overlap metrics at same alpha points
- Tolerance: 10% relative error
- Metrics: Q_Y, Q_W, Q_X, Q_W', Q_X'

**Validation 2: Phase Transition Accuracy**
```python
# Use Mode 1 to analyze both
analyzer_step2 = PhaseTransitionAnalyzer(smart_alphas, step2_results)
analyzer_baseline = PhaseTransitionAnalyzer(smart_alphas, baseline_results)

alpha_c_step2 = analyzer_step2.detect_phase_transition_enhanced()['alpha_c']
alpha_c_baseline = analyzer_baseline.detect_phase_transition_enhanced()['alpha_c']

# Compare phase transition detection accuracy
phase_error = abs(alpha_c_step2 - alpha_c_baseline)
assert phase_error < 0.1  # Within 0.1 in alpha
```

**Phase 4: Epoch Sweep (10% → 100%)**
- Repeat validation with epochs: [2k, 4k, 8k, 12k, 16k, 20k]
- Ensure convergence behavior is consistent
- Verify speedup claims hold across epoch ranges

#### Why Dual Validation?

**Traditional validation alone is insufficient** because:
- Point-by-point comparison may miss global behavior changes
- Phase transition is the **core physical phenomenon** being studied
- If α_c shifts, the entire physical interpretation changes

**Dual validation ensures**:
1. **Correctness**: Same numerical results
2. **Physical consistency**: Same phase transition behavior
3. **Scientific validity**: Results are publishable

### Critical Design Decisions (Anti-Amnesia Record)

These decisions were made through extensive discussion and must NOT be changed without careful consideration:

#### 1. Manual Adam Implementation
**Why**: PyTorch's `torch.optim.Adam` applies updates to all parameters simultaneously, but we need **alternating gradient descent** (update W, then update X separately).

**Implementation**: Maintain separate momentum (`m_W`, `m_X`) and velocity (`v_W`, `v_X`) states, apply bias correction per step.

**Critical**: Parallel alpha training shares **same step counter** for all alphas - this caused Q_Y=0.11 bug in early implementation.

#### 2. Parallel Alpha Training Architecture
**Design**: Batch all alphas together from the start (Step2 uses parallel, not sequential)

**Shape**: `(num_alphas, S, N1, M)` where `num_alphas` is batched dimension

**Why**: Reduces Python loop overhead by N_alpha times (e.g., 31 alphas → 31x fewer loop iterations)

**Correct**: Step2 = `Main_multi_alpha.py` framework + Adam + LR scheduler (already parallel!)

#### 3. PhaseTransitionAnalyzer Does NOT Train
**Design**: Analyzer receives training results, does NOT contain training logic

**Reason**: Separation of concerns - analysis logic should be independent of training implementation

**Implementation**: Analyzer calls `train_callback` function provided by main program

#### 4. Fair Comparison in 对拍
**Critical**: When comparing Step2 (adaptive) vs Baseline, **both must use same alpha points**

**Workflow**:
1. Step2 generates smart alphas → trains → results A
2. **Reuse same smart alphas** → train baseline → results B
3. Compare A vs B at exact same alpha points

**Wrong**: Generate different alphas for baseline (can't compare fairly)

#### 5. File Organization Rules
**Production**: Root directory (`Main*.py`)
**Testing**: `optimization_tests/` directory
**Never mix**: Test code must stay in `optimization_tests/` until fully validated

**Reason**: Prevents premature deployment of buggy code - test thoroughly in `optimization_tests/` before promoting to root

#### 6. Result Separation
**Production results**: `Result/` directory
**Test results**: `optimization_tests/Result/` directory
**Precise analysis results**: `optimization_tests/Result/{N1}_{N2}_{M}/precise_analysis/` directory

**Critical**: Never modify production `Result/` until all testing stages complete

**Why**: Preserves baseline data for validation, prevents data corruption

### Expected Performance Gains

| Optimization | Mechanism | Speedup | Cumulative |
|---|---|---|---|
| Baseline | SGD, fixed LR | 1x | 1x |
| Step1: Adam+LR Scheduler | Adaptive optimization + LR decay | **10x** | **10x** |
| Step2: Adaptive Sampling | Reduced alpha points | ~3x | **~30x** |

**Note**: Original "Step1 (Adam alone)" removed - proved ineffective (failed even with 5M epochs).

**Final Target**: 300k baseline epochs → ~10k optimized epochs (production-ready)
