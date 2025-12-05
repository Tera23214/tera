# Teacher-Student Masked Matrix Factorization

スパース観測下での行列分解における**相転移現象**を研究するための数値シミュレーションコードです。

## 概要

### 問題設定

観測モデル:
```
Y_observed[i,j] = (W* @ X*)[i,j]  for (i,j) ∈ Ω
```

- `W*`: Teacher matrix (N₁ × M)、各要素は N(0, 1/√M) に従う
- `X*`: Teacher matrix (M × N₂)、各要素は N(0, 1/√M) に従う
- `Ω`: 観測位置の集合、density parameter α で決定

### Overlap Metrics

| Metric | 定義 | 範囲 | 物理的意味 |
|--------|------|------|-----------|
| `Q_Y` | Gram overlap of W@X vs W*@X* | [0, 1] | Reconstruction quality（回転不変） |
| `Q_W` | Gram overlap of W vs W* (cosine) | [-1, 1] | Left factor recovery |
| `Q_X` | Gram overlap of X vs X* (cosine) | [-1, 1] | Right factor recovery |
| `Q_W'` | Normalized Q_W | [0, 1] | Normalized left overlap |
| `Q_X'` | Normalized Q_X | [0, 1] | Normalized right overlap |
| `Gen_Error` | MSE between Y_student and Y_teacher | [0, ∞) | Generalization error |

**相転移点 (α_c)**: Q_Y が急激に 1 に近づく臨界観測密度

---

## ディレクトリ構成

```
.
├── agd/                      # Alternating Gradient Descent
│   ├── train_sequential.py   # Sequential α scan
│   └── train_parallel.py     # Parallel α scan（推奨）
│
├── bigamp/                   # Bilinear Generalized AMP
│   ├── train.py              # Standard BiG-AMP training
│   ├── compare_sizes.py      # Finite size effects analysis
│   ├── low_loop_graph.py     # Low-loop graph experiments
│   ├── orthogonal_teacher.py # Orthogonal teacher matrix
│   └── replica_overlap.py    # Replica overlap analysis
│
├── analysis/                 # Analysis tools
│   ├── compare_algorithms.py # AGD vs BiG-AMP comparison
│   ├── slope_ratio.py        # Pre-transition slope analysis
│   └── degree_distribution.py # Degree distribution verification
│
└── results/                  # Experiment results
    ├── standard/             # Standard experiments
    ├── size_scaling/         # Size dependence
    ├── orthogonal_teacher/   # Orthogonal teacher experiments
    ├── low_loop_graph/       # Low-loop experiments
    ├── replica_overlap/      # Replica experiments
    └── comparison/           # Algorithm comparison
```

---

## プログラム詳細

### AGD (Alternating Gradient Descent)

#### `agd/train_sequential.py`
**目的**: Sequential α scan - 各αを順番に学習して相転移曲線を描画

**アルゴリズム**: W と X を交互に勾配降下で最適化
```
W ← W - η * ∇_W L(W, X)
X ← X - η * ∇_X L(W, X)
```

**内部オプション**:
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `USE_BIREGULAR_GRAPH` | `False` | Bi-regular graph vs Random graph |
| `RESAMPLE_MASK_EACH_TRIAL` | `True` | Resample mask for each trial |
| `USE_EARLY_STOP` | `False` | Enable early stopping |
| `TARGET_LOSS_THRESHOLD` | `1e-8` | Absolute loss threshold |
| `RELATIVE_CHANGE_THRESHOLD` | `1e-7` | Relative change threshold |
| `EARLY_STOP_PATIENCE` | `5` | Consecutive checks before stopping |

#### `agd/train_parallel.py`
**目的**: Parallel α scan - 全αを同時にバッチ処理（高速）

**特徴**:
- Kernel fusion（GPUカーネル起動を66%削減）
- BF16 mixed precision（CUDAのみ）
- `torch.compile` 対応

**推奨**: 本番実験にはこちらを使用

---

### BiG-AMP (Bilinear Generalized AMP)

#### `bigamp/train.py`
**目的**: Standard BiG-AMP training

**アルゴリズム**: Approximate Message Passing に基づく高速アルゴリズム

**内部オプション**:
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `DAMPING` | `0.5` | Message damping factor（安定性） |
| `NOISE_VAR` | `1e-10` | Assumed noise variance |
| `MAX_STEPS` | `1000` | Maximum iteration steps |

**収束速度**: 200-5000 steps（AGDの ~20k epochs に比べて高速）

#### `bigamp/compare_sizes.py`
**目的**: Finite size effects analysis - 異なる N, M での相転移比較

**出力**: Multiple configuration comparison plot

#### `bigamp/low_loop_graph.py`
**目的**: Low-loop graph experiments

**背景**: AMPはループの少ないグラフでより安定。4-loopを最小化したグラフの影響を調査。

#### `bigamp/orthogonal_teacher.py`
**目的**: Orthogonal teacher matrix experiments

**背景**: 直交教師行列を使用して finite-size fluctuations を消去し、理論値との比較を容易にする。

#### `bigamp/replica_overlap.py`
**目的**: Replica overlap analysis

**意義**: 複数の独立した解（replica）間の重なりを計算し、解の唯一性/多峰性を検証。

---

### Analysis Tools

#### `analysis/compare_algorithms.py`
**目的**: AGD vs BiG-AMP performance comparison

**出力**: 収束速度・精度の比較プロット

#### `analysis/slope_ratio.py`
**目的**: Pre-transition slope analysis

**分析内容**: 相転移前の Q_Y 傾きと N/M の関係を調査。Fitting formula を出力。

#### `analysis/degree_distribution.py`
**目的**: Degree distribution verification

**用途**: グラフ生成アルゴリズムが期待通りの次数分布を生成しているか検証。

---

## 使用方法

### 環境設定

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# CUDA対応PyTorch（推奨）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 基本的な実行

```bash
# AGD (parallel version, recommended)
python agd/train_parallel.py

# BiG-AMP (fastest)
python bigamp/train.py

# Algorithm comparison
python analysis/compare_algorithms.py
```

### パラメータ設定

各スクリプトの先頭で設定:

```python
# Matrix dimensions
N1 = 200   # Number of rows
N2 = 200   # Number of columns
M = 50     # Rank (hidden dimension)

# Observation density range
ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 4
ALPHA_TILDE_STEP = 0.1

# Training settings
EPOCHS_PER_ALPHA = 5000   # AGD
MAX_STEPS = 1000          # BiG-AMP
LEARNING_RATE = 0.01
```

---

## アルゴリズム比較

| Algorithm | Convergence | Best for | Entry point |
|-----------|-------------|----------|-------------|
| AGD (Sequential) | ~20k epochs | Small matrices, baseline | `agd/train_sequential.py` |
| AGD (Parallel) | ~20k epochs | Faster batched processing | `agd/train_parallel.py` |
| **BiG-AMP** | ~200-5000 steps | **Large matrices (N>1000)** | `bigamp/train.py` |

**推奨**: 本番実験には BiG-AMP を使用（高速、スケーラブル）

---

## 内部オプション一覧

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `USE_BIREGULAR_GRAPH` | agd/*.py | `False` | Use bi-regular graph (Dinic algorithm) |
| `RESAMPLE_MASK_EACH_TRIAL` | agd/*.py | `True` | Resample observation mask each trial |
| `USE_EARLY_STOP` | agd/train_sequential.py | `False` | Enable early stopping |
| `DAMPING` | bigamp/*.py | `0.5` | Message damping factor |
| `NOISE_VAR` | bigamp/*.py | `1e-10` | Assumed noise variance |

---

## 出力形式

各実験は以下のファイルを生成:

- `*.json`: 数値結果（Q_Y, Q_W, Q_X の mean/std など）
- `*.png`: 可視化（相転移曲線プロット）

### プロット内容
- Q_Y（赤）: Reconstruction overlap
- Q_W'（紫）: Normalized left factor overlap
- Q_X'（茶）: Normalized right factor overlap
- Error bars: Standard deviation across trials
