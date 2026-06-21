# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイドラインを提供します。

---

## 必ず守るルール

### 言語規範
- **コミュニケーション言語**：日本語
- **コード/コメント/commit**：英語





### 検証ループ（重要）
コード修正後は必ず以下を実行して品質を担保する：
1. **実行して確認**：修正したスクリプトを実行し、エラーがないか確認
2. **結果を検証**：出力されたグラフやCSVが期待通りか確認
3. **明らかな異常のみ報告**：NaN、無限大、負の値（0〜1の範囲外）など明確なエラーのみ指摘。予想外の結果は研究上ありうるため、異常とは判断しない

**推測だけで「完了」と報告しない。必ず実行結果を見て判断する。**

---

---

## プロジェクト構成

### 主要ディレクトリ

| ディレクトリ | 用途 |
|-------------|------|
| `terao_gamp_gaussian/` | Gaussian prior / Gaussian noise のG-AMP実験群 |
| `terao_gd/` | 勾配降下法・ミニバッチGDによる比較実験 |
| `AMP_single_variable/` | 単一変数・小規模検証用AMP実験 |
| `graph_core/` | グラフ生成・グラフ関連の共通コード |
| `../p-body_tensor_completion/` | p-body tensor completion のC++ AMP/BP実装とSEノートブック |

### terao_gamp_gaussian/

Gaussian prior / Gaussian noise のPython実装群。現在の主な研究対象。

```
terao_gamp_gaussian/
├── graph.py                    # 二部グラフ生成
├── utils.py                    # f_input, g_out, 初期化, 正規化
├── Dence_Alternating/          # Dense系の交互更新G-AMP
│   ├── random_graph_version/   # F=1, random graph 版
│   ├── random_F_version/       # random F を保持して計算する版
│   └── non_uniform_n1_graph_version/
├── Edge_Alternating/           # Edge観測上の交互更新G-AMP
│   ├── random_graph_version/   # F=1, edge graph 版
│   ├── random_F_version/       # random F をテンソルとして保持する版
│   ├── random_F_sequentially/  # random F を逐次生成・集約する省メモリ版
│   └── F_1_sequentially/       # F=1 の逐次集約版
├── Dence_cosine/               # Dense系のcosine評価実験
├── Dence_scaler_var_cosine/    # Dense系のscalar variance近似・cosine評価
├── F_1_onsager*/               # F=1 Onsager 実装の旧系列
└── Result_data/, results/      # 実験結果
```

#### random_F_sequentially の要点

`Edge_Alternating/random_F_sequentially/` は、巨大な `E x M` の `F_edge` を保持せず、chunkごとにRademacher `F=±1` を再生成して逐次集約する版。

```
random_F_sequentially/
├── core.py                      # 逐次F生成、W/X half-step、order parameter計算
├── order_parameters_vs_step.py  # 固定条件でstepごとのorder parameterを保存
├── run_noise_sweep.py  # noise_varを横軸にしたcosine/loss sweep
├── Result_data/        # 手動整理済み・比較用データ
└── results/            # 実行ごとの自動出力
```

注意：
- `step = W更新 + X更新`。
- `step 0` は初期状態、`step 1` はWとXを1回ずつ更新した後。
- `order_parameters_vs_step.py` の履歴は step ごとの order parameter で、初期状態も含める。
- 固定条件の step 履歴では loss/cosine ではなく、dense 全体 matrix 上の `q_Y`, `m_overlap_Y`, `convergence` を主に見る。

### terao_gd/

G-AMPとの比較用のGD/AGD実験群。

```
terao_gd/
├── gd_cosine/                    # F=1 cosine評価
├── gd_cosine_F_random/           # random F のGD cosine評価
├── gd_cosine_minibatch/          # ミニバッチGD
├── gd_cosine_minibatch_F_random/ # random F + ミニバッチGD
├── gd_warm/                      # warm start / continuation実験
├── gd_simulation/                # シミュレーション補助
└── results/, gd_result/          # 実験結果
```

### p-body_tensor_completion/

`tera/` の外側にある比較対象のC++実装群。Python G-AMPとの差分調査で頻繁に参照する。

```
p-body_tensor_completion/
├── Message_Passing/
│   ├── AMP_GaussGauss.cpp       # Gaussian prior, Gaussian noise AMP
│   ├── AMP_GaussSign.cpp        # Gaussian prior, Sign output AMP
│   ├── AMP_IsingGaussian.cpp    # Ising prior, Gaussian noise AMP
│   ├── BP_*.cpp                 # BP実装群
│   └── AMP_GaussSign/data/      # C++実行結果
└── SE/
    ├── SE.nb                    # Mathematica State Evolution notebook
    └── SE_*.dat                 # SE出力データ
```

C++ AMP とPython G-AMPを比較するときは、グラフ以外にも以下を必ず確認する：
- 同時更新か交互更新か
- `step` の定義
- 初期化式と `v` の初期値
- Onsager memory の時刻管理
- 評価指標が factor overlap か signal-space cosine か
- `lambda` と `noise_var` の渡し方

---

## 参考論文

本研究は以下の論文に基づいています：

**Graphical model for tensor factorization by sparse sampling**
- arXiv: https://arxiv.org/abs/2510.17886
- 概要: ランダムグラフ上でのスパースサンプリングによるテンソル分解
- 手法: メッセージパッシングアルゴリズム、レプリカ理論
- 設定: Teacher-Student、ベイズ最適設定、dense limit での解析

---

## Dense Limit 定義（重要）

### 条件
```
N1, N2 → ∞, M → ∞
N1, N2 >> M >> 1
```

### パラメータ定義

| 記号 | 定義 | 説明 |
|------|------|------|
| **C1** | `α1 × M` | 各Wノード（行）の次数 |
| **C2** | `α2 × M` | 各Xノード（列）の次数 |
| **α1, α2** | O(1) | 次数パラメータ（定数） |
| **E** | `N1 × C1 = N2 × C2` | 総エッジ（観測）数 |

### 制約条件
```
N1 × C1 = N2 × C2
→ N1 × α1 = N2 × α2
→ α2 = (N1 / N2) × α1
```

### グラフ構造（二部正則グラフ）
- 各行 `W_i` は正確に **C1個** の観測点と接続
- 各列 `X_j` は正確に **C2個** の観測点と接続
- `smf/modules/graphs/random.py` および `terao_gamp/graph.py` で実装

---

## 研究背景

### 問題定義
Teacher-Student マスク行列分解：部分観測 `Y_obs = mask(W₀ × X₀)` が与えられたとき、`Y ≈ Y₀` となるような `W, X` を復元する

### コア研究：相転移現象
- **観測密度** `α̃ = (観測数) / (N₁ × N₂)`
- `α̃ > α̃_c` のとき、再構成品質 Q_Y が急激に 1 に近づく
- 臨界値 `α̃_c` は行列次元比率に依存

### アルゴリズム選択
| アルゴリズム | 収束 | 適用 |
|-------------|------|------|
| **BiG-AMP** | 200-5000 ステップ | 大行列 (N≥500)、推奨 |
| **AGD** | ~20k エポック | デバッグ、小行列 |

---

## コア指標

| 指標 | 範囲 | 意味 |
|------|------|------|
| `Q_Y` | [0, 1] | 再構成品質（回転不変） |
| `Q_Y_unobserved` | [0, 1] | 未観測位置の一致度（汎化） |
| `q_Y` | [-1, 1] | 正規化後のY空間 teacher-student overlap |

### AMP/G-AMPで観測するパラメータ

AMP/G-AMP の状態を比較するときは、生徒の推定値そのものと、teacher-student overlap を混同しない。
このドキュメントでは以下の名前を使う。

- `W_hat`, `X_hat`: 生徒の推定値。コード上の `m_W`, `m_X` に対応する。
- `m_overlap_*`: 正規化前の平均 overlap。
- `Q_*`: 推定値の二乗平均。
- `QQ_*`: 二次モーメント推定値 `v_*` の平均。
- `q_Y`: 正規化後のY空間 overlap。正規化後の order parameter は基本的に `q_Y` のみを見る。
  `q_Y` は観測edge上ではなく、未観測位置も含む dense な全体 matrix 上で計算し、汎化性能を見る。

| 指標 | 定義 | 意味 |
|------|------|------|
| `m_overlap_Y` | `(1 / (N1 * N2 * M)) * sum_{i,j,mu} W_teacher[i,mu] * X_teacher[j,mu] * W_hat[i,mu] * X_hat[j,mu]` | dense 全体 matrix 上での正規化前 teacher-student overlap |
| `m_overlap_W` | `(1 / (N1 * M)) * sum_{i,mu} W_teacher[i,mu] * W_hat[i,mu]` | `W` 因子の正規化前 teacher-student overlap |
| `m_overlap_X` | `(1 / (N2 * M)) * sum_{j,mu} X_teacher[j,mu] * X_hat[j,mu]` | `X` 因子の正規化前 teacher-student overlap |
| `Q_Y_teacher` | `(1 / (N1 * N2 * M)) * sum_{i,j,mu} (W_teacher[i,mu] * X_teacher[j,mu])^2` | 教師Y成分の二乗平均 |
| `Q_Y_student` | `(1 / (N1 * N2 * M)) * sum_{i,j,mu} (W_hat[i,mu] * X_hat[j,mu])^2` | 生徒Y成分の二乗平均 |
| `q_Y` | `m_overlap_Y / sqrt(Q_Y_teacher * Q_Y_student)` | dense 全体 matrix 上での正規化後 teacher-student overlap |
| `Q_W` | `(1 / (N1 * M)) * sum_{i,mu} W_hat[i,mu]^2` | 推定値 `W_hat` の二乗平均 |
| `Q_X` | `(1 / (N2 * M)) * sum_{j,mu} X_hat[j,mu]^2` | 推定値 `X_hat` の二乗平均 |
| `QQ_W` | `(1 / (N1 * M)) * sum_{i,mu} v_W[i,mu]` | 二次モーメント推定値 `v_W` の平均 |
| `QQ_X` | `(1 / (N2 * M)) * sum_{j,mu} v_X[j,mu]` | 二次モーメント推定値 `v_X` の平均 |
| `convergence` | `(sum abs(W_proposal - W_hat_old) + sum abs(X_proposal - X_hat_old)) / ((N1 + N2) * M)` | 生徒の推定値そのものの step 間変化量の平均 |

`m_overlap_W`, `m_overlap_X`, `m_overlap_Y` は正規化前の量であり、`q_Y` とは別に保存する。
`m_overlap_Y`, `Q_Y_teacher`, `Q_Y_student`, `q_Y` は、観測された edge だけではなく、`N1 * N2` 全ペアを使って計算する。
Bayes optimal 設定では教師の二乗平均は1に近いが、推定値 `W_hat`, `X_hat` の二乗平均 `Q_W`, `Q_X` は推定状態を表すため、1に固定して扱わない。
`convergence` は overlap や `q_Y` の変化ではなく、生徒の推定値 `W_hat`, `X_hat` の変化から計算する。
C++ AMP と合わせるため、ダンピング後の実変化量ではなく、ダンピング前の proposal と更新前推定値の差を使う。
`step 0` は前の状態が存在しないため、`convergence` は未定義として `NaN` にする。`step 1` の `convergence` が `step 1 - step 0` に対応する。

### GD/AGDでのconvergence

GD/AGD は AMP/G-AMP と異なり、明示的な最適化目的関数 `loss` を下げるアルゴリズムである。
そのため、GD/AGD の `convergence` は生徒推定値 `W_hat`, `X_hat` の要素ごとの差分ではなく、loss の step 間差分で見る。

| 指標 | 定義 | 意味 |
|------|------|------|
| `convergence` | `abs(loss_t - loss_{t-1})` | 最適化目的関数の step 間変化量 |

GD/AGD で sparse 観測上の loss を使う場合、`convergence` に使う `loss_t` は以下の正規化済み loss とする。

```text
loss_t = (M / |E_obs|) * sum_{(i,j) in E_obs} (Y_obs[i,j] - Y_hat_t[i,j])^2
```

ここで `|E_obs|` は観測 edge 数であり、`Y_hat_t[i,j] = (1 / sqrt(M)) * sum_mu W_hat_t[i,mu] * X_hat_t[mu,j]` とする。
この定義は、実装上の総和 loss `M * sum residual^2` を観測 edge 数で割った `loss_per_edge` に対応する。
ミニバッチGDであっても、convergence 判定や履歴保存に使う `loss_t` は現在のミニバッチ上の loss ではなく、固定された観測 edge 全体 `E_obs` 上で評価する。
`step 0` は前の loss が存在しないため、`convergence` は `NaN` にする。`step 1` の `convergence` が `abs(loss_1 - loss_0)` に対応する。
AMP/G-AMP と GD/AGD を比較するときは、`convergence` の定義が異なるため、収束速度の比較では同じ閾値を機械的に共有しない。

---

## 重要な設定パラメータ

```python
# 行列次元
N1, N2, M = 200, 200, 50        # 行数, 列数, ランク

# Alpha スキャン
ALPHA_TILDE_START = 0.0
ALPHA_TILDE_STOP = 4.0
ALPHA_TILDE_STEP = 0.1

# BiG-AMP
MAX_STEPS = 1000                # イテレーション数
DAMPING = 0.5                   # ダンピング係数

# グラフ構造
USE_BIREGULAR_GRAPH = False    # True=Dinicグラフ, False=ランダムグラフ
```

---

## 交互更新の用語規約

`Dence_Alternating` 系では、用語を以下の意味で固定する。

- **step**:
  `W` と `X` の両方を 1 回ずつ更新し終える単位。
  すなわち、`W(t), X(t)` から出発して
  1. `W(t+1), X(t)` を作る
  2. `W(t+1), X(t+1)` を作る
  の 2 段階をまとめて 1 step と呼ぶ。

- **half-step**:
  上の 2 段階のそれぞれを指す。
  前半は `W` 更新、後半は `X` 更新。

- **更新順**:
  常に `W -> X`。
  つまり、時刻 `t` ではまず `W` を更新し、その更新後の `W(t+1)` と凍結された `X(t)` を用いて `X` を更新する。

- **g, omega, V, dg の再計算**:
  `X` 更新時には、直前に更新した `W(t+1)` と現在の `X(t)` を用いて再計算する。
  `W` 更新と `X` 更新で同じ `g` を使い回さない。

- **Onsager memory の時刻管理**:
  交互最適化では、更新しない変数は凍結されているとみなす。
  そのため Onsager の memory も half-step ごとに進める。
  `W` 更新時と `X` 更新時で、同じ「前時刻」を機械的に共有してはいけない。

- **history / loss / cosine の記録タイミング**:
  `W` 更新後ではなく、`X` 更新まで完了して `W(t+1), X(t+1)` が揃った時点で 1 回だけ記録する。

この規約に従う限り、`step` は「W と X の両方を更新した回数」を表し、half-step はその内部の補助用語としてのみ使う。

---

## Git ブランチ

| ブランチ | 用途 |
|---------|------|
| `main` | 本番コード、リモートにプッシュ |
| `dev` | ローカル開発、smf/ を含む |

```bash
git checkout main             # 本番ブランチに切り替え
git checkout dev              # 開発ブランチに切り替え
git push origin main dev      # 両ブランチをプッシュ
```

**リモートリポジトリ**: `https://github.com/Sulocus/Sparse-Matrix-Factorization.git`

---

## セッション間タスク引き継ぎ

| コマンド | 用途 |
|---------|------|
| `/pass` | 会話終了前にタスク状態を HANDOVER.md に保存 |
| `/rem` | 新しい会話開始時にコンテキストを復元 |

---

## 依存関係

```bash
# コア依存
pip install torch numpy scipy matplotlib rich tqdm pyyaml

# GPU アクセラレーション (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
