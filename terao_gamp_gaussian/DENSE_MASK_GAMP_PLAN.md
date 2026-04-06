# Dense Mask G-AMP 導入計画書

## 1. 目的

`terao_gamp_gaussian` に、現行の sparse backend に加えて dense mask backend を導入する。

目的は次の 3 点に限定する。

1. 中高密度領域で GPU 向けの計算性能を改善する
2. 現行 sparse 実装を残したまま、密度に応じて backend を選択できるようにする
3. `F_1_onsager_scaler_var` の数式を維持したまま dense 化し、別アルゴリズムへの置換を避ける


## 2. 前提整理

### 2.1 現行実装の構造

現行 `F_1_onsager_scaler_var` は、観測を edge list として持つ。

- 観測: `i_idx`, `j_idx`, `Y`
- ノード状態: `m_W`, `v_W`, `m_X`, `v_X`
- 出力メッセージ: `g_prev`
- 集約: `scatter_add_`

### 2.2 dense 化してよい対象

初期対象は次に限定する。

- `F_1_onsager_scaler_var`

理由:

- scalar variance 近似を使っており、dense 行列演算へ落としやすい
- full Onsager 版より実装ミスのリスクが低い
- random spreading のような edge-dependent `F[c, μ]` を持たない

### 2.3 今回の非対象

- `F_1_onsager`
- `*_cosine`
- random spreading 系
- graph 理論の改修
- CPU 固定の `run_gamp_parallel.py` の高速化

注記:

- `run_gamp_parallel.py` は現状 CPU 固定であり、dense backend 導入の初期対象にしない


## 3. 到達目標

最終的な構成は、別ディレクトリ増殖ではなく同一モジュール内の backend 切替とする。

- `backend="sparse"`: 現行実装
- `backend="dense"`: 新規 dense mask 実装
- `backend="auto"`: 密度と device に基づく自動選択

MVP では `F_1_onsager_scaler_var` の single-process 実行系のみを対象とする。


## 4. dense backend の表現

### 4.1 観測表現

edge list から次の dense tensor を構築する。

- `A`: `(N1, N2)` の binary mask
- `Y_full`: `(N1, N2)` の観測値行列
- `g_prev_dense`: `(N1, N2)` の出力メッセージ

未観測要素では必ず 0 を保つ。

### 4.2 helper

`graph.py` に以下の helper を追加する。

```python
def generate_dense_mask(self, N1, N2, M, alpha1, device, seed=None):
    i_idx, j_idx, E, C1, C2, alpha2 = self.generate(
        N1, N2, M, alpha1, device, seed
    )
    mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
    if E > 0:
        mask[i_idx, j_idx] = 1.0
    return mask, i_idx, j_idx, E, C1, C2, alpha2
```

MVP では edge list から dense mask を生成する。
高密度向け direct dense graph 生成は後段の最適化項目とする。


## 5. dense 数式設計

## 5.1 設計原則

dense backend は sparse 実装と同じ数式を dense tensor 上に写像する。

重要:

- `omega` の Onsager 項は full matrix product に置き換えない
- `T` 側 Onsager 項も matmul に置き換えない
- `V` は broadcast 和で実装し、不要な行列積を増やさない

### 5.2 dense 版の正しい対応

記号:

- `scale = lam / sqrt(M)`
- `scale_sq = lam**2 / M`
- `chi_W = mean(clamp(v_W - m_W**2, min=0))`
- `chi_X = mean(clamp(v_X - m_X**2, min=0))`

#### Step 1: `omega`

```python
z_hat = scale * (m_W @ m_X)  # (N1, N2)

row_cross_W = (m_W * m_W_prev).sum(dim=1)  # (N1,)
col_cross_X = (m_X * m_X_prev).sum(dim=0)  # (N2,)

omega = z_hat - g_prev_dense * scale_sq * (
    chi_X * row_cross_W[:, None] +
    chi_W * col_cross_X[None, :]
)
```

注記:

- これは現行 sparse 実装の `row_cross_W[i_idx]`, `col_cross_X[j_idx]` を dense に broadcast した形である
- `m_W @ m_W_prev.T` や `m_X_prev @ m_X` への置換はしない

#### Step 2: `V`

```python
row_sq_W = (m_W ** 2).sum(dim=1)  # (N1,)
col_sq_X = (m_X ** 2).sum(dim=0)  # (N2,)

V = scale_sq * (
    M * chi_W * chi_X +
    chi_X * row_sq_W[:, None] +
    chi_W * col_sq_X[None, :]
)
V = torch.clamp(V, min=1e-10)
```

注記:

- `V` は matmul ではなく broadcast 和で十分

#### Step 3: `g`, `dg`

```python
denom = V + noise_var
g_raw = A * (Y_full - omega) / denom
dg = -A / denom

g = damping * g_prev_dense + (1 - damping) * g_raw
g = A * torch.clamp(g, min=-100.0, max=100.0)
```

#### Step 4: `T` 側 Onsager 用係数

```python
g_pair = A * g * g_prev_dense

onsager_W_vec = scale_sq * chi_X * g_pair.sum(dim=1)  # (N1,)
onsager_X_vec = scale_sq * chi_W * g_pair.sum(dim=0)  # (N2,)
```

注記:

- ここは行和・列和であり、matmul ではない

#### Step 5: `Sigma`, `T`

```python
Sigma_W_denom = scale_sq * ((-dg) @ (m_X ** 2).T)
Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)

sum_W = scale * (g @ m_X.T)
T_W = m_W + Sigma_W * (sum_W - onsager_W_vec[:, None] * m_W_prev)

Sigma_X_denom = scale_sq * ((m_W ** 2).T @ (-dg))
Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)

sum_X = scale * (m_W.T @ g)
T_X = m_X + Sigma_X * (sum_X - onsager_X_vec[None, :] * m_X_prev)
```

この形で sparse 実装と数式的に整合する。


## 6. 実装方針

### 6.1 ディレクトリ構成

別バリアント `F_1_onsager_scaler_var_dense` は作らない。
長期的な drift を避けるため、同一パッケージ内で backend を分ける。

推奨ファイル構成:

- `F_1_onsager_scaler_var/core_sparse.py`
- `F_1_onsager_scaler_var/core_dense.py`
- `F_1_onsager_scaler_var/core.py`

`core.py` は dispatcher の役割にする。

### 6.2 実装対象ファイル

新規追加:

- `terao_gamp_gaussian/F_1_onsager_scaler_var/core_dense.py`
- `terao_gamp_gaussian/F_1_onsager_scaler_var/core_sparse.py`
- `terao_gamp_gaussian/benchmarks/benchmark_dense_sparse.py`
- `terao_gamp_gaussian/tests/test_dense_sparse_equivalence.py`

変更:

- `terao_gamp_gaussian/graph.py`
- `terao_gamp_gaussian/F_1_onsager_scaler_var/core.py`
- `terao_gamp_gaussian/F_1_onsager_scaler_var/run_gamp.py`

### 6.3 backend API

`train_single_replica()` に次を追加する。

- `backend: Literal["sparse", "dense", "auto"] = "sparse"`

`run_gamp.py` に次を追加する。

- `--backend sparse|dense|auto`


## 7. フェーズ計画

### Phase 0: ベースライン固定

目的:

- 現行 sparse 実装の速度・収束・メモリを基準化する

作業:

- 小規模・中規模・高密度ケースのベンチスクリプトを用意
- `wall-clock`, `ms/step`, `Q_Y`, `loss`, `peak memory` を保存
- CPU / MPS / CUDA の測定フォーマットを統一

成果物:

- `benchmark_dense_sparse.py`
- baseline JSON / CSV

### Phase 1: dense backend MVP

目的:

- `F_1_onsager_scaler_var` に dense backend を追加する

作業:

- `generate_dense_mask()` を追加
- `core_dense.py` に `gamp_step_with_onsager_dense()` を実装
- `train_single_replica(..., backend=...)` を実装
- `run_gamp.py --backend` を追加

制約:

- `float32` のみ
- single replica の single-process 実行のみ
- `run_gamp_parallel.py` は対象外

### Phase 2: 数学的整合性検証

目的:

- dense と sparse の更新が一致することを確認する

比較条件:

- 同一 seed
- 同一 graph
- 同一 teacher
- 同一初期値

比較項目:

- 1 step ごとの `omega`
- 1 step ごとの `V`
- 1 step ごとの `Sigma_W`, `Sigma_X`
- 1 step ごとの `T_W`, `T_X`
- 最終 `Q_Y`
- observed loss

受入基準:

- CPU では step ごとの差が非常に小さいこと
- GPU では加算順序の違いを許容しつつ、最終 `Q_Y` と loss が整合すること

実務上の初期閾値:

- CPU: `max_abs_diff <= 1e-5` を目標
- GPU: `Q_Y` 差 `<= 1e-3` を暫定許容

### Phase 3: GPU 性能評価

目的:

- dense backend が中高密度で実用上有利かを確認する

測定ケース:

- 低密度: `N=2048, M=64, alpha=0.5`
- 中密度: `N=512, M=128, alpha=0.5`
- 高密度: `N=512, M=128, alpha=4.0`

評価項目:

- `wall-clock`
- `ms/step`
- peak memory

判断:

- 中高密度で dense が優位なら継続
- 低密度では sparse を維持

### Phase 4: auto backend

目的:

- 密度に応じた backend 自動切替を導入する

初期ルール:

- `density <= 0.02`: sparse
- `density >= 0.10`: dense
- その中間: device と実測結果に基づく暫定選択

注記:

- この閾値はローカル測定に基づく暫定値
- SQUID の A100 実測で再調整する

### Phase 5: 後続拡張

対象:

- `F_1_onsager`
- `*_cosine`

条件:

- `scalar_var` 版で dense backend が安定した後に着手する


## 8. メモリ計画

### 8.1 dense backend のコスト

dense backend では `(N1, N2)` tensor を複数持つ。

代表例:

- `A`
- `Y_full`
- `g_prev_dense`
- `omega`
- `V`
- `g`
- `dg`
- 中間 tensor

### 8.2 見積り

float32 では、`N1 x N2` tensor 1 枚のサイズは次の通り。

- `2520 x 2520`: 約 25 MB
- `10000 x 10000`: 約 400 MB

10 枚前後の full tensor を同時に使うと、

- `2520 x 2520`: 数百 MB
- `10000 x 10000`: 数 GB

となる。

### 8.3 MVP の運用制約

- MVP では alpha 並列をしない
- MVP では replica 並列をしない
- 大規模ケースで dense を既定にしない
- `backend=auto` 導入までは手動選択とする


## 9. 検証計画

### 9.1 単体検証

対象:

- `generate_dense_mask()`
- `core_dense.py`

テスト:

- edge list と dense mask の一致
- 未観測要素が常に 0 を保つこと
- `g_prev_dense`, `g`, `dg` が mask 外で 0 であること

### 9.2 等価性検証

ケース:

- `N=32, M=8`, `alpha=0.5, 1.0, 2.0`
- `N=128, M=32`, `alpha=0.5, 1.0, 2.0`

実行:

- dense / sparse を同一初期条件で step-by-step 比較

### 9.3 収束検証

ケース:

- `alpha sweep` を実行し、phase transition が定性的に一致するか確認

比較項目:

- `Q_Y vs alpha`
- observed loss
- steps 固定時の収束挙動


## 10. リスクと対策

### リスク 1: 数式の取り違え

内容:

- dense 化の際に Onsager 項を別の行列積へ置換してしまう

対策:

- `omega`, `V`, `T` の dense 式を本計画書に固定
- 実装前に step-level 等価性テストを書く

### リスク 2: dense backend のメモリ過大

内容:

- 中高密度では速くても、大規模でメモリが先に破綻する

対策:

- MVP は single-process のみ
- `auto` で低密度は sparse に逃がす
- peak memory を必須記録項目にする

### リスク 3: GPU でも dense が常勝ではない

内容:

- 低密度ではゼロ計算が支配的になる

対策:

- dense を全ケース既定にしない
- 閾値を固定で決めず、ベンチ結果で補正する

### リスク 4: graph 実装の制約

内容:

- 現行 graph は厳密 bi-regular ではない

対策:

- backend 比較では graph 実装を固定
- graph 改修は別計画として切り離す


## 11. 受入条件

MVP 完了条件は次の通り。

1. `F_1_onsager_scaler_var` に `backend=sparse|dense|auto` が導入される
2. dense backend が CPU 小規模ケースで sparse backend と step-level に整合する
3. 中高密度ケースで dense backend が sparse backend より高速になる
4. 低密度では sparse backend を選択できる
5. 実行ログに `backend`, `density`, `device`, `peak memory` が記録される


## 12. 実施順序

推奨順序は次の通り。

1. baseline ベンチ整備
2. `generate_dense_mask()` 追加
3. `core_dense.py` 実装
4. step-level 等価性テスト
5. `run_gamp.py --backend` 対応
6. GPU ベンチ
7. `backend=auto` 導入
8. 後続バリアントへ拡張


## 13. 判断

本件は実施価値が高い。

ただし、正しい実施方針は次の通りである。

- `G-AMP` を `BiG-AMP` に置換しない
- `F_1_onsager_scaler_var` の数式を維持したまま dense backend を追加する
- sparse backend を残し、最終的には hybrid 構成にする

MVP の着手対象は `F_1_onsager_scaler_var` の dense backend に限定する。
