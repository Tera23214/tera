# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイドラインを提供します。

---

## 必ず守るルール

### 言語規範
- **コミュニケーション言語**：日本語
- **コード/コメント/commit**：英語

### Wang/ プログラム規範（重要）
1. **プログラムの独立性**：各プログラムは完全に独立し、共有モジュールをインポートしない
2. **コードの一致性**：異なるプログラム間で同じコードブロックは**完全に一致**させる（コピペレベル）
3. **ドキュメント言語**：Wang/README.md は日本語、技術用語は英語

### SMF ドキュメント同期ルール（重要）
- **smf/ モジュールを更新する際は、必ず `smf_docs/` の対応するドキュメントも同期更新する**
- 新機能がわからない時は、まず `smf_docs/README.md` を参照

### 検証ループ（重要）
コード修正後は必ず以下を実行して品質を担保する：
1. **実行して確認**：修正したスクリプトを実行し、エラーがないか確認
2. **結果を検証**：出力されたグラフやCSVが期待通りか確認
3. **明らかな異常のみ報告**：NaN、無限大、負の値（0〜1の範囲外）など明確なエラーのみ指摘。予想外の結果は研究上ありうるため、異常とは判断しない

**推測だけで「完了」と報告しない。必ず実行結果を見て判断する。**

---

## よく使うコマンド

```bash
# SMF フレームワークのインストール（開発モード）
pip install -e .

# テスト
pytest tests/                           # 全テスト実行
pytest tests/test_file.py -v            # 単一テスト、詳細出力

# SMF CLI
smf                # 対話モード（自然言語設定）
smf run            # 実験ウィザード
smf run --bg       # バックグラウンド実行
smf resume         # チェックポイント復元
smf log            # ログ表示
smf log -f         # リアルタイムログ追跡
smf vis            # 結果ブラウザ
smf test           # クイックテスト

# 本番トレーニング (Wang/) - 独立プログラム、smf に依存しない
python Wang/bigamp/train.py           # BiG-AMP（推奨）
python Wang/agd/train_parallel.py     # AGD 並列版

# terao_gd/ シミュレーション
python terao_gd/gd.py                 # 基本AGD
python terao_gd/gd_noisy.py           # ノイズ付きAGD
python terao_gd/gd_order_params.py    # オーダーパラメータ解析
```

---

## プロジェクト構成

### デュアルトラックシステム

| ディレクトリ | Git | 用途 |
|-------------|-----|------|
| `Wang/` | main ブランチ | 本番コード、日本の同僚と共有 |
| `smf/` | dev ブランチ | モジュール化フレームワーク、ローカル開発 |
| `smf_docs/` | dev ブランチ | SMF モジュールドキュメント |
| `terao_gd/` | - | 寺尾さん用 勾配降下法実験 |
| `_legacy/` | - | アーカイブ済み旧コード |

### Wang/ ディレクトリ
```
Wang/
├── bigamp/              # BiG-AMP アルゴリズム（推奨）
│   ├── train.py         # 標準トレーニング
│   ├── compare_sizes.py # サイズ比較実験
│   ├── orthogonal_teacher.py
│   ├── low_loop_graph.py
│   └── replica_overlap.py
├── agd/                 # 交互勾配降下法
│   ├── train_parallel.py
│   └── train_sequential.py
├── analysis/            # 分析ツール
└── results/             # 実験結果
```

### terao_gd/ ディレクトリ
```
terao_gd/
├── gd.py                # 基本AGD（Q_Y vs α プロット）
├── gd_noisy.py          # ノイズ付き観測
├── gd_order_params.py   # オーダーパラメータ解析
└── gd_varying_M.py      # ランクM変化実験
```

### smf/ フレームワーク
```
smf/
├── cli.py              # コマンドラインエントリ
├── core/               # コア機能
│   ├── config.py       # 設定システム
│   ├── device.py       # GPU/CPU 検出
│   ├── experiment.py   # 実験ランナー
│   ├── checkpoint.py   # チェックポイント
│   ├── llm_advisor.py  # 自然言語設定
│   └── progress.py     # 進捗表示
└── modules/            # プラグイン可能モジュール
    ├── algorithms/     # bigamp, agd
    ├── graphs/         # random, dinic, low_loop
    ├── teachers/       # standard, orthogonal
    ├── metrics/        # Q_Y, overlap
    └── outputs/        # plotting, storage
```

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
| `Q_W`, `Q_X` | [-1, 1] | 因子コサイン類似度 |

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
