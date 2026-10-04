# mergekit-cabs

**CABS**（ICML 2025, [arXiv:2503.01874](https://arxiv.org/abs/2503.01874)）および
**CABS+**（[arXiv:2608.12842](https://arxiv.org/abs/2608.12842)）モデルマージ手法を
[mergekit](https://github.com/arcee-ai/mergekit) に統合する実装です。

> CABS に対応する公式の mergekit ブランチは存在しないため、本パッケージを新規に実装しました。
> 公式リポジトリ [zongzhenyang/CABS](https://github.com/zongzhenyang/CABS) のコードを参照し、
> CABS+ は論文（arXiv:2608.12842, Algorithm 1 / Sec. III）の記述に基づいて再現しています。

## 概要

### CABS（Conflict-Aware and Balanced Sparsification）

タスクベクトル `τ_i = W_i − W_base` をマージ前にスパース化する手法で、2 つの構成要素からなります。

| 要素 | 内容 |
|------|------|
| **CA** (Conflict-Aware Sparsification) | タスクベクトルを**逐次**剪定し、先行ベクトルが確保した位置を後続ベクトルの選択から除外（`τ_B ← τ_B ⊙ (1 − mask_A)`）。これによりタスク間のパラメータ重複（コンフリクト）を排除する |
| **BS** (Balanced Sparsification) | **n:m ブロック剪定**：全パラメータを m 個ずつのブロックに分割し、各ブロックで絶対値上位 n 個のみ保持。保持重みが層全体に均等に分布し、マージ後もモデルは密（dense）のまま |

最終的なマージ式は `W_final = W_base + Σ_i λ_i · τ̃_i` です。
CABS は Open LLM Leaderboard（2025-02-19 時点）で 8B 以下の上位 4 位を独占した
`qwen2.5-7b-cabs` シリーズの手法です。

### CABS+（CABS + Adaptive Weight Allocation）

CABS の論文誌拡張版（2026-08 公開）。スケーリング係数 λ の決定を、グリッドサーチから
**CMA-ES（共分散行列適応進行戦略）ベースの勾配不要最適化 AWA** に置き換えたものです。

- **非対称適応度関数**：各タスクの損失を基準（λ=1）からの相対変化 `Δ_t` に正規化し、
  悪化には α=100、改善には β=1 の重みを付与（`F(λ) = Σ_t f_t`）。
  損失スケールの大きいタスクに最適化が支配される問題を防ぐ
- **ボックス制約**：`λ ∈ [0.1, 2]` への射影（論文 Eq. 3-4）
- **対数重み付き平均更新**：`w_i ∝ ln(μ + 0.5) − ln(i)`（上位 μ = K/2 個体）
- **ステップサイズ σ と共分散行列 C の適応更新**（論文 Eq. 13-17）
- メモリ使用量は AdaMerging の 25% 未満、WUDI-Merging 比 ~4 倍高速と報告
- さらに **RSS (Relative Synergy Score)** というマージ可能性指標も提案

## インストール

```bash
pip install -e .            # コア（torch のみ）
pip install -e ".[full]"    # モデル読み込み / AWA / mergekit レシピ実行に必要
```

`import mergekit_cabs` するだけで `cabs` / `cabs_plus` が mergekit のレジストリに登録されます。

## 完全な使用ステップ

Step 0 から順に実行すれば、インストール直後からマージ済みモデルの検証まで完結します。

### Step 0: 前提条件

| 項目 | 要求 | 備考 |
|------|------|------|
| Python | 3.9 以上 | 開発・検証は 3.13 で実施 |
| PyTorch | 2.0 以上 | CUDA 版推奨（CPU のみでも動作可） |
| RAM | モデルサイズの 2〜3 倍 | 7B fp16 なら実質 16 GB 以上推奨。0.5B〜1B なら 8 GB でも可 |
| ディスク | モデルサイズ × (入力数 + 1) | 入出力はすべて safetensors |
| GPU | 任意 | `--cuda` 指定時のみ使用。省略時は CPU で完結 |

ゲート付きモデル（Llama 系など）を使う場合は事前に `huggingface-cli login` を実行しておきます。

### Step 1: インストールと動作確認

```bash
git clone https://github.com/CroudGoat/mergekit-cabs.git
cd mergekit-cabs
pip install -e ".[full]"     # mergekit / transformers / safetensors 等を含む
pip install -e ".[dev]"      # テスト実行用（pytest）
pytest tests/ -q             # 30 tests が pass すれば環境 OK
```

### Step 2: モデルの準備

```bash
# 例: ベース 1 + ファインチューン 2 を用意
huggingface-cli download mistralai/Mistral-7B-v0.1
huggingface-cli download WildMarcoroni-Variant1-7B
huggingface-cli download WestSeverus-7B-DPO-v2
```

`recipes/cabs_2model.yml` のモデル名・`n` / `m` / `weight` を環境に合わせて編集します。
レシピの `models` 列の**先頭**が CA スパース化の最優先（最も保護される）タスクベクトルになります。

### Step 3: CABS でマージする

```bash
mergekit-cabs-yaml recipes/cabs_2model.yml ./output --cuda --lazy-unpickle
```

- `--cuda` を外せば CPU のみで実行（メモリが厳しいときは `--low-cpu-memory` を追加）
- `--write-model-card` でマージ設定を焼き込んだモデルカードを同時生成
- `--trust-remote-code` はカスタムアーキテクチャのモデルのときのみ付与

完了すると `./output/` に safetensors 形式のマージ済みモデルとトークナイザが出力されます。

### Step 4: マージ結果を検証する

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./output"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

inputs = tok("The meaning of life is", return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

エラーなく生成できれば最低限の整合性は確保できています。品質の比較は元のファインチューン群
や `task_arithmetic` マージとのベンチマーク（Perplexity、lm-evaluation-harness 等）で行ってください。

### Step 5: CABS+（AWA）でスケーリング係数 λ を最適化する

**(1)** タスクごとのキャリブレーションデータを JSONL で用意（1 行 1 例文）:

```bash
mkdir -p calibration
cat > calibration/task_a.jsonl <<'EOF'
{"text": "Question: ...? Answer: ..."}
{"text": "..."}
EOF
```

**(2)** 設定 JSON を作成（雛形: `recipes/awa_config.example.json`）:

| キー | 内容 |
|------|------|
| `base_model` / `models` | ベースとファインチューン群（モデル ID またはローカルパス） |
| `pruning` | `method`（`nm` / `magnitude`）, `n`, `m`, `consensus` |
| `awa` | `n_generations`（既定 30）, `popsize`（null = 自動設定）, `bounds`, `sigma0`, `alpha`, `beta`, `seed` |
| `tasks` | `name`, `data`（JSONL パス）, `max_examples`, `max_len` |
| `device` / `dtype` / `batch_size` | 実行環境に合わせて指定 |
| `output_dir` / `save_merged` | 出力先 / マージ済みモデルの同時保存 |

**(3)** 実行:

```bash
mergekit-cabs awa --config awa_config.json
```

**(4)** 出力物:

- `awa_result.json` — 最適係数 λ*、世代ごとの履歴、タスク別損失
- `cabs_plus_recipe.yml` — λ* を書き込んだレシピ。Step 3 と同じコマンドでそのままマージ可能
- `save_merged: true` の場合はマージ済みモデルも同時保存

### Step 6: うまくいかないとき

| 症状 | 対処 |
|------|------|
| `Unknown merge method: cabs` | 素の `mergekit-yaml` ではなく **`mergekit-cabs-yaml`** を使う（メソッド登録は本パッケージの import 時に行われる） |
| CUDA out of memory | `--low-cpu-memory` を追加 / `m` を小さくしてスパース化を強める / `dtype: float16` を確認 |
| CPU で遅い・メモリ不足 | `--lazy-unpickle` を必ず付ける。まず 0.5B〜1B 級で試す |
| ゲート付きモデルで 401 エラー | `huggingface-cli login` 後に再実行 |
| マージ結果の品質が低い | `consensus: ties` を明示 / `normalize: true` を試す / `weight` を 1.0 近傍に |

## 使い方

### 1. mergekit レシピで使う（推奨）

```yaml
# recipe.yml
models:
  - model: WildMarcoroni-Variant1-7B      # リスト先頭 = CA の最優先
    parameters:
      weight: 1.2                         # λ_A
      n: 64                               # BS: ブロック内保持数
      m: 256                              # BS: ブロックサイズ
  - model: WestSeverus-7B-DPO-v2
    parameters:
      weight: 1.2                         # λ_B
      n: 64
      m: 256

merge_method: cabs                          # または cabs_plus
base_model: mistralai/Mistral-7B-v0.1
dtype: float16
```

```bash
# cabs / cabs_plus を登録済みの mergekit-yaml ラッパー
mergekit-cabs-yaml recipe.yml ./output --cuda --lazy-unpickle
```

`prune_method: magnitude` を指定した場合は `density`（保持率）で剪定します。
`cabs_plus` は既定で `consensus: ties`（重複領域の多数決符号選出）が有効です。

### 2. CABS+ の AWA で λ* を求める

```bash
mergekit-cabs awa --config recipes/awa_config.example.json
```

設定 JSON には `base_model` / `models` / タスクごとのキャリブレーションデータ
（JSONL、1 行 1 例文 `{"text": "..."}`）を記述します。実行すると：

- `awa_result.json` — 最適係数 λ*、世代ごとの履歴、タスク別損失
- `cabs_plus_recipe.yml` — λ* を書き込んだすぐに使える mergekit レシピ
- `--save_merged`（設定キー `save_merged`）でマージ済みモデルも保存

メモリ戦略：ベース重みと（スパースな）剪定済みタスクベクトルは CPU に置き、
候補ごとにテンソル単位でストリーミング適用するため、GPU メモリは推論時相当に抑えられます
（論文の AdaMerging に対する優位点を再現）。

### 3. ステップ実行 CLI

```bash
# タスクベクトル抽出 (τ = W_ft − W_base)
mergekit-cabs extract --base-model BASE --models FT_A FT_B --output-dir tv/

# CA + BS スパース化
mergekit-cabs prune --task-vectors tv/tv_FT_A tv/tv_FT_B \
    --prune-method nm --n 64 --m 256 --output-dir pruned/

# 重み固定でマージ
mergekit-cabs merge --base-model BASE --task-vectors pruned/pruned_0 pruned/pruned_1 \
    --weights 1.2 1.2 --output-dir merged/
```

### 4. Python API

```python
import torch
import mergekit_cabs  # cabs / cabs_plus を登録
from mergekit.merge_methods.api import merge_tensors

merged = merge_tensors(
    [base_tensor, ft_a_tensor, ft_b_tensor],
    method="cabs_plus",
    base_index=0,
    parameters={"weight": [1.15, 0.95], "n": 64, "m": 256},
)
```

## パラメータ一覧

| パラメータ | スコープ | 既定値 | 説明 |
|-----------|---------|--------|------|
| `weight` | モデル毎 | 1.0 | スケーリング係数 λ_i（cabs_plus では AWA で最適化） |
| `n` / `m` | モデル毎 | 64 / 256 | BS のブロック剪定設定（保持率 = n/m） |
| `density` | モデル毎 | なし | `prune_method: magnitude` 時の保持率 |
| `prune_method` | モデル毎 | `nm` | `nm`（BS）または `magnitude` |
| `consensus` | 共通 | cabs: `none` / cabs_plus: `ties` | 重複領域の解決（`ties` = TIES 風多数決符号） |
| `normalize` | 共通 | false | 混合デルタを位置ごとの確保数で割る |

## 実装上の注意（公式実装との対応）

- **n:m（BS）**：公式 `n_m_block_pruning` と同様に、m で割り切れない末尾ブロックは 0 にする
- **2 ベクトル + n < m/2**：公式の `param2 *= (1 - mask1)` → `nm(...)` と一致
- **2 ベクトル + n ≥ m/2**：空き領域をすべて確保し、残りクォータを確保済み領域から
  大きさ順に補填（公式の `mask_half` 分岐と一致）
- **magnitude**：公式 `block_random_pruning` の逐次動作（第 2 ベクトルは空き領域から
  `(1−s)` 分保持）と一致
- **k ≥ 3 ベクトルでクォータ合計が 1 を超える場合**：CABS+ 論文 Sec. III-A
  （"Minimizing overlap under low sparsity"）に従い、重複領域からの補填 +
  （`consensus: ties` 時）多数決符号選出で解決
- **AWA**：C = I、σ₀ = 0.05、λ₀ = 1、境界 [0.1, 2]、K 個体/世代、μ = K/2、
  対数重み付き更新、Eq. 13-17 の経路更新を論文どおり実装
  （c_σ, d_σ, c_c, c_1, c_μ は Hansen の標準定数）

## テスト

```bash
pip install -e ".[dev]"
pytest tests/
```

30 件のテストが含まれます（BS マスクの正しさ、CA の非重複保証、公式実装との一致、
TIES 合意、CMA-ES の収束・境界・σ/C の整合、非対称適応度、および実 mergekit を用いた
小型 Llama 3 個のエンドツーエンドマージ）。

## ライセンス

MIT。元論文・公式実装：CABS (ICML 2025) は Zongzhen Yang ら、
CABS+ は Yuchen Liu, Zongzhen Yang, Binhang Qi, Hailong Sun, Xiang Gao（北航大学）によるものです。
