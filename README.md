# Agent-Driven Learning (ADL)

LLM/VLM を基盤とする Agent が、CNN の内部表現を観察・分析・直接編集することで、誤差逆伝播に代わる学習を担えるかを検証する研究用コードです。

- 研究方針：[docs/research_plan_v1.md](docs/research_plan_v1.md)
- 先行研究と成立可能性の報告：[docs/feasibility_report_v1.md](docs/feasibility_report_v1.md)
- Stage 0 の比較手法の予備結果：[docs/stage0_baselines.md](docs/stage0_baselines.md)

## 中心の問い

Agent は、勾配を使わずに、**自分も名前を知らない特徴**をデータから見つけ、凍結した CNN に永続的に獲得させられるか。

## 構成

| パス | 内容 |
|---|---|
| `adl/data/shapes.py` | 合成データ。基礎3クラス＋新概念（`named`：向きのある棒、`glyph`：名前のない色パターン） |
| `adl/models/cnn.py` | 編集しやすい小型CNN（BNなし。チャネルの追加・削除が可能） |
| `adl/train_base.py` | ベースCNNの事前学習（研究ルール上、逆伝播を使ってよい唯一の場所） |
| `adl/env/environment.py` | Agent 用ツール群：観察（画像・カーネル・特徴マップ・top活性）、因果介入、NumPy サンドボックス、永続編集、評価、checkpoint/rollback、メモ |
| `adl/env/sandbox.py` | サンドボックス：torch 不可。最小二乗・逆行列・反復更新ループなどの「当てはめ」を検知して記録（条件 H では禁止） |
| `adl/env/budget.py` | 予算（順伝播・評価・重みの読み出し・編集）と ledger（全操作の記録） |
| `adl/baselines/methods.py` | 比較手法：matched filter、ランダム編集、ES、逆伝播（参照用） |
| `adl/agent/` | OpenAI 互換 API（既定はローカル Ollama）で VLM を動かす観察→推論→操作→検証ループ |
| `scripts/run_experiment.py` | 1条件を実行して JSON レポートを出力 |

研究ルールの実装：
- 対象モデルのパラメータは勾配を持たない（全ツール呼び出し後に検査）。
- `heldout` と `shift` は封印されており、どのツールからも読めない。Agent が `submit` した後にだけ採点される。
- 条件 `no_semantics`（画像を見せない）と `no_observation`（内部を見せない）で、情報源のアブレーションができる。

## 使い方（WSL2 + Docker Desktop、NVIDIA GPU）

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3.8:27b   # VRAM に合わせてモデルを選ぶ
docker compose run --rm adl pytest -q

# 比較手法（LLM 不要）
docker compose run --rm adl python scripts/run_experiment.py --concept glyph --method es

# Agent
docker compose run --rm -e ADL_LLM_MODEL=qwen3.8:27b adl \
  python scripts/run_experiment.py --concept named --method agent --condition full
```

Docker を使わない場合：`pip install -e .[dev]` の後、`ollama serve` を起動し、`ADL_LLM_BASE_URL=http://localhost:11434/v1` を設定する。

結果は `runs/<タグ>/` に `report.json`、`ledger.jsonl`、`transcript.json` として保存される。
