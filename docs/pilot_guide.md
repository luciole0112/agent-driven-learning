# パイロット実行手順（E0、RTX 4060 Laptop 8GB）

## 使うモデル

**bonsai-2:9b**（Ollama、約3.2GB、Apache 2.0）を既定にした。

- 選定理由：Ollama の一覧で「画像入力」「ツール呼び出し」「思考モード」の3つすべてに対応し、8GB の VRAM に、16K の文脈と画像を載せる余裕を残して収まる。
- 代替候補：`qwen3-vl:4b`（約3.3GB、画像入力に対応。ツール呼び出しの対応は未確認）、`qwen3-vl:8b`（約6.1GB、8GB では文脈がかなり厳しい）。`scripts/check_llm.py` で動作を確かめてから使うこと。
- **研究上の注意**：先行研究では、30B 未満のモデルは重みの編集に失敗しやすい。9B モデルで失敗しても、「ADL は成立しない」の根拠にはならない。今回のパイロットの目的は、配線の確認と、小型モデルでどこまでできるかの把握である。

## 0. 前提の確認（初回のみ）

Windows 側に NVIDIA ドライバが入っていて、Docker Desktop が WSL2 バックエンドで動いていること。WSL2 のターミナルで、次のコマンドを実行する。

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

GPU 名（RTX 4060 Laptop）が表示されれば OK。

## 1. コードの取得

```bash
git clone https://github.com/luciole0112/agent-driven-learning.git   # 2回目以降は git pull
cd agent-driven-learning
```

## 2. VLM の準備

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull bonsai-2:9b
```

## 3. 実験コンテナのビルドとテスト

```bash
docker compose build adl          # 初回は PyTorch イメージの取得に時間がかかる
docker compose run --rm adl pytest -q
```

13 件が passed になればよい。

## 4. VLM の動作確認

```bash
docker compose run --rm adl python scripts/check_llm.py
```

`chat`、`image input`、`tool calling` の3つが PASS になること。FAIL があれば、その出力を共有してほしい（モデルを替える）。

## 5. パイロット本体（名前のある棒、E0）

まず比較用の ES を実行する（LLM 不要、1分程度）。

```bash
docker compose run --rm adl python scripts/run_experiment.py --concept named --method es
```

次に Agent を実行する（数十分〜1時間程度の見込み。モデルの速度による）。

```bash
docker compose run --rm adl python scripts/run_experiment.py \
  --concept named --method agent --condition full --max-turns 60
```

途中経過は `[turn N] ツール名(...) -> 結果` の形で表示される。止めたいときは Ctrl+C。

## 6. 結果の共有

`runs/` の下に、実行ごとのフォルダ（例：`named0_agent_full_s0_...`）ができる。次の3ファイルを共有してほしい。

- `report.json`：封印データでの採点結果
- `ledger.jsonl`：Agent の全操作と理由
- `transcript.json`：VLM とのやり取り（画像は省略）

フォルダごと zip にしてチャットにアップロードするのが簡単。

## よくある問題

| 症状 | 対処 |
|---|---|
| `check_llm.py` が接続できない | `docker compose ps` で ollama が動いているか確認する |
| VRAM 不足（Ollama が遅い、落ちる） | `OLLAMA_CONTEXT_LENGTH=8192 docker compose up -d ollama` で文脈を短くする |
| Agent がツールを呼ばずに文章だけ返す | 小型モデルでよく起きる。ledger を共有してもらえれば、プロンプトを調整する |
