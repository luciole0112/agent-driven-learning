"""Build a self-contained HTML experiment record from one agent run.

Reads ``report.json`` and ``ledger.jsonl`` of an agent run (and the
``report.json`` of comparison runs), regenerates the training examples the
agent looked at, and writes one HTML file with: summary, the task, the
AUROC / base-accuracy trajectory, a turn-by-turn table, the comparison and an
optional hand-written analysis fragment.

Example
-------
python scripts/make_run_report.py --run runs/named0_agent_full_s0_1795776001 \
    --compare runs/named0_es_full_s0_1795775935 \
    --analysis reports/pilot_E0_2026-11-27_analysis.html \
    --title "E0 パイロット実験記録（2026-11-27）" --out reports/pilot_E0_2026-11-27.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

import numpy as np

from adl.data.shapes import ConceptSpec, make_splits
from adl.env.environment import _grid, _png

OBSERVE = {"inspect_model", "get_tensor", "render_kernels", "show_examples", "feature_map",
           "top_activations", "channel_stats", "capture_activations", "probe_ablate", "run_sandbox"}
HYPOTHESIS = {"note", "hypothesis"}
EDIT = {"set_weights", "add_to_weights", "scale_channel", "lowrank_update", "add_channel",
        "remove_channel", "connect"}
VERIFY = {"evaluate"}
DECIDE = {"decide", "rollback", "checkpoint", "diff", "submit"}
PHASE_JA = {"observe": "観察", "hypothesis": "仮説・メモ", "edit": "編集", "verify": "評価",
            "decide": "採否・管理", "error": "エラー", "blocked": "手順で阻止"}
EVAL_RE = re.compile(r"concept AUROC=([0-9.]+); base accuracy=([0-9.]+)")


def phase(e: dict) -> str:
    r = e["result_summary"]
    if r.startswith("BLOCKED"):
        return "blocked"
    if r.startswith("ERROR"):
        return "error"
    t = e["tool"]
    for name, group in (("observe", OBSERVE), ("hypothesis", HYPOTHESIS), ("edit", EDIT),
                        ("verify", VERIFY), ("decide", DECIDE)):
        if t in group:
            return name
    return "observe"


def esc(s) -> str:
    return html.escape(str(s))


def b64png(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def example_images(report: dict, ledger: list[dict]) -> dict[int, tuple[list[int], str]]:
    """Regenerate the train images the agent was shown, per label."""
    if report.get("concept") != "named":
        spec = ConceptSpec(kind="glyph", concept_seed=report.get("concept_seed", 0))
    else:
        spec = ConceptSpec(kind="named")
    train = make_splits(spec, n_train=report.get("n_train", 200))["train"]
    out = {}
    for e in ledger:
        if e["tool"] != "show_examples":
            continue
        m = re.search(r"ids=\[([0-9, ]*)\]", e["result_summary"])
        lab = int(e["args"].get("label", 1))
        if m and lab not in out and m.group(1).strip():
            ids = [int(i) for i in m.group(1).split(",")][:10]
            tiles = [train["x"][i].transpose(1, 2, 0) for i in ids]
            out[lab] = (ids, b64png(_png(_grid(tiles, ncol=len(tiles), pad=2), 3)))
    return out


def chart(ledger: list[dict], n_turns: int) -> str:
    W, H, L, R, T, B = 900, 320, 52, 130, 20, 62
    pw, ph = W - L - R, H - T - B
    lo, hi = 0.4, 1.0
    x = lambda t: L + pw * t / max(1, n_turns - 1)
    y = lambda v: T + ph * (hi - v) / (hi - lo)
    evals = []
    for e in ledger:
        m = EVAL_RE.search(e["result_summary"])
        if e["tool"] == "evaluate" and m:
            evals.append((e["step"], float(m.group(1)), float(m.group(2)), e["args"].get("split", "dev")))
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="評価ごとの概念AUROCと元の3クラス精度の推移" class="chart">']
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    parts.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(0.5):.1f}" y2="{y(0.5):.1f}" class="chance"/>'
                 f'<text x="{L + 6}" y="{y(0.5) - 6:.1f}" class="tick">AUROC 0.5 = 当て推量</text>')
    for t in range(0, n_turns, 10):
        parts.append(f'<text x="{x(t):.1f}" y="{T + ph + 18}" class="tick" text-anchor="middle">{t}</text>')
    parts.append(f'<text x="{L + pw / 2}" y="{H - 6}" class="tick" text-anchor="middle">ターン</text>')
    # edit markers on a strip under the plot
    sy = T + ph + 34
    parts.append(f'<text x="{L - 8}" y="{sy + 4}" class="tick" text-anchor="end">編集</text>')
    for e in ledger:
        p = phase(e)
        if e["tool"] in EDIT:
            cls = "edit-err" if p in ("error", "blocked") else "edit-ok"
            tip = esc(f"ターン{e['step']}: {e['tool']} {json.dumps(e['args'], ensure_ascii=False)[:90]} -> "
                      f"{e['result_summary'][:80]}")
            shape = (f'<rect x="{x(e["step"]) - 4:.1f}" y="{sy - 4}" width="8" height="8" rx="2" class="{cls}"/>'
                     if cls == "edit-ok" else
                     f'<path d="M{x(e["step"]) - 4:.1f},{sy - 4} l8,8 M{x(e["step"]) + 4:.1f},{sy - 4} l-8,8" class="{cls}"/>')
            parts.append(f'<g class="hit"><title>{tip}</title><rect x="{x(e["step"]) - 7:.1f}" y="{sy - 8}" '
                         f'width="14" height="16" fill="transparent"/>{shape}</g>')
    sp_name = "・".join(sorted({sp for *_, sp in evals})) or "train"
    for idx, cls, label in ((1, "s1", f"概念AUROC（{sp_name}）"), (2, "s2", "元の3クラス精度")):
        pts = " ".join(f"{x(t):.1f},{y(v[idx - 1]):.1f}" for t, *v in [(s, a, b) for s, a, b, _ in evals])
        parts.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for s, a, b, sp in evals:
            v = a if idx == 1 else b
            parts.append(f'<g class="hit"><title>ターン{s} evaluate({sp}): AUROC {a:.4f} / 精度 {b:.4f}</title>'
                         f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="9" fill="transparent"/>'
                         f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="4" class="dot {cls}"/></g>')
        if evals:
            s, a, b, _ = evals[-1]
            v = a if idx == 1 else b
            parts.append(f'<text x="{x(s) + 10:.1f}" y="{y(v) + 4:.1f}" class="dlabel">{label} {v:.3f}</text>')
    parts.append("</svg>")
    legend = (f'<div class="legend"><span><i class="sw s1"></i>概念AUROC（Agentが評価した{sp_name}）</span>'
              '<span><i class="sw s2"></i>元の3クラス精度（base_val）</span>'
              '<span><i class="sq ok"></i>編集（成功）</span><span><i class="sq err">×</i>編集（エラー）</span></div>')
    return legend + "".join(parts)


def turn_table(ledger: list[dict]) -> str:
    rows = []
    for e in ledger:
        p = phase(e)
        m = EVAL_RE.search(e["result_summary"])
        metric = f"{float(m.group(1)):.3f} / {float(m.group(2)):.3f}" if m else ""
        args = json.dumps(e["args"], ensure_ascii=False)
        res = e["result_summary"].split("budget remaining")[0]
        rows.append(
            f'<tr class="ph-{p}"><td class="num">{e["step"]}</td>'
            f'<td><span class="chip {p}">{PHASE_JA[p]}</span></td>'
            f'<td><code>{esc(e["tool"])}</code><div class="args">{esc(args[:220])}</div></td>'
            f'<td class="why">{esc(e.get("rationale") or (e["args"].get("text", "") if e["tool"] == "note" else ""))}</td>'
            f'<td class="res">{esc(res[:180])}{"…" if len(res) > 180 else ""}</td>'
            f'<td class="num">{metric}</td></tr>')
    return ('<div class="tablewrap"><table class="turns"><colgroup><col style="width:5%"><col style="width:9%">'
            '<col style="width:22%"><col style="width:28%"><col style="width:26%"><col style="width:10%"></colgroup>'
            '<thead><tr><th>ターン</th><th>段階</th>'
            '<th>ツールと引数</th><th>Agentの理由（原文）</th><th>結果</th><th>AUROC / 精度</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table></div>")


def compare_table(runs: list[tuple[str, dict]]) -> str:
    keys = [("auroc_train", "AUROC train"), ("auroc_dev", "AUROC dev"),
            ("auroc_heldout", "AUROC heldout（封印）"), ("auroc_shift", "AUROC shift（封印・分布ずれ）"),
            ("base_acc_final", "元の3クラス精度（最終）"), ("wall_s", "所要時間（秒）")]
    head = "".join(f"<th>{esc(n)}</th>" for n, _ in runs)
    body = []
    for k, label in keys:
        cells = "".join(f'<td class="num">{r.get(k):.3f}</td>' if isinstance(r.get(k), float)
                        else f'<td class="num">{esc(r.get(k, "—"))}</td>' for _, r in runs)
        body.append(f"<tr><th>{esc(label)}</th>{cells}</tr>")
    bu = "".join(f'<td class="small">{esc(", ".join(f"{k} {v}" for k, v in r.get("budget_used", {}).items()))}</td>'
                 for _, r in runs)
    body.append(f"<tr><th>予算の使用量</th>{bu}</tr>")
    return f'<table class="cmp"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def tool_counts(ledger: list[dict]) -> str:
    cnt: dict[str, list[int]] = {}
    for e in ledger:
        c = cnt.setdefault(e["tool"], [0, 0])
        c[0] += 1
        c[1] += phase(e) in ("error", "blocked")
    rows = "".join(f'<tr><td><code>{esc(t)}</code></td><td class="num">{n}</td><td class="num">{err or ""}</td></tr>'
                   for t, (n, err) in sorted(cnt.items(), key=lambda kv: -kv[1][0]))
    return f'<table class="cnt"><thead><tr><th>ツール</th><th>回数</th><th>うちエラー</th></tr></thead><tbody>{rows}</tbody></table>'


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--panel:#f3f2ef;--text:#0b0b0b;--text2:#52514e;--muted:#7a7974;
--border:#dedcd6;--s1:#2a78d6;--s2:#eb6834;--bad:#c4302f;--ok:#2f7d32}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--panel:#242423;--text:#f0efec;
--text2:#c3c2b7;--muted:#9a998f;--border:#3a3a37;--s1:#3987e5;--s2:#d95926;--bad:#e66767;--ok:#5cb860}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.7 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 24px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:40px 0 12px;padding-top:8px;border-top:1px solid var(--border)}
h3{font-size:16px;margin:20px 0 8px}.sub{color:var(--text2);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--panel);border-radius:10px;padding:14px 16px}.tile .k{color:var(--text2);font-size:13px}
.tile .v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}.tile .d{color:var(--text2);font-size:13px}
.panel{background:var(--panel);border-radius:10px;padding:14px 18px}
.imgs{display:flex;flex-direction:column;gap:12px}.imgs figure{margin:0}.imgs img{max-width:100%;image-rendering:pixelated;border-radius:6px}
figcaption{color:var(--text2);font-size:13px}
.chart{width:100%;height:auto}.grid{stroke:var(--border);stroke-width:1}.chance{stroke:var(--muted);stroke-dasharray:4 4}
.tick{fill:var(--text2);font-size:12px}.dlabel{fill:var(--text);font-size:12px}
.line{fill:none;stroke-width:2}.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}
.dot{stroke:var(--bg);stroke-width:2}.dot.s1{fill:var(--s1)}.dot.s2{fill:var(--s2)}
.edit-ok{fill:var(--text2)}.edit-err{stroke:var(--bad);stroke-width:2.5;fill:none}
.legend{display:flex;flex-wrap:wrap;gap:18px;color:var(--text2);font-size:13px;margin-bottom:6px}
.legend i{display:inline-block;vertical-align:middle;margin-right:6px}
.sw{width:18px;height:3px;border-radius:2px}.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}
.sq{width:9px;height:9px;border-radius:2px;font-style:normal;font-size:12px;line-height:9px;text-align:center}
.sq.ok{background:var(--text2)}.sq.err{color:var(--bad);font-weight:700;width:auto;height:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border-bottom:1px solid var(--border);
padding:6px 8px;text-align:left;vertical-align:top}thead th{color:var(--text2);font-weight:600;position:sticky;top:0;background:var(--bg)}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.small{font-size:12px;color:var(--text2)}
.tablewrap{max-height:760px;overflow:auto;border:1px solid var(--border);border-radius:8px}
.turns{table-layout:fixed}.turns th{white-space:nowrap}.turns td.why{color:var(--text2)}.turns td.res{word-break:break-word}
.args{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--text2);word-break:break-all}
code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;border:1px solid var(--border);white-space:nowrap}
.chip.edit{font-weight:600}.chip.error,.chip.blocked{color:var(--bad);border-color:var(--bad)}
tr.ph-error td,tr.ph-blocked td{background:color-mix(in srgb,var(--bad) 7%,transparent)}
.cmp th{white-space:nowrap}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:760px){.two{grid-template-columns:1fr}}
.note{color:var(--text2);font-size:13px}ul.tight li{margin:4px 0}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--compare-labels", nargs="*", default=[], help="column names for --compare runs")
    ap.add_argument("--analysis", help="HTML fragment with the hand-written analysis")
    ap.add_argument("--title", default="ADL 実験記録")
    ap.add_argument("--model", default="bonsai-2:9b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
    ledger = [json.loads(l) for l in (run / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    comps = [(Path(c).name, json.loads((Path(c) / "report.json").read_text(encoding="utf-8"))) for c in a.compare]
    imgs = example_images(rep, ledger)
    n_turns = len(ledger)
    analysis = Path(a.analysis).read_text(encoding="utf-8") if a.analysis else ""

    def method_name(r):
        return {"es": "ES（進化戦略）", "agent": "Agent"}.get(r.get("method"), r.get("method"))

    labels = a.compare_labels + [None] * len(comps)
    runs = [(f"Agent（{a.model}）", rep)] + [(lab or method_name(r), r) for lab, (_, r) in zip(labels, comps)]
    es = next((r for _, r in comps if r.get("method") == "es"), None)
    tiles = [
        ("AUROC heldout（封印データ）", f"{rep['auroc_heldout']:.3f}",
         f"ES {es['auroc_heldout']:.3f}" if es else "0.5 = 当て推量"),
        ("AUROC shift（分布ずれ）", f"{rep['auroc_shift']:.3f}", f"ES {es['auroc_shift']:.3f}" if es else ""),
        ("元の3クラス精度", f"{rep['base_acc_initial']:.3f} → {rep['base_acc_final']:.3f}",
         f"ES {es['base_acc_initial']:.3f} → {es['base_acc_final']:.3f}" if es else ""),
        ("ターン数 / 所要時間", f"{n_turns} / {rep['wall_s'] / 60:.0f}分",
         "submit なし（上限で終了）" if not any(e["tool"] == "submit" for e in ledger) else "submit あり"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
                         f'<div class="d">{esc(d)}</div></div>' for k, v, d in tiles)
    img_html = "".join(
        f'<figure><img src="{src}" alt="label {lab} の学習画像 ids {ids}"><figcaption>label {lab}'
        f'（{"概念あり" if lab == 1 else "概念なし"}）: train ids {ids}</figcaption></figure>'
        for lab, (ids, src) in sorted(imgs.items(), reverse=True))
    concept_desc = ("45°の向きの棒（色はランダム）。負例には 90°・135°・180° の棒が入る。"
                    "背景にはどの画像にも、元の3クラス（円・四角・三角）の図形が1つずつある。"
                    if rep.get("concept") == "named" else "生成グリフ（色の数は負例と同じ）")
    doc = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(a.title)}</title>
<style>{CSS}</style></head><body><main>
<h1>{esc(a.title)}</h1>
<p class="sub">実行フォルダ <code>{esc(run.name)}</code> ・ 条件 {esc(rep.get('condition'))} ・ 概念 {esc(rep.get('concept'))} ・
VLM {esc(a.model)}（Ollama、RTX 4060 Laptop 8GB） ・ 試行 1 回（seed {esc(rep.get('seed'))}）</p>
<div class="tiles">{tiles_html}</div>

<h2>課題</h2>
<div class="two"><div>
<p>元の3クラス（円・四角・三角）を学習済みの小さなCNNに、新しい2値概念を教える。勾配も最適化器も使わず、
Agent がツールで内部を観察し、重みを直接編集する。<code>head_concept</code> の重みは最初すべて0で、
そのままなら AUROC は 0.5 になる。</p>
<p><b>正解の概念：</b>{esc(concept_desc)} Agent には概念を教えていない。</p>
<p class="note">評価に使う heldout（1000枚）と shift（500枚、明るくノイズの多い背景・大きい棒）は、Agent からは見えない。</p>
</div><div class="imgs">{img_html or '<p class="note">画像は再生成できなかった。</p>'}
<p class="note">Agent が show_examples で実際に見た画像を、同じ乱数シードから再生成した。</p></div></div>

<h2>精度の推移</h2>
<div class="panel">{chart(ledger, n_turns)}</div>
<p class="note">点は evaluate を呼んだターンだけにある。AUROC は Agent が evaluate で使った分割の値。棄却した編集は元に戻されているので、線は累積の変化ではない。点や編集の印にマウスを載せると詳細が出る。</p>

{analysis}

<h2>ターンごとの記録</h2>
<p class="note">ledger.jsonl の全{n_turns}件。「Agentの理由」は Agent がツール呼び出しに付けた rationale をそのまま載せている。</p>
{turn_table(ledger)}

<h2>ツールの使用回数</h2>
<div class="two"><div>{tool_counts(ledger)}</div><div>
<h3>比較</h3>{compare_table(runs)}
<p class="note">ES は dev 分割で 200 回の順伝播を使う。Agent は train 分割しか評価に使っていない。</p></div></div>

<h2>最終モデルの差分（初期との比較）</h2>
<pre class="panel" style="white-space:pre-wrap;font-size:12.5px">{esc(rep.get('diff_from_initial', ''))}</pre>

<h2>記録元</h2>
<ul class="tight note"><li><code>{esc(run)}/report.json</code>・<code>ledger.jsonl</code>・<code>transcript.json</code></li>
{''.join(f'<li><code>{esc(c)}/report.json</code></li>' for c in a.compare)}
<li>このページは <code>scripts/make_run_report.py</code> で生成した。</li></ul>
</main></body></html>"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"wrote {a.out} ({len(doc) // 1024} KB)")


if __name__ == "__main__":
    main()
