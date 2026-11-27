"""Build the HTML record of the run in which Claude itself acted as the agent.

Reads runs/claude_agent/{actions.json,transcript.json,report.json,img/},
the comparison runs and the same-edit-space reference, and writes one
self-contained HTML file.

    python scripts/make_claude_agent_report.py --out reports/claude_agent_named_2026-11-27.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_run_report import CSS  # noqa: E402

from adl.env.environment import _diverging, _grid, _png  # noqa: E402

EVAL_RE = re.compile(r"concept AUROC=([0-9.]+); base accuracy=([0-9.]+)")


def esc(s) -> str:
    return html.escape(str(s))


def img_tag(path: str | Path, alt: str, width: int | None = None) -> str:
    b = base64.b64encode(Path(path).read_bytes()).decode()
    w = f' style="max-width:{width}px"' if width else ""
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b}"{w}>'


def png_tag(png: bytes, alt: str, width: int | None = None) -> str:
    b = base64.b64encode(png).decode()
    w = f' style="max-width:{width}px"' if width else ""
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b}"{w}>'


def fig(inner: str, cap: str) -> str:
    return f"<figure>{inner}<figcaption>{cap}</figcaption></figure>"


def f3(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def kernel_png(actions: list[dict]) -> bytes:
    """conv1[8] (/), conv1[9] (\\) as luminance maps; conv2[16] slices from conv1[8], conv1[9]."""
    k8 = np.array(next(s for s in actions if s["tool"] == "add_channel" and s["args"]["layer"] == "conv1")["args"]["kernel"]).sum(0)
    k9 = np.array([s for s in actions if s["tool"] == "add_channel" and s["args"]["layer"] == "conv1"][1]["args"]["kernel"]).sum(0)
    c2 = np.array(next(s for s in actions if s["tool"] == "add_channel" and s["args"]["layer"] == "conv2")["args"]["kernel"])[8]
    inh = np.array(next(s for s in actions if s["tool"] == "connect" and s["args"]["dst"] == "conv2")["args"]["values"])
    t = lambda m: np.kron(_diverging(np.asarray(m, float)), np.ones((1, 1, 1)))  # noqa: E731
    tiles = [t(k8), t(k9), np.pad(t(c2), ((1, 1), (1, 1), (0, 0)), constant_values=1),
             np.pad(t(inh * 0.999), ((1, 1), (1, 1), (0, 0)), constant_values=1)]
    return _png(_grid(tiles, ncol=4, pad=1), 28)


def cycles_table(tr: list[dict]) -> str:
    """Group the transcript into hypothesis -> edit -> evaluate -> (observations) -> decide."""
    rows, cur, n = [], None, 0
    for s in tr:
        t = s["tool"]
        if t == "hypothesis":
            cur = {"h": s["args"], "edits": [], "eval": None, "obs": [], "steps": [s["step"]]}
        elif cur is not None and t in {"add_channel", "connect", "set_weights"}:
            cur["edits"].append(s)
            cur["steps"].append(s["step"])
        elif cur is not None and t == "evaluate":
            cur["eval"] = s["text"]
            cur["steps"].append(s["step"])
        elif cur is not None and t in {"channel_stats", "feature_map", "top_activations"}:
            first = s["text"].split("\n")
            o = first[0] if t != "channel_stats" else first[1] if len(first) > 1 else first[0]
            cur["obs"].append(o if len(o) < 110 else o[:110] + " …")
        elif cur is not None and t == "decide":
            n += 1
            cur["steps"].append(s["step"])
            ed = []
            for e in cur["edits"]:
                a = dict(e["args"])
                for k in ("kernel", "values"):
                    if k in a and isinstance(a[k], list) and len(json.dumps(a[k])) > 80:
                        a[k] = "…(配列、下の図を参照)"
                err = e["text"].startswith("ERROR")
                ed.append(f'<div class="args">{esc(e["tool"])} {esc(json.dumps(a, ensure_ascii=False))}'
                          f'{" <b style=color:var(--bad)>→ エラー（未適用）</b>" if err else ""}</div>')
            m = EVAL_RE.findall(cur["eval"] or "")
            ev = f"dev AUROC {float(m[-1][0]):.4f}・元の精度 {float(m[-1][1]):.3f}" if m else "—"
            rows.append(
                f"<tr><td class=num>{n}<div class=small>step {cur['steps'][0]}–{cur['steps'][-1]}</div></td>"
                f"<td><b>仮説：</b>{esc(cur['h']['statement'])}<div class=small><b>正例の根拠：</b>{esc(cur['h']['positive_evidence'])}"
                f"<br><b>負例の根拠：</b>{esc(cur['h']['negative_evidence'])}</div></td>"
                f"<td>{esc(cur['h']['edit_plan'])}{''.join(ed)}</td>"
                f"<td>{esc(cur['h']['prediction'])}</td>"
                f"<td>{ev}</td>"
                f"<td>{''.join(f'<div class=small>{esc(o)}</div>' for o in cur['obs'])}</td>"
                f"<td><b>{'採用' if s['args']['verdict'] == 'accept' else '棄却'}</b><div class=small>{esc(s['args']['reason'])}</div></td></tr>")
            cur = None
    return ("<div class=tablewrap><table class=turns><thead><tr><th style=width:58px>#</th><th style=width:24%>仮説と根拠</th>"
            "<th style=width:17%>編集</th><th style=width:13%>予測</th><th style=width:10%>検証（dev）</th>"
            "<th style=width:14%>編集後の観察</th><th>判断</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def step_table(tr: list[dict]) -> str:
    rows = []
    for s in tr:
        a = dict(s["args"])
        for k in ("kernel", "values", "code"):
            if k in a and len(json.dumps(a[k], ensure_ascii=False)) > 90:
                a[k] = "…"
        for k in ("statement", "positive_evidence", "negative_evidence", "edit_plan", "prediction", "reason", "text"):
            if k in a and len(a[k]) > 60:
                a[k] = a[k][:60] + "…"
        res = s["text"]
        if s["tool"] == "run_sandbox":
            res = res.split("\n", 0)[0]
        res = res if len(res) < 700 else res[:700] + " …"
        cls = " class=ph-error" if res.startswith("ERROR") else ""
        rows.append(f"<tr{cls}><td class=num>{s['step']}</td><td><span class=chip>{esc(s['tool'])}</span>"
                    f"<div class=args>{esc(json.dumps(a, ensure_ascii=False))}</div></td>"
                    f"<td class=why>{esc(s['rationale'])}</td><td class=res><pre style='white-space:pre-wrap;margin:0;font-size:12px'>{esc(res)}</pre></td></tr>")
    return ("<div class=tablewrap><table class=turns><thead><tr><th style=width:48px>step</th><th style=width:26%>ツールと引数</th>"
            "<th style=width:20%>理由</th><th>結果</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/claude_agent")
    ap.add_argument("--analysis", default="reports/claude_agent_named_2026-11-27_analysis.html")
    ap.add_argument("--out", default="reports/claude_agent_named_2026-11-27.html")
    a = ap.parse_args()
    run = Path(a.run)
    rep = load(run / "report.json")
    tr = load(run / "transcript.json")
    actions = load(run / "actions.json")
    es = load(sorted(glob.glob("runs/named0_es_full_s0_*/report.json"))[-1])
    bph = load(sorted(glob.glob("runs/named0_backprop_head_full_s0_*/report.json"))[-1])
    bpc = load(sorted(glob.glob("runs/named0_backprop_channel_full_s0_*/report.json"))[-1])
    mch = load(sorted(glob.glob("runs/named0_matched_channel_full_s0_*/report.json"))[-1])
    ss = load("runs/same_space/report_lr1e-2.json")
    ssbp = load("runs/same_space_bp/report.json")
    tmpl = Path(a.analysis).read_text(encoding="utf-8")

    def row(name, r, note="", edits="—", evals="—"):
        return (f"<tr><td>{name}</td><td class=num>{f3(r.get('auroc_heldout'))}</td><td class=num>{f3(r.get('auroc_shift'))}</td>"
                f"<td class=num>{f3(r.get('auroc_dev'))}</td><td class=num>{f3(r.get('base_acc_final'))}</td>"
                f"<td class=num>{edits}</td><td class=num>{evals}</td><td class=small>{note}</td></tr>")
    ss_es = ss["es"]
    es_rows = "".join(row(f"ES・Claudeと同じ編集空間（seed {r['seed']}）", r, f"新しい経路（{ss['dim']}パラメータ）をランダム初期値から。dev で順伝播200回",
                          evals="200") for r in ss_es)
    bp_ss = ssbp["backprop"]
    bp_mean = {k: float(np.mean([r[k] for r in bp_ss])) for k in ("auroc_heldout", "auroc_shift", "auroc_dev", "base_acc_final")}
    bu = rep["budget_used"]
    cmp_table = (
        "<table class=cmp><thead><tr><th>方法</th><th class=num>heldout</th><th class=num>shift</th><th class=num>dev</th>"
        "<th class=num>元の精度</th><th class=num>編集</th><th class=num>評価・順伝播</th><th>備考</th></tr></thead><tbody>"
        + row("<b>Claude が Agent（今回）</b>", rep, "勾配なし。手設計の検出器を7回の採用で組み込んだ",
              edits=f"8（採用7・エラー1）", evals=f"評価{bu['evaluations']}・順伝播{bu['forward_passes']}")
        + row("ES（既定、新しい conv3 チャネル1本）", es, "クラウドで今回と同じベースに対して再実行。dev で順伝播200回", evals="200")
        + es_rows
        + row("VLM bonsai-2:9b（ガードなし、PC）", {"auroc_heldout": 0.435, "auroc_shift": 0.243, "auroc_dev": 0.468, "base_acc_final": 0.678},
              "前回のパイロット。PC上の別のベースモデル", edits="15", evals="評価13")
        + row("VLM bonsai-2:9b（ガードあり、PC）", {"auroc_heldout": 0.500, "auroc_shift": 0.500, "auroc_dev": 0.500, "base_acc_final": 1.0},
              "7回とも棄却。PC上の別のベースモデル", edits="7（採用0）", evals="評価7")
        + row("逆伝播・概念ヘッドのみ（参照）", bph, "既存32チャネルの線形結合。勾配を使う")
        + row("逆伝播・新しい conv3 チャネル1本（参照）", bpc, "ES と同じ編集空間。この設定では学習が進まなかった")
        + row("逆伝播・Claudeと同じ編集空間（参照、3 seed 平均）", bp_mean, "lr 1e-2・1e-3 とも、3 seed すべて当て推量のまま（死んだ ReLU と考えられる。未調整）")
        + row("matched filter（新しい conv3 チャネル）", mch, "正例平均−負例平均の方向")
        + "</tbody></table>")

    show = {s["step"]: s for s in tr}
    def im(step, cap, w=None):
        p = show[step]["images"][0]
        return fig(img_tag(p, cap, w), cap)

    imgs = {
        "pos": im(2, "step 2：train の正例（概念あり）16枚", 520),
        "neg": im(3, "step 3：train の負例16枚", 520),
        "k1": im(4, "step 4：既存 conv1 の8カーネル（RGB）", 520),
        "fm1": im(16, "step 16：追加した conv1[8]（/ 線検出器）の反応。左が画像、右が反応（赤）。上2行が正例、下2行が負例", 520),
        "top2": im(22, "step 22：conv2[16] を最も強く反応させた train の12パッチ（すべて正例）", 520),
        "miss": im(35, "step 35：bias -1.0 のとき取りこぼした dev の正例。すべて明るい図形の上の / の棒", 520),
        "fp": im(47, "step 47：bias -0.6 にした後、誤反応した負例（上2行）とまだ弱い正例（下2行）", 520),
        "kern": fig(png_tag(kernel_png(actions), "追加したカーネル", 520),
                    "追加した重み（赤＝正、青＝負）。左から conv1[8]（/、RGB合計）、conv1[9]（\\、RGB合計）、"
                    "conv2[16] の conv1[8] からの3×3（/ 方向の共線連結）、conv2[16] の conv1[9] からの3×3（全体 −1 の抑制）"),
    }
    tiles = (
        f"<div class=tile><div class=k>heldout AUROC（封印）</div><div class=v>{rep['auroc_heldout']:.4f}</div><div class=d>ES 0.845、VLM 0.500</div></div>"
        f"<div class=tile><div class=k>shift AUROC（封印）</div><div class=v>{rep['auroc_shift']:.4f}</div><div class=d>ES 0.938、VLM 0.500</div></div>"
        f"<div class=tile><div class=k>元の3クラス精度</div><div class=v>{rep['base_acc_initial']:.3f} → {rep['base_acc_final']:.3f}</div><div class=d>既存の重みは1つも変えていない</div></div>"
        f"<div class=tile><div class=k>dev 評価の回数</div><div class=v>{bu['evaluations']}</div><div class=d>順伝播 {bu['forward_passes']}／200、勾配 0</div></div>")
    body = tmpl.format(cmp_table=cmp_table, cycles=cycles_table(tr), steps=step_table(tr), tiles=tiles,
                       diff=esc(rep["diff_from_initial"]), audit=esc(rep["audit_flags"]),
                       guard=esc(rep["guard_events"]), **imgs)
    doc = (f"<!DOCTYPE html><html lang=ja><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Claude 自身が Agent となった E0 実験記録（2026-11-27）</title><style>{CSS}"
           f".imgs img,figure img{{max-width:100%;image-rendering:pixelated;border-radius:6px}}figure{{margin:0 0 12px}}"
           f".warn{{border-left:4px solid var(--s2);background:var(--panel);padding:12px 16px;border-radius:6px}}</style></head>"
           f"<body><main>{body}</main></body></html>")
    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote", a.out, len(doc) // 1024, "KB")


if __name__ == "__main__":
    main()
