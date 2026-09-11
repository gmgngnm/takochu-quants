"""週次レポートの生成.

金曜の引け後にこれを回し、月曜の朝に人間が 10 分レビューして手で発注する。
これが「案A」の完成形。

レポートは単一 HTML ファイル（外部リソースなし）。メール添付でも、
ブラウザで開いても、スマホで見ても崩れないようにしてある。
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RISK_LABELS = {
    "one_time_gain": "一過性利益",
    "one_time_loss": "一過性損失",
    "accounting_change": "会計変更",
    "guidance_cut_risk": "下方修正リスク",
    "demand_weakness": "需要減速",
    "cost_pressure": "コスト圧迫",
    "fx_dependency": "為替依存",
    "customer_concentration": "顧客集中",
    "dilution": "希薄化",
    "litigation": "訴訟",
}


def build_report(
    picks: pd.DataFrame,
    snapshot: pd.DataFrame,
    decision_date: pd.Timestamp,
    exec_date: pd.Timestamp | None,
    previous_picks: pd.DataFrame | None = None,
    capital: float | None = None,
    diagnostics: dict | None = None,
) -> str:
    """週次レポートの HTML を組み立てて返す."""
    detail = picks.merge(snapshot, on="Code", how="left", suffixes=("", "_snap"))

    exits = _exits(picks, previous_picks)
    universe_size = int(snapshot["tradable"].sum()) if "tradable" in snapshot else len(snapshot)

    parts = [
        _head(decision_date),
        _banner(),
        _summary(decision_date, exec_date, len(picks), len(exits), universe_size, diagnostics),
        _exit_section(exits),
        _entry_section(detail, capital),
        _rationale_section(detail),
        _footer(diagnostics),
        "</main></body>",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------- 部品


def _head(decision_date: pd.Timestamp) -> str:
    return f"""<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>週次スクリーニング {decision_date:%Y-%m-%d}</title>
<style>
:root {{
  --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862; --line: #e3e0da;
  --card: #ffffff; --accent: #6b5b4d; --warn-bg: #fdf6e8; --warn-fg: #7a5a1a;
  --warn-line: #e8d5a8; --pos: #2f6b4a; --neg: #9b3b2f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1a1917; --fg: #eeece7; --muted: #a09c94; --line: #35322d;
    --card: #232120; --accent: #c4b3a1; --warn-bg: #2e2617; --warn-fg: #e0c489;
    --warn-line: #4d4127; --pos: #7fbd96; --neg: #d98b7c;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
  line-height: 1.65; font-size: 15px;
}}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.05rem; margin: 36px 0 12px; padding-bottom: 6px;
      border-bottom: 1px solid var(--line); }}
.sub {{ color: var(--muted); font-size: 0.875rem; margin: 0 0 20px; }}
.banner {{ background: var(--warn-bg); color: var(--warn-fg);
  border: 1px solid var(--warn-line); border-radius: 8px;
  padding: 12px 16px; margin: 0 0 24px; font-size: 0.9rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px; margin-bottom: 8px; }}
.stat {{ background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 12px 14px; }}
.stat .label {{ color: var(--muted); font-size: 0.75rem; letter-spacing: 0.02em; }}
.stat .value {{ font-size: 1.25rem; font-variant-numeric: tabular-nums; margin-top: 2px; }}
.scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ padding: 8px 10px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem;
  text-align: right; position: sticky; top: 0; background: var(--bg); }}
th:first-child, td:first-child, th.l, td.l {{ text-align: left; }}
td.num {{ font-variant-numeric: tabular-nums; }}
tbody tr:hover {{ background: var(--card); }}
.pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }}
.tag {{ display: inline-block; background: var(--warn-bg); color: var(--warn-fg);
  border: 1px solid var(--warn-line); border-radius: 4px;
  padding: 1px 6px; margin: 0 3px 3px 0; font-size: 0.72rem; white-space: nowrap; }}
.card {{ background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }}
.card .code {{ font-weight: 600; }}
.card .why {{ color: var(--muted); font-size: 0.875rem; margin-top: 6px;
  white-space: normal; }}
footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 0.8rem; }}
code {{ background: var(--card); border: 1px solid var(--line);
  border-radius: 4px; padding: 1px 5px; font-size: 0.85em; }}
</style></head><body><main>"""


def _banner() -> str:
    return (
        '<div class="banner"><strong>これは売買提案であって発注ではありません。</strong><br>'
        "内容を確認してから手で発注してください。システムが知らない材料"
        "（指数除外・不祥事・TOB・直近の報道など）は必ず自分で確認すること。</div>"
    )


def _summary(
    decision_date: pd.Timestamp,
    exec_date: pd.Timestamp | None,
    n_entry: int,
    n_exit: int,
    universe: int,
    diagnostics: dict | None,
) -> str:
    exec_text = f"{exec_date:%Y-%m-%d}" if exec_date is not None else "翌営業日"
    stats = [
        ("判断日（引け後）", f"{decision_date:%Y-%m-%d}"),
        ("執行予定（寄り）", exec_text),
        ("保有銘柄", str(n_entry)),
        ("入替（売り）", str(n_exit)),
        ("対象ユニバース", f"{universe:,}"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="label">{html.escape(k)}</div>'
        f'<div class="value">{html.escape(v)}</div></div>'
        for k, v in stats
    )
    return (
        f"<h1>週次スクリーニング</h1>"
        f'<p class="sub">生成 {datetime.now():%Y-%m-%d %H:%M} / '
        f"プライム市場・週次リバランス</p>"
        f'<div class="grid">{cards}</div>'
    )


def _exits(picks: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    if previous is None or previous.empty:
        return pd.DataFrame()
    held = set(previous["Code"].astype(str))
    keep = set(picks["Code"].astype(str))
    gone = held - keep
    return previous[previous["Code"].astype(str).isin(gone)].copy()


def _exit_section(exits: pd.DataFrame) -> str:
    if exits.empty:
        return "<h2>売り（入替）</h2><p class=\"sub\">今週の手仕舞いはありません。</p>"
    rows = "".join(
        f'<tr><td class="l">{html.escape(str(r.Code))}</td>'
        f'<td class="num">{_fmt(getattr(r, "weight", None), pct=True)}</td>'
        f'<td class="l">{html.escape(str(getattr(r, "sector", "") or ""))}</td></tr>'
        for r in exits.itertuples()
    )
    return (
        "<h2>売り（入替）</h2>"
        '<p class="sub">前回保有から外れた銘柄。全株を寄りで手仕舞う。</p>'
        f'<div class="scroll"><table><thead><tr><th class="l">コード</th>'
        f"<th>前回ウェイト</th><th class=\"l\">業種</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


ENTRY_COLUMNS = [
    ("Code", "コード", "text"),
    ("weight", "ウェイト", "pct"),
    ("shares", "株数", "int"),
    ("raw_close", "株価", "int"),
    ("score", "合成", "num"),
    ("score_revision", "修正", "num"),
    ("score_quality", "質", "num"),
    ("score_momentum", "モメ", "num"),
    ("score_value", "割安", "num"),
    ("score_llm", "定性", "num"),
    ("revision_op", "予想修正", "pct"),
    ("op_yoy", "営利YoY", "pct"),
    ("progress_gap", "進捗乖離", "pct"),
    ("roe", "ROE", "pct"),
    ("per", "PER", "num"),
    ("vol_20", "ボラ", "pct"),
    ("days_to_next_earnings", "決算まで", "int"),
]


def _entry_section(detail: pd.DataFrame, capital: float | None) -> str:
    df = detail.copy()
    unbuyable = 0
    if capital and "raw_close" in df.columns:
        # 単元株 100 株に丸める。端株は買えない。
        units = (capital * df["weight"] / (df["raw_close"] * 100)).round()
        df["shares"] = (units.clip(lower=0) * 100).fillna(0).astype("Int64")
        # 目標ウェイトが 1 単元に届かない銘柄。小口資金では実際に起きる。
        unbuyable = int((df["shares"] == 0).sum())

    available = [(c, label, kind) for c, label, kind in ENTRY_COLUMNS if c in df.columns]
    header = "".join(
        f'<th class="l">{html.escape(label)}</th>' if kind == "text"
        else f"<th>{html.escape(label)}</th>"
        for _, label, kind in available
    )

    body = []
    for row in df.itertuples():
        cells = []
        for column, _, kind in available:
            value = getattr(row, column, None)
            if kind == "text":
                cells.append(f'<td class="l">{html.escape(str(value))}</td>')
            else:
                cells.append(f'<td class="num">{_fmt(value, kind)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    if not capital:
        note = '<p class="sub">株数は <code>--capital</code> を渡すと出ます。</p>'
    elif unbuyable:
        note = (
            f'<div class="banner">目標ウェイトが <strong>1 単元（100株）に届かない銘柄が '
            f"{unbuyable} 件</strong>あります。資金を増やすか、保有銘柄数"
            "（<code>n_positions</code>）を減らすか、単価の高い銘柄を外してください。"
            "そのまま等金額で買うとウェイトが崩れます。</div>"
        )
    else:
        note = ""
    return (
        "<h2>買い（保有目標）</h2>"
        '<p class="sub">翌営業日の寄りで執行する前提。スコアは断面内の z-score。</p>'
        f"{note}"
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _rationale_section(detail: pd.DataFrame) -> str:
    if "rationale" not in detail.columns and "llm_score" not in detail.columns:
        return (
            "<h2>定性分析</h2>"
            '<p class="sub">LLM 定性層は未実行です。'
            "<code>takochu analyze</code> を実行すると、決算開示から抽出した"
            "定性評価がここに出ます。</p>"
        )

    cards = []
    for row in detail.itertuples():
        rationale = getattr(row, "rationale", None)
        score = getattr(row, "llm_score", None)
        if not isinstance(rationale, str) and pd.isna(score):
            continue
        flags = getattr(row, "risk_flags", None)
        tags = ""
        if isinstance(flags, (list, tuple)):
            tags = "".join(
                f'<span class="tag">{html.escape(RISK_LABELS.get(f, str(f)))}</span>'
                for f in flags
            )
        why = html.escape(rationale) if isinstance(rationale, str) else ""
        cards.append(
            f'<div class="card"><span class="code">{html.escape(str(row.Code))}</span> '
            f'<span class="num">定性 {_fmt(score, "num")}'
            f' / 確信度 {_fmt(getattr(row, "llm_confidence", None), "num")}</span>'
            f"{(' ' + tags) if tags else ''}"
            f'<div class="why">{why}</div></div>'
        )

    if not cards:
        return (
            "<h2>定性分析</h2>"
            '<p class="sub">候補銘柄に対応する定性分析がありません。</p>'
        )
    return (
        "<h2>定性分析</h2>"
        '<p class="sub">決算開示の定性情報から抽出した評価。数値ではなく'
        "文章から読める部分を担当する層。</p>" + "".join(cards)
    )


def _footer(diagnostics: dict | None) -> str:
    lines = [
        "スコアの重みはバックテストで検証した仮説であって、将来の成績を保証しない。",
        "決算発表を保有期間中に跨ぐ銘柄は除外済み。",
    ]
    if diagnostics:
        items = " / ".join(f"{k}: {v}" for k, v in diagnostics.items())
        lines.append(html.escape(items))
    return "<footer>" + "<br>".join(lines) + "</footer>"


def _fmt(value, kind: str = "num", pct: bool = False) -> str:
    if pct:
        kind = "pct"
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))

    if kind == "pct":
        css = "pos" if number > 0 else ("neg" if number < 0 else "")
        text = f"{number * 100:,.1f}%"
        return f'<span class="{css}">{text}</span>' if css else text
    if kind == "int":
        return f"{number:,.0f}"
    css = "pos" if number > 0 else ("neg" if number < 0 else "")
    text = f"{number:,.2f}"
    return f'<span class="{css}">{text}</span>' if css else text


def write_report(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
