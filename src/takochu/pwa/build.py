"""週次レポートを iPhone のホーム画面に置ける PWA として書き出す.

用途は「読む」ではなく「**確認して発注する**」。月曜の朝、電車の中で
1銘柄ずつ確認し、最後に発注メモをコピーして証券会社のアプリに移る、
という導線に合わせてある。

データは index.html に埋め込む。別ファイルを fetch しないので、
保有銘柄という機微なデータがネットワークを余計に流れないし、
オフラインでも確実に開く。
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

from takochu.pwa.icon import render_icon
from takochu.report import RISK_LABELS

log = logging.getLogger(__name__)

APP_NAME = "takochu"
PAYLOAD_TOKEN = "__TAKOCHU_PAYLOAD__"
BUILD_TOKEN = "__TAKOCHU_BUILD__"


def build_pwa(
    out_dir: Path,
    picks: pd.DataFrame,
    snapshot: pd.DataFrame,
    decision_date: pd.Timestamp,
    exec_date: pd.Timestamp | None,
    previous_picks: pd.DataFrame | None = None,
    capital: float | None = None,
    alerts: dict | None = None,
    demo: bool = False,
) -> Path:
    """PWA 一式を out_dir に書き出し、index.html のパスを返す."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _payload(picks, snapshot, decision_date, exec_date, previous_picks, capital)
    payload["alerts"] = _alerts(alerts)
    # 架空データであることをページ自身に持たせる。公開先で取り違えられないように。
    payload["isDemo"] = bool(demo)
    build_id = f"{decision_date:%Y%m%d}-{len(payload['buys'])}-{len(payload['sells'])}"

    index = (
        INDEX_HTML.replace(PAYLOAD_TOKEN, json.dumps(payload, ensure_ascii=False))
        .replace(BUILD_TOKEN, build_id)
    )
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    (out_dir / "sw.js").write_text(SERVICE_WORKER.replace(BUILD_TOKEN, build_id), encoding="utf-8")
    (out_dir / "manifest.webmanifest").write_text(MANIFEST, encoding="utf-8")
    # GitHub Pages の Jekyll 処理を止める（_ 始まりのファイルが消えるのを防ぐ）
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    for size in (180, 512):
        (out_dir / f"icon-{size}.png").write_bytes(render_icon(size))

    log.info("PWA を書き出しました: %s", out_dir)
    return out_dir / "index.html"


# ------------------------------------------------------------------ データ整形


def _payload(
    picks: pd.DataFrame,
    snapshot: pd.DataFrame,
    decision_date: pd.Timestamp,
    exec_date: pd.Timestamp | None,
    previous_picks: pd.DataFrame | None,
    capital: float | None,
) -> dict:
    detail = picks.merge(snapshot, on="Code", how="left", suffixes=("", "_snap"))

    buys = []
    for row in detail.itertuples():
        price = _f(getattr(row, "raw_close", None))
        weight = _f(getattr(row, "weight", None)) or 0.0
        shares = None
        amount = None
        if capital and price:
            units = round(capital * weight / (price * 100))
            shares = int(max(units, 0) * 100)
            amount = shares * price

        flags = getattr(row, "risk_flags", None)
        buys.append(
            {
                "code": str(row.Code),
                "weight": weight,
                "price": price,
                "shares": shares,
                "amount": amount,
                "sector": _s(getattr(row, "sector", None)),
                "score": _f(getattr(row, "score", None)),
                "parts": {
                    "修正": _f(getattr(row, "score_revision", None)),
                    "質": _f(getattr(row, "score_quality", None)),
                    "モメ": _f(getattr(row, "score_momentum", None)),
                    "割安": _f(getattr(row, "score_value", None)),
                    "定性": _f(getattr(row, "score_llm", None)),
                },
                "facts": {
                    "予想修正": _pct(getattr(row, "revision_op", None)),
                    "営利YoY": _pct(getattr(row, "op_yoy", None)),
                    "ROE": _pct(getattr(row, "roe", None)),
                    "PER": _num(getattr(row, "per", None)),
                    "ボラ": _pct(getattr(row, "vol_20", None)),
                },
                "risks": [
                    RISK_LABELS.get(f, str(f))
                    for f in (flags if isinstance(flags, (list, tuple)) else [])
                ],
                "why": _s(getattr(row, "rationale", None)),
            }
        )

    sells = []
    if previous_picks is not None and not previous_picks.empty:
        keep = set(picks["Code"].astype(str))
        for row in previous_picks.itertuples():
            code = str(row.Code)
            if code in keep:
                continue
            sells.append({"code": code, "weight": _f(getattr(row, "weight", None))})

    unbuyable = sum(1 for b in buys if b["shares"] == 0)
    return {
        "decisionDate": f"{decision_date:%Y-%m-%d}",
        "execDate": f"{exec_date:%Y-%m-%d}" if exec_date is not None else None,
        "capital": capital,
        "universe": int(snapshot["tradable"].sum()) if "tradable" in snapshot else len(snapshot),
        "unbuyable": unbuyable,
        "buys": buys,
        "sells": sells,
    }


def _alerts(alerts: dict | None) -> list[dict]:
    """monitor が出した手仕舞い推奨。週次の判断とは別に、最優先で見せる."""
    if not alerts:
        return []
    records = alerts.get("alerts") or []
    return [
        {
            "code": str(a.get("code")),
            "reason": "損切り" if a.get("reason") == "stop_loss" else "利確",
            "kind": str(a.get("reason")),
            "change": a.get("change"),
            "touchedOn": a.get("touched_on"),
            "entry": a.get("entry_price"),
            "last": a.get("last_close"),
        }
        for a in records
    ]


def _f(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _s(value) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return html.escape(str(value))


def _pct(value) -> str | None:
    number = _f(value)
    return None if number is None else f"{number * 100:,.1f}%"


def _num(value) -> str | None:
    number = _f(value)
    return None if number is None else f"{number:,.1f}"


# ----------------------------------------------------------------- テンプレート

MANIFEST = json.dumps(
    {
        "name": "takochu 週次スクリーニング",
        "short_name": APP_NAME,
        "description": "国内プライム株の週次スイング候補を確認して発注するためのアプリ",
        "start_url": "./index.html",
        "scope": "./",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#101318",
        "theme_color": "#101318",
        "lang": "ja",
        "icons": [
            {"src": "./icon-180.png", "sizes": "180x180", "type": "image/png"},
            {"src": "./icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        ],
    },
    ensure_ascii=False,
    indent=2,
)

SERVICE_WORKER = """\
// ビルドごとにキャッシュ名を変える。新しい週のレポートを出したら
// 古いキャッシュは捨てられ、次回起動時に新しい内容が出る。
const CACHE = 'takochu-__TAKOCHU_BUILD__';
const ASSETS = ['./', './index.html', './manifest.webmanifest', './icon-180.png', './icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// ネットワーク優先・キャッシュフォールバック。新しいレポートを開いたときに
// 古い週の内容が出続けないようにするため。
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match('./index.html')))
  );
});
"""

INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>takochu 週次スクリーニング</title>
<link rel="manifest" href="./manifest.webmanifest">
<link rel="apple-touch-icon" href="./icon-180.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="takochu">
<meta name="theme-color" content="#101318">
<style>
:root {
  --bg: #f6f7f9; --surface: #ffffff; --sunken: #eef1f5;
  --ink: #151a21; --muted: #5c6674; --rule: #dce1e8;
  --indigo: #2a4a7f; --crimson: #b23440; --moss: #35705a;
  --amber-bg: #fbf3e2; --amber-fg: #7d5a17; --amber-ln: #e6d3a8;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  --bar: 76px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101318; --surface: #171b21; --sunken: #1d222a;
    --ink: #e7ecf2; --muted: #96a1b1; --rule: #272e37;
    --indigo: #85a9de; --crimson: #e28188; --moss: #6dbb9c;
    --amber-bg: #251f12; --amber-fg: #d9be85; --amber-ln: #443925;
  }
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html { color-scheme: light dark; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Hiragino Kaku Gothic ProN", system-ui, sans-serif;
  font-size: 16px; line-height: 1.6;
  padding-bottom: calc(var(--bar) + env(safe-area-inset-bottom));
  overscroll-behavior-y: contain;
}
.wrap { padding-inline: 16px; max-width: 560px; margin: 0 auto; }

header {
  padding-block: calc(env(safe-area-inset-top) + 18px) 16px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 18px;
}
.kicker {
  font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--indigo);
}
h1 { font-size: 1.35rem; margin: 6px 0 10px; letter-spacing: -0.01em; }
.dates { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.82rem; color: var(--muted); }
.dates b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }

.notice {
  background: var(--amber-bg); color: var(--amber-fg); border: 1px solid var(--amber-ln);
  border-radius: 10px; padding: 12px 14px; font-size: 0.85rem; margin-bottom: 20px;
}
.notice b { font-weight: 600; }

h2 {
  font-size: 0.95rem; margin: 26px 0 10px; display: flex;
  align-items: center; justify-content: space-between; gap: 10px;
}
h2 .count { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); font-weight: 400; }
.hint { font-size: 0.8rem; color: var(--muted); margin: -4px 0 12px; }

.list { display: flex; flex-direction: column; gap: 10px; }

.item {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 12px; overflow: hidden; transition: opacity .18s ease;
}
.item.checked { opacity: .48; }
.item .top {
  display: grid; grid-template-columns: 1fr auto; gap: 12px;
  align-items: center; padding: 13px 14px; cursor: pointer;
}
.item .code {
  font-family: var(--mono); font-size: 1.1rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.item .sub { font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.item .qty { text-align: right; font-variant-numeric: tabular-nums; }
.item .qty .shares { font-family: var(--mono); font-size: 1rem; font-weight: 600; }
.item .qty .shares.skip { color: var(--crimson); font-family: inherit; font-size: 0.88rem; }
.item.alert { border-color: color-mix(in srgb, var(--crimson) 45%, var(--rule)); }
.item .qty .shares.reason-label {
  font-family: inherit; font-size: 0.9rem; color: var(--ink);
}
.item .qty .amount.pos { color: var(--crimson); }
.item .qty .amount.neg { color: var(--moss); }
.item .qty .amount { font-size: 0.76rem; color: var(--muted); }

.tick {
  width: 30px; height: 30px; border-radius: 50%; border: 2px solid var(--rule);
  display: grid; place-items: center; flex: none; color: transparent;
  font-size: 15px; line-height: 1; transition: background .15s, border-color .15s, color .15s;
}
.item.checked .tick { background: var(--moss); border-color: var(--moss); color: #fff; }
.row { display: flex; align-items: center; gap: 12px; }

.meta { padding: 0 14px 12px; display: flex; flex-wrap: wrap; gap: 5px; }
.pill {
  font-family: var(--mono); font-size: 0.7rem; padding: 3px 7px;
  border-radius: 5px; background: var(--sunken); color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.pill.risk { background: var(--amber-bg); color: var(--amber-fg); font-family: inherit; }
.why {
  padding: 0 14px 13px; font-size: 0.82rem; color: var(--muted);
  line-height: 1.65;
}
details.parts { border-top: 1px solid var(--rule); }
details.parts summary {
  padding: 9px 14px; font-size: 0.78rem; color: var(--muted);
  cursor: pointer; list-style: none;
}
details.parts summary::-webkit-details-marker { display: none; }
details.parts summary::after { content: " ▾"; }
details.parts[open] summary::after { content: " ▴"; }
.parts-grid {
  padding: 0 14px 13px; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(62px, 1fr)); gap: 8px;
}
.parts-grid div { font-size: 0.72rem; color: var(--muted); }
.parts-grid b {
  display: block; font-family: var(--mono); font-size: 0.85rem;
  color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums;
}
.pos { color: var(--crimson); } .neg { color: var(--moss); }

.empty { color: var(--muted); font-size: 0.85rem; padding: 4px 0 2px; }

.bar {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 10;
  background: color-mix(in srgb, var(--surface) 92%, transparent);
  backdrop-filter: saturate(140%) blur(12px);
  border-top: 1px solid var(--rule);
  padding: 10px 16px calc(10px + env(safe-area-inset-bottom));
}
.bar .inner {
  max-width: 560px; margin: 0 auto; display: grid;
  grid-template-columns: 1fr auto; gap: 12px; align-items: center;
}
.progress { font-size: 0.8rem; color: var(--muted); }
.progress b { color: var(--ink); font-variant-numeric: tabular-nums; }
.track {
  height: 4px; background: var(--sunken); border-radius: 2px;
  margin-top: 6px; overflow: hidden;
}
.track i {
  display: block; height: 100%; background: var(--moss);
  width: 0; transition: width .25s ease;
}
button {
  font: inherit; font-size: 0.85rem; font-weight: 600;
  padding: 11px 16px; border-radius: 10px; border: 1px solid var(--indigo);
  background: var(--indigo); color: #fff; cursor: pointer; white-space: nowrap;
}
button:active { transform: translateY(1px); }
button.ghost { background: transparent; color: var(--indigo); }
button:focus-visible { outline: 2px solid var(--indigo); outline-offset: 2px; }

.tools { display: flex; gap: 10px; margin: 22px 0 8px; }
.tools button { flex: 1; }

footer {
  margin-top: 26px; padding-top: 16px; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 0.75rem; line-height: 1.7;
}
dialog {
  border: 1px solid var(--rule); border-radius: 14px; padding: 0;
  background: var(--surface); color: var(--ink);
  max-width: 92vw; width: 460px;
}
dialog::backdrop { background: rgba(0,0,0,.45); }
dialog .dlg { padding: 18px; }
dialog h3 { margin: 0 0 10px; font-size: 1rem; }
dialog pre {
  font-family: var(--mono); font-size: 0.78rem; line-height: 1.7;
  background: var(--sunken); border-radius: 8px; padding: 12px;
  max-height: 46vh; overflow: auto; white-space: pre-wrap; margin: 0 0 14px;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="kicker">週次スクリーニング</div>
    <h1 id="title">takochu</h1>
    <div class="dates" id="dates"></div>
  </header>

  <div class="notice">
    <b>これは売買提案であって発注ではありません。</b>
    システムが知らない材料（指数除外・不祥事・TOB・直近の報道）は自分で確認すること。
  </div>

  <div class="notice" id="demo-banner" hidden style="border-style:dashed">
    <b>これは UI デモです。</b>
    表示されている銘柄コード・株価・スコアはすべて<b>乱数から作った架空のデータ</b>で、
    実在の銘柄とも相場とも関係ありません。
  </div>

  <section id="alert-section" hidden>
    <h2>0. 手仕舞い（優先）<span class="count" id="alert-count"></span></h2>
    <p class="hint">利確・損切りの水準に達した保有。週次の入替より先に処理する。</p>
    <div class="list" id="alerts"></div>
    <p class="hint">日足ベースの判定。ザラ場で即座に反応するには証券会社の API が要る。</p>
  </section>

  <div id="unbuyable"></div>

  <section>
    <h2>1. 売り（入替）<span class="count" id="sell-count"></span></h2>
    <p class="hint">先に手仕舞ってから買う。</p>
    <div class="list" id="sells"></div>
  </section>

  <section>
    <h2>2. 買い（保有目標）<span class="count" id="buy-count"></span></h2>
    <p class="hint">タップで確認済みに。スコアは断面内の z-score。</p>
    <div class="list" id="buys"></div>
  </section>

  <div class="tools">
    <button class="ghost" id="reset">確認をリセット</button>
  </div>

  <footer id="footer"></footer>
</div>

<div class="bar">
  <div class="inner">
    <div class="progress">
      <span id="progress-text">—</span>
      <div class="track"><i id="progress-bar"></i></div>
    </div>
    <button id="memo">発注メモ</button>
  </div>
</div>

<dialog id="dlg">
  <div class="dlg">
    <h3>発注メモ</h3>
    <pre id="memo-text"></pre>
    <div class="tools" style="margin:0">
      <button class="ghost" id="close">閉じる</button>
      <button id="copy">コピー</button>
    </div>
  </div>
</dialog>

<script>
const DATA = __TAKOCHU_PAYLOAD__;
const KEY = 'takochu:checked:' + DATA.decisionDate;

let checked = new Set();
try {
  checked = new Set(JSON.parse(localStorage.getItem(KEY) || '[]'));
} catch (e) {
  checked = new Set();
}

function persist() {
  // 非公開ブラウズなどで書き込めないことがある。落とさず黙って諦める。
  try { localStorage.setItem(KEY, JSON.stringify([...checked])); } catch (e) { /* noop */ }
}

const yen = (n) => '¥' + Math.round(n).toLocaleString('ja-JP');
const signed = (v) => (v === null || v === undefined) ? '—' : v.toFixed(2);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function card(entry, kind) {
  const id = kind + ':' + entry.code;
  const item = el('div', 'item' + (checked.has(id) ? ' checked' : ''));

  const top = el('div', 'top');
  const left = el('div', 'row');
  const tick = el('div', 'tick', '✓');
  const names = el('div');
  names.appendChild(el('div', 'code', entry.code));
  if (kind === 'buy') {
    const bits = [];
    if (entry.weight !== null) bits.push((entry.weight * 100).toFixed(1) + '%');
    if (entry.price !== null) bits.push(yen(entry.price));
    if (entry.sector) bits.push('業種 ' + entry.sector);
    names.appendChild(el('div', 'sub', bits.join(' · ')));
  } else {
    names.appendChild(el('div', 'sub', '全株を手仕舞う'));
  }
  left.append(tick, names);

  const qty = el('div', 'qty');
  if (kind === 'buy' && entry.shares === 0) {
    // 目標ウェイトが1単元に満たない。黙って0株と出すと買い忘れに見えるので明示する。
    qty.appendChild(el('div', 'shares skip', '見送り'));
    qty.appendChild(el('div', 'amount', '1単元に届かず'));
  } else if (kind === 'buy' && entry.shares !== null) {
    qty.appendChild(el('div', 'shares', entry.shares.toLocaleString('ja-JP') + '株'));
    if (entry.amount) qty.appendChild(el('div', 'amount', yen(entry.amount)));
  } else if (kind === 'buy') {
    qty.appendChild(el('div', 'amount', '株数は資金未設定'));
  }
  top.append(left, qty);
  item.appendChild(top);

  if (kind === 'buy') {
    const meta = el('div', 'meta');
    for (const [label, value] of Object.entries(entry.facts)) {
      if (value !== null) meta.appendChild(el('span', 'pill', label + ' ' + value));
    }
    for (const risk of entry.risks) meta.appendChild(el('span', 'pill risk', risk));
    if (meta.childNodes.length) item.appendChild(meta);

    if (entry.why) item.appendChild(el('div', 'why', entry.why));

    const det = el('details', 'parts');
    det.appendChild(el('summary', null, 'スコア内訳 ' + signed(entry.score)));
    const grid = el('div', 'parts-grid');
    for (const [label, value] of Object.entries(entry.parts)) {
      const cell = el('div');
      const strong = el('b', (value > 0 ? 'pos' : value < 0 ? 'neg' : ''), signed(value));
      cell.append(strong, document.createTextNode(label));
      grid.appendChild(cell);
    }
    det.appendChild(grid);
    item.appendChild(det);
  }

  top.addEventListener('click', () => {
    if (checked.has(id)) { checked.delete(id); } else { checked.add(id); }
    item.classList.toggle('checked');
    persist();
    render();
  });

  return item;
}

function alertCard(a) {
  const id = 'alert:' + a.code;
  const item = el('div', 'item alert' + (checked.has(id) ? ' checked' : ''));
  const top = el('div', 'top');
  const left = el('div', 'row');
  left.append(el('div', 'tick', '✓'));
  const names = el('div');
  names.appendChild(el('div', 'code', a.code));
  names.appendChild(el('div', 'sub', a.touchedOn + ' に到達 · 全株を手仕舞う'));
  left.appendChild(names);

  const qty = el('div', 'qty');
  const pct = a.change === null || a.change === undefined
    ? '—' : (a.change >= 0 ? '+' : '') + (a.change * 100).toFixed(1) + '%';
  // ラベルは中立色。色は騰落率だけに使う（日本式で赤=上げ・緑=下げ）。
  // 「損切り」を下落色で出すと良い知らせに見えてしまう。
  qty.appendChild(el('div', 'shares reason-label', a.reason));
  const move = el('div', 'amount ' + (a.change >= 0 ? 'pos' : 'neg'), pct);
  qty.appendChild(move);
  top.append(left, qty);
  item.appendChild(top);

  top.addEventListener('click', () => {
    if (checked.has(id)) { checked.delete(id); } else { checked.add(id); }
    item.classList.toggle('checked');
    persist();
    render();
  });
  return item;
}

function render() {
  const total = DATA.buys.length + DATA.sells.length + (DATA.alerts || []).length;
  const done = [...checked].length;
  document.getElementById('progress-text').innerHTML =
    '<b>' + done + '</b> / ' + total + ' 確認済み';
  document.getElementById('progress-bar').style.width =
    (total ? (done / total) * 100 : 0) + '%';
}

function boot() {
  document.getElementById('title').textContent =
    DATA.decisionDate + ' 判断';
  const dates = document.getElementById('dates');
  dates.innerHTML = '';
  const add = (label, value) => {
    const wrapEl = el('div');
    wrapEl.append(document.createTextNode(label + ' '), el('b', null, value));
    dates.appendChild(wrapEl);
  };
  add('執行', DATA.execDate || '翌営業日の寄り');
  add('候補', DATA.buys.length + '銘柄');
  add('ユニバース', String(DATA.universe));
  if (DATA.capital) add('資金', yen(DATA.capital));

  if (DATA.unbuyable > 0) {
    const box = el('div', 'notice');
    box.innerHTML = '<b>目標ウェイトが1単元（100株）に届かない銘柄が ' + DATA.unbuyable +
      ' 件</b>あります。そのまま買うとウェイトが崩れます。資金を増やすか銘柄数を減らしてください。';
    document.getElementById('unbuyable').appendChild(box);
  }

  if (DATA.isDemo) document.getElementById('demo-banner').hidden = false;

  if (DATA.alerts && DATA.alerts.length) {
    const section = document.getElementById('alert-section');
    section.hidden = false;
    document.getElementById('alert-count').textContent = DATA.alerts.length + '件';
    const list = document.getElementById('alerts');
    DATA.alerts.forEach((a) => list.appendChild(alertCard(a)));
  }

  const sells = document.getElementById('sells');
  document.getElementById('sell-count').textContent = DATA.sells.length + '件';
  if (!DATA.sells.length) {
    sells.appendChild(el('div', 'empty', '今週の手仕舞いはありません。'));
  } else {
    DATA.sells.forEach((s) => sells.appendChild(card(s, 'sell')));
  }

  const buys = document.getElementById('buys');
  document.getElementById('buy-count').textContent = DATA.buys.length + '件';
  if (!DATA.buys.length) {
    buys.appendChild(el('div', 'empty', '候補がありません。'));
  } else {
    DATA.buys.forEach((b) => buys.appendChild(card(b, 'buy')));
  }

  document.getElementById('footer').textContent =
    'スコアの重みはバックテストで検証した仮説であって、将来の成績を保証しない。' +
    '決算発表を保有期間中に跨ぐ銘柄は除外済み。損益はすべて利用者の責任。';

  render();
}

function memoText() {
  const lines = [DATA.decisionDate + ' 判断 / ' + (DATA.execDate || '翌営業日') + ' 寄りで執行'];
  if (DATA.alerts && DATA.alerts.length) {
    lines.push('', '【手仕舞い】利確・損切り到達');
    DATA.alerts.forEach((a) => lines.push(a.code + '  全株  (' + a.reason + ')'));
  }
  if (DATA.sells.length) {
    lines.push('', '【売り】');
    DATA.sells.forEach((s) => lines.push(s.code + '  全株'));
  }
  const orderable = DATA.buys.filter((b) => b.shares !== 0);
  const skipped = DATA.buys.filter((b) => b.shares === 0);
  if (orderable.length) {
    lines.push('', '【買い】');
    orderable.forEach((b) => {
      const qty = b.shares === null ? '(株数未算出)' : b.shares.toLocaleString('ja-JP') + '株';
      const amt = b.amount ? '  ' + yen(b.amount) : '';
      lines.push(b.code + '  ' + qty + amt);
    });
  }
  if (skipped.length) {
    lines.push('', '【見送り】1単元に届かないため発注しない');
    skipped.forEach((b) => lines.push(b.code));
  }
  return lines.join('\\n');
}

const dlg = document.getElementById('dlg');
document.getElementById('memo').addEventListener('click', () => {
  document.getElementById('memo-text').textContent = memoText();
  dlg.showModal();
});
document.getElementById('close').addEventListener('click', () => dlg.close());
document.getElementById('copy').addEventListener('click', async () => {
  const button = document.getElementById('copy');
  try {
    await navigator.clipboard.writeText(memoText());
    button.textContent = 'コピーしました';
  } catch (e) {
    // クリップボードが使えない場合は選択できる状態にして知らせる
    const pre = document.getElementById('memo-text');
    const range = document.createRange();
    range.selectNodeContents(pre);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    button.textContent = '長押しでコピー';
  }
  setTimeout(() => { button.textContent = 'コピー'; }, 2200);
});
document.getElementById('reset').addEventListener('click', () => {
  checked.clear();
  persist();
  document.querySelectorAll('.item.checked').forEach((n) => n.classList.remove('checked'));
  render();
});

boot();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {
      // http:// で開いた場合など。オフライン化されないだけで、閲覧はできる。
    });
  });
}
</script>
</body>
</html>
"""
