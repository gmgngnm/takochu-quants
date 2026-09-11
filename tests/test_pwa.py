"""PWA ビルドのテスト."""

from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd

from takochu.pwa.build import build_pwa
from takochu.pwa.icon import render_icon


def _snapshot(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-06-28"),
            "Code": [f"{7200 + i}0" for i in range(n)],
            "raw_close": [1000.0, 2000.0, 400_000.0, 800.0][:n],
            "score": np.linspace(1.5, 0.2, n),
            "score_revision": np.linspace(1.2, 0.1, n),
            "score_quality": np.linspace(0.8, 0.1, n),
            "score_momentum": np.linspace(0.5, -0.2, n),
            "score_value": np.linspace(0.3, 0.0, n),
            "score_llm": np.linspace(1.0, -0.5, n),
            "revision_op": np.linspace(0.15, 0.01, n),
            "op_yoy": np.linspace(0.30, 0.05, n),
            "roe": np.linspace(0.14, 0.06, n),
            "per": np.linspace(11, 20, n),
            "vol_20": np.linspace(0.22, 0.35, n),
            "sector33": ["3050"] * n,
            "tradable": True,
        }
    )


def _picks(snapshot: pd.DataFrame) -> pd.DataFrame:
    n = len(snapshot)
    return pd.DataFrame(
        {
            "Code": snapshot["Code"].values,
            "score": snapshot["score"].values,
            "weight": [1 / n] * n,
            "sector": ["3050"] * n,
        }
    )


def _payload(html: str) -> dict:
    marker = "const DATA = "
    start = html.index(marker) + len(marker)
    end = html.index("\nconst KEY", start)
    return json.loads(html[start:end].rstrip().rstrip(";"))


def _build(tmp_path, capital=10_000_000, previous=None):
    snapshot = _snapshot()
    index = build_pwa(
        out_dir=tmp_path / "pwa",
        picks=_picks(snapshot),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=pd.Timestamp("2024-07-01"),
        previous_picks=previous,
        capital=capital,
    )
    return index, index.read_text(encoding="utf-8")


def test_PWAに必要なファイルが揃う(tmp_path):
    index, _ = _build(tmp_path)
    out = index.parent
    for name in ["index.html", "sw.js", "manifest.webmanifest", "icon-180.png", "icon-512.png"]:
        assert (out / name).exists(), name
    # GitHub Pages の Jekyll に消されないようにする
    assert (out / ".nojekyll").exists()


def test_マニフェストがホーム画面追加に必要な項目を持つ(tmp_path):
    index, _ = _build(tmp_path)
    manifest = json.loads((index.parent / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["display"] == "standalone"
    assert manifest["start_url"].startswith("./")
    sizes = {icon["sizes"] for icon in manifest["icons"]}
    assert "180x180" in sizes and "512x512" in sizes


def test_アイコンが正しいPNGで指定サイズになる():
    for size in (180, 512):
        data = render_icon(size)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        # IHDR は先頭 8 バイトの後、長さ4 + タグ4 の直後から幅・高さ
        width, height = struct.unpack(">2I", data[16:24])
        assert (width, height) == (size, size)


def test_データがHTMLに埋め込まれ外部取得がない(tmp_path):
    index, html = _build(tmp_path)
    payload = _payload(html)
    assert payload["decisionDate"] == "2024-06-28"
    assert len(payload["buys"]) == 4
    # 保有銘柄を別ファイルに置かない（余計にネットワークへ出さないため）
    assert not (index.parent / "data.json").exists()


def test_1単元に届かない銘柄を数える(tmp_path):
    """40万円の銘柄は1単元4000万円。1000万円の25%では買えない."""
    _, html = _build(tmp_path, capital=10_000_000)
    payload = _payload(html)
    assert payload["unbuyable"] >= 1
    expensive = next(b for b in payload["buys"] if b["price"] == 400_000.0)
    assert expensive["shares"] == 0


def test_前回保有から外れた銘柄が売りに入る(tmp_path):
    previous = pd.DataFrame({"Code": ["72000", "99999"], "weight": [0.5, 0.5]})
    _, html = _build(tmp_path, previous=previous)
    payload = _payload(html)
    assert [s["code"] for s in payload["sells"]] == ["99999"]


def test_定性分析の文はエスケープされる(tmp_path):
    snapshot = _snapshot()
    snapshot["rationale"] = "<img src=x onerror=alert(1)> 増収増益"
    snapshot["risk_flags"] = [["cost_pressure"]] * len(snapshot)
    index = build_pwa(
        out_dir=tmp_path / "pwa",
        picks=_picks(snapshot),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
        capital=10_000_000,
    )
    payload = _payload(index.read_text(encoding="utf-8"))
    why = payload["buys"][0]["why"]
    assert "<img" not in why
    assert "&lt;img" in why
    assert payload["buys"][0]["risks"] == ["コスト圧迫"]


def test_サービスワーカーのキャッシュ名が週ごとに変わる(tmp_path):
    """新しい週のレポートを出したら古い内容が残らないこと."""
    index, _ = _build(tmp_path)
    sw = (index.parent / "sw.js").read_text(encoding="utf-8")
    assert "takochu-20240628-" in sw
