"""週次レポート生成のテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd

from takochu.report import build_report


def _snapshot(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-06-28"),
            "Code": [f"{7200 + i}0" for i in range(n)],
            "raw_close": np.linspace(1000, 3000, n),
            "score": np.linspace(1.5, -0.5, n),
            "score_revision": np.linspace(1.2, -0.2, n),
            "score_quality": np.linspace(0.8, 0.1, n),
            "score_momentum": np.linspace(0.5, -0.5, n),
            "score_value": np.linspace(0.3, 0.0, n),
            "score_llm": np.linspace(1.0, -1.0, n),
            "revision_op": np.linspace(0.15, -0.05, n),
            "op_yoy": np.linspace(0.30, 0.02, n),
            "roe": np.linspace(0.14, 0.05, n),
            "per": np.linspace(11, 25, n),
            "vol_20": np.linspace(0.22, 0.40, n),
            "days_to_next_earnings": [45] * n,
            "sector33": ["3050"] * n,
            "tradable": True,
        }
    )


def _picks(snapshot: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    top = snapshot.head(n)
    return pd.DataFrame(
        {
            "Code": top["Code"].values,
            "score": top["score"].values,
            "weight": [1 / n] * n,
            "sector": ["3050"] * n,
        }
    )


def test_レポートが生成され候補銘柄を含む():
    snapshot = _snapshot()
    picks = _picks(snapshot)
    html_out = build_report(
        picks=picks,
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=pd.Timestamp("2024-07-01"),
    )

    assert html_out.startswith("<!doctype html>")
    assert "2024-06-28" in html_out and "2024-07-01" in html_out
    for code in picks["Code"]:
        assert code in html_out


def test_発注ではないことを明示する():
    snapshot = _snapshot()
    html_out = build_report(
        picks=_picks(snapshot),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
    )
    assert "発注ではありません" in html_out


def test_前回保有から外れた銘柄が売り候補に出る():
    snapshot = _snapshot()
    picks = _picks(snapshot, n=2)
    previous = pd.DataFrame(
        {"Code": ["72000", "79999"], "weight": [0.5, 0.5], "sector": ["3050", "3051"]}
    )
    html_out = build_report(
        picks=picks,
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=pd.Timestamp("2024-07-01"),
        previous_picks=previous,
    )
    assert "79999" in html_out       # 外れた銘柄は売り
    assert "手仕舞いはありません" not in html_out


def test_資金を渡すと単元株に丸めた株数が出る():
    snapshot = _snapshot()
    picks = _picks(snapshot, n=2)
    html_out = build_report(
        picks=picks,
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
        capital=10_000_000,
    )
    assert "株数" in html_out


def test_LLM層未実行でも壊れない():
    snapshot = _snapshot().drop(columns=["score_llm"])
    html_out = build_report(
        picks=_picks(snapshot),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
    )
    assert "takochu analyze" in html_out


def test_分析文のHTMLをエスケープする():
    """rationale は LLM 出力。そのまま埋め込むとレポートが壊れる."""
    snapshot = _snapshot()
    snapshot["rationale"] = "<script>alert(1)</script> 増収増益"
    snapshot["llm_score"] = 1.0
    snapshot["llm_confidence"] = 0.8
    snapshot["risk_flags"] = [["cost_pressure"]] * len(snapshot)

    html_out = build_report(
        picks=_picks(snapshot),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
    )
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "コスト圧迫" in html_out


def test_単元株に届かない銘柄を警告する():
    """小口資金では目標ウェイトが100株に満たない銘柄が出る。黙って0株にしない."""
    snapshot = _snapshot(n=3)
    snapshot["raw_close"] = [500_000.0, 300_000.0, 1_000.0]
    html_out = build_report(
        picks=_picks(snapshot, n=3),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
        capital=1_000_000,
    )
    assert "1 単元（100株）に届かない銘柄" in html_out


def test_全銘柄が買えるなら警告は出ない():
    snapshot = _snapshot(n=3)
    snapshot["raw_close"] = [1000.0, 1200.0, 900.0]
    html_out = build_report(
        picks=_picks(snapshot, n=3),
        snapshot=snapshot,
        decision_date=pd.Timestamp("2024-06-28"),
        exec_date=None,
        capital=10_000_000,
    )
    assert "届かない銘柄" not in html_out
