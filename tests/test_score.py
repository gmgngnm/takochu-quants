"""スコア合成とポートフォリオ構築のテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.strategy.score import build_scores, select_portfolio


def _snapshot(n: int = 20, sectors: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Date": pd.Timestamp("2023-06-30"),
            "Code": [f"{1000 + i}0" for i in range(n)],
            "score": np.linspace(2.0, -2.0, n),
            "sector33": [str(3050 + i % sectors) for i in range(n)],
            "vol_20": rng.uniform(0.15, 0.6, n),
            "tradable": True,
        }
    )


def test_ウェイトの合計は1になる():
    picks = select_portfolio(
        _snapshot(), {"n_positions": 10, "weighting": "equal", "max_weight": 1.0}
    )
    assert len(picks) == 10
    assert picks["weight"].sum() == pytest.approx(1.0)


def test_1銘柄あたりの上限が守られる():
    picks = select_portfolio(
        _snapshot(),
        {"n_positions": 10, "weighting": "inverse_vol", "max_weight": 0.15},
    )
    assert picks["weight"].max() <= 0.15 + 1e-9
    assert picks["weight"].sum() == pytest.approx(1.0)


def test_業種上限を超えて同じ業種を持たない():
    snapshot = _snapshot(n=40, sectors=2)
    picks = select_portfolio(
        snapshot,
        {"n_positions": 10, "max_sector_weight": 0.3, "weighting": "equal", "max_weight": 1.0},
    )
    assert picks["sector"].value_counts().max() <= 3


def test_売買不可の銘柄は選ばれない():
    snapshot = _snapshot()
    snapshot.loc[snapshot.index[:5], "tradable"] = False
    picks = select_portfolio(
        snapshot, {"n_positions": 10, "weighting": "equal", "max_weight": 1.0}
    )
    assert not set(picks["Code"]) & set(snapshot["Code"].iloc[:5])


def test_欠損した構成要素の重みは残りに配分される():
    """LLM スコア未算出でも、他の要素だけで正しくスコアが出ること."""
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2023-06-30")] * 4,
            "Code": ["A", "B", "C", "D"],
            "sector33": ["1", "1", "2", "2"],
            "revision_op": [0.3, 0.1, -0.1, -0.3],
            "mom_12_1": [0.2, 0.1, 0.0, -0.1],
            "roe": [0.15, 0.12, 0.08, 0.05],
            "per": [10.0, 12.0, 15.0, 20.0],
        }
    )
    scored = build_scores(df, {"revision": 0.5, "momentum": 0.5, "llm": 0.2})

    assert scored["score"].notna().all()
    assert scored["score_llm"].isna().all()
    # revision も momentum も A が最上位なので、合成スコアも A が首位
    assert scored.loc[scored["score"].idxmax(), "Code"] == "A"


def test_全要素が欠損ならスコアはNaN():
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2023-06-30")] * 3,
            "Code": ["A", "B", "C"],
            "revision_op": [np.nan] * 3,
        }
    )
    scored = build_scores(df, {"revision": 1.0})
    assert scored["score"].isna().all()
