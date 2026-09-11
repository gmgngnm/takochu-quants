"""週の途中での手仕舞い（利確・損切り）と週末ノーポジのテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.backtest.engine import run_backtest
from takochu.backtest.exits import HELD, STOP_LOSS, TAKE_PROFIT, simulate_exits


def _matrices(prices: dict[str, list[tuple[float, float, float, float]]], dates):
    """{code: [(open, high, low, close), ...]} から 4 つの行列を作る."""
    frames = {}
    for i, key in enumerate(["open", "high", "low", "close"]):
        frames[key] = pd.DataFrame(
            {code: [bar[i] for bar in bars] for code, bars in prices.items()}, index=dates
        )
    return frames


def test_水準に触れなければ最後まで持ち切る():
    dates = pd.bdate_range("2024-06-03", periods=5)
    m = _matrices({"A": [(100, 101, 99, 100)] * 5}, dates)
    result = simulate_exits(
        codes=pd.Index(["A"]),
        entry_price=pd.Series({"A": 100.0}),
        final_price=pd.Series({"A": 100.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        take_profit=0.10, stop_loss=0.10,
    )
    assert result.reasons["A"] == HELD
    assert result.returns["A"] == pytest.approx(0.0)


def test_利確ラインに触れたらそこで降りる():
    dates = pd.bdate_range("2024-06-03", periods=4)
    bars = [(100, 100, 100, 100), (100, 104, 99, 103), (103, 120, 102, 119), (119, 120, 118, 120)]
    m = _matrices({"A": bars}, dates)
    result = simulate_exits(
        codes=pd.Index(["A"]),
        entry_price=pd.Series({"A": 100.0}),
        final_price=pd.Series({"A": 120.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        take_profit=0.08,
    )
    # 3日目に高値 120 で +8% に触れる -> 最後まで持った +20% にはならない
    assert result.reasons["A"] == TAKE_PROFIT
    assert result.returns["A"] == pytest.approx(0.08)
    assert result.exit_dates["A"] == dates[2]


def test_損切りラインに触れたらそこで降りる():
    dates = pd.bdate_range("2024-06-03", periods=4)
    bars = [(100, 100, 100, 100), (100, 101, 94, 95), (95, 96, 80, 82), (82, 83, 81, 82)]
    m = _matrices({"A": bars}, dates)
    result = simulate_exits(
        codes=pd.Index(["A"]),
        entry_price=pd.Series({"A": 100.0}),
        final_price=pd.Series({"A": 82.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        stop_loss=0.05,
    )
    assert result.reasons["A"] == STOP_LOSS
    assert result.returns["A"] == pytest.approx(-0.05)


def test_同じ日に両方へ触れたら損切りを採る():
    """日足からはどちらが先か分からない。保守側を採ること."""
    dates = pd.bdate_range("2024-06-03", periods=2)
    m = _matrices({"A": [(100, 100, 100, 100), (100, 115, 90, 100)]}, dates)
    result = simulate_exits(
        codes=pd.Index(["A"]),
        entry_price=pd.Series({"A": 100.0}),
        final_price=pd.Series({"A": 100.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        take_profit=0.10, stop_loss=0.05,
    )
    assert result.reasons["A"] == STOP_LOSS
    assert result.returns["A"] == pytest.approx(-0.05)


def test_ギャップダウンではトリガー価格ではなく寄り値で約定する():
    """損切りラインを飛び越えて寄ったら、実際の損失はラインより大きい."""
    dates = pd.bdate_range("2024-06-03", periods=2)
    m = _matrices({"A": [(100, 100, 100, 100), (88, 90, 86, 89)]}, dates)
    result = simulate_exits(
        codes=pd.Index(["A"]),
        entry_price=pd.Series({"A": 100.0}),
        final_price=pd.Series({"A": 89.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        stop_loss=0.05,
    )
    assert result.reasons["A"] == STOP_LOSS
    assert result.returns["A"] == pytest.approx(-0.12)   # -5% では済まない


def test_銘柄ごとに異なる損切り幅を渡せる():
    dates = pd.bdate_range("2024-06-03", periods=2)
    m = _matrices(
        {"A": [(100, 100, 100, 100), (100, 101, 93, 94)],
         "B": [(100, 100, 100, 100), (100, 101, 93, 94)]},
        dates,
    )
    result = simulate_exits(
        codes=pd.Index(["A", "B"]),
        entry_price=pd.Series({"A": 100.0, "B": 100.0}),
        final_price=pd.Series({"A": 94.0, "B": 94.0}),
        final_date=dates[-1],
        window_dates=dates[1:],
        opens=m["open"], highs=m["high"], lows=m["low"],
        stop_loss=pd.Series({"A": 0.05, "B": 0.10}),
    )
    assert result.reasons["A"] == STOP_LOSS
    assert result.reasons["B"] == HELD


# ------------------------------------------------------------------ エンジン


class _Cfg:
    def __init__(self, **kwargs):
        self.holding = kwargs.get("holding", {})
        self.backtest = kwargs.get("backtest", {})
        self.portfolio = kwargs.get("portfolio", {})
        self.score_weights = {}


def _panel(n_weeks: int = 8, n_codes: int = 6, daily_return: float = 0.0):
    dates = pd.bdate_range("2024-01-01", periods=n_weeks * 5)
    rows = []
    for i in range(n_codes):
        price = 1000 * (1 + daily_return) ** np.arange(len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Code": f"{1000 + i}0",
                    "open": price, "high": price, "low": price, "close": price,
                    "raw_close": price, "vol_20": 0.3, "atr_pct": 0.02,
                    "sector33": "3050", "tradable": True, "score": float(n_codes - i),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _cfg(**holding):
    return _Cfg(
        holding=holding,
        backtest={"initial_capital": 1_000_000, "cost_bps_oneway": 20, "max_gross": 1.0},
        portfolio={"n_positions": 3, "weighting": "equal", "max_weight": 1.0,
                   "max_sector_weight": 1.0, "sector_field": "sector33"},
    )


def test_週末ノーポジなら毎週全部売って全部買い直す():
    result = run_backtest(_panel(), _cfg(weekend_flat=True))
    # 買い 1.0 + 売り 1.0 = 回転率 2.0
    assert result.equity["turnover"].iloc[0] == pytest.approx(2.0, abs=1e-6)
    assert result.diagnostics["weekend_flat"] is True


def test_持ち越すなら銘柄が同じ週の回転率はゼロ():
    result = run_backtest(_panel(), _cfg(weekend_flat=False))
    assert result.equity["turnover"].iloc[1:].max() == pytest.approx(0.0, abs=1e-9)


def test_週末ノーポジはコストが倍以上かかる():
    """ギャップリスクを避ける代わりに何を払うのかを数字で押さえる."""
    hold = run_backtest(_panel(), _cfg(weekend_flat=False))
    flat = run_backtest(_panel(), _cfg(weekend_flat=True))

    assert flat.diagnostics["total_cost_paid"] > hold.diagnostics["total_cost_paid"] * 2


def test_損切り設定がバックテストにも効く():
    panel = _panel(daily_return=-0.01)   # 毎日下げ続ける
    without = run_backtest(panel, _cfg())
    with_stop = run_backtest(panel, _cfg(stop_loss=0.02))

    assert with_stop.diagnostics["exits"].get("stop_loss", 0) > 0
    # 下げ続ける相場では、損切りがあるほうが総リターンは悪化しない
    assert with_stop.equity["gross_return"].sum() > without.equity["gross_return"].sum()
