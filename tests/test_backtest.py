"""バックテストエンジンのテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.backtest.engine import run_backtest
from takochu.backtest.metrics import ic_summary, information_coefficient, summarize
from takochu.features.build import build_feature_panel, weekly_decision_dates


class _Cfg:
    """run_backtest が受け取る最小限の設定オブジェクト."""

    def __init__(self, **kwargs):
        self.holding = kwargs.get("holding", {})
        self.backtest = kwargs.get("backtest", {})
        self.portfolio = kwargs.get("portfolio", {})
        self.score_weights = kwargs.get("score_weights", {})


def _flat_panel(n_days: int = 60, n_codes: int = 10, daily_return: float = 0.0):
    """全銘柄が同じリターンで動くパネル。損益とコストを解析的に検算できる."""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for i in range(n_codes):
        price = 1000 * (1 + daily_return) ** np.arange(n_days)
        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Code": f"{1000 + i}0",
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "raw_close": price,
                    "vol_20": 0.3,
                    "sector33": str(3050 + i % 3),
                    "tradable": True,
                    "score": float(n_codes - i),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_値動きゼロならコストの分だけ資産が減る():
    panel = _flat_panel(daily_return=0.0)
    cfg = _Cfg(
        backtest={"initial_capital": 1_000_000, "cost_bps_oneway": 50, "max_gross": 1.0},
        portfolio={"n_positions": 5, "weighting": "equal", "max_weight": 1.0,
                   "max_sector_weight": 1.0, "sector_field": "sector33"},
        score_weights={},
    )
    result = run_backtest(panel, cfg)

    assert (result.equity["gross_return"].abs() < 1e-12).all()
    assert (result.equity["return"] <= 0).all()
    # 初回は全額買い付けなので回転率 1.0、コストは 50bps
    assert result.equity["turnover"].iloc[0] == pytest.approx(1.0)
    assert result.equity["cost"].iloc[0] == pytest.approx(0.005)


def test_銘柄が入れ替わらなければ2期目以降の回転率はゼロ():
    panel = _flat_panel(daily_return=0.0)
    cfg = _Cfg(
        backtest={"initial_capital": 1_000_000, "cost_bps_oneway": 20, "max_gross": 1.0},
        portfolio={"n_positions": 5, "weighting": "equal", "max_weight": 1.0,
                   "max_sector_weight": 1.0, "sector_field": "sector33"},
        score_weights={},
    )
    result = run_backtest(panel, cfg)
    # スコアが固定なので同じ 5 銘柄を持ち続ける -> 値動きゼロならリバランス不要
    assert result.equity["turnover"].iloc[1:].max() == pytest.approx(0.0, abs=1e-9)


def test_上昇相場ではグロスリターンが正になる():
    panel = _flat_panel(daily_return=0.001)
    cfg = _Cfg(
        backtest={"initial_capital": 1_000_000, "cost_bps_oneway": 0, "max_gross": 1.0},
        portfolio={"n_positions": 5, "weighting": "equal", "max_weight": 1.0,
                   "max_sector_weight": 1.0, "sector_field": "sector33"},
        score_weights={},
    )
    result = run_backtest(panel, cfg)
    assert (result.equity["gross_return"] > 0).all()
    assert result.equity["equity"].iloc[-1] > 1_000_000


def test_執行は判断日の翌営業日である():
    panel = _flat_panel()
    cfg = _Cfg(
        backtest={"initial_capital": 1_000_000, "cost_bps_oneway": 0, "max_gross": 1.0},
        portfolio={"n_positions": 5, "weighting": "equal", "max_weight": 1.0,
                   "max_sector_weight": 1.0, "sector_field": "sector33"},
        score_weights={},
    )
    result = run_backtest(panel, cfg)
    assert (result.equity["exec_date"] > result.equity["decision_date"]).all()


def test_パネル構築からバックテストまで通しで動く(quotes, listed, statements, config, trading_days):
    panel = build_feature_panel(quotes, listed, statements, config, pd.Series(trading_days))

    assert len(panel) > 0
    assert panel["in_universe"].sum() > 0
    # プライム以外（MarketCode 0112）はユニバースに入らない
    assert set(panel.loc[panel["in_universe"], "market_code"].unique()) == {"0111"}

    result = run_backtest(panel, config)
    stats = summarize(result.equity)

    assert result.diagnostics["periods"] > 20
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert np.isfinite(stats["max_drawdown"])

    ic = information_coefficient(panel.assign(score=panel["mom_12_1"]),
                                 weekly_decision_dates(panel))
    assert ic_summary(ic) != {}


def test_無情報データではICが立たない(null_quotes, listed, statements, config, trading_days):
    """リーク検出.

    銘柄間に予測可能な差が無いデータでスコアが将来リターンを当ててしまうなら、
    それは戦略の実力ではなく look-ahead。バックテスト基盤そのものの健全性を
    ここで担保する。
    """
    panel = build_feature_panel(null_quotes, listed, statements, config, pd.Series(trading_days))
    from takochu.strategy.score import build_scores

    scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])
    ic = ic_summary(information_coefficient(scored, weekly_decision_dates(scored)))

    assert ic, "IC を計算できる断面がありません"
    assert abs(ic["mean_ic"]) < 0.03, (
        f"無情報データで mean_ic={ic['mean_ic']:.4f} が出ています。"
        "どこかに未来の情報が漏れている可能性が高い。"
    )
