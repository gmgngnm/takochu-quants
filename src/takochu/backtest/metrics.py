"""評価指標.

損益曲線の見栄えより、Information Coefficient（スコアと将来リターンの
順位相関）を重視する。IC が立たないのに損益が良い場合、それは
少数の銘柄の偶然か、どこかに look-ahead が残っている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52


def summarize(equity: pd.DataFrame, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """週次リターン列から運用指標を計算する."""
    if equity is None or equity.empty:
        return {}

    r = equity["return"].astype(float)
    curve = (1.0 + r).cumprod()
    n = len(r)
    years = n / periods_per_year

    total_return = float(curve.iloc[-1] - 1.0)
    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    sharpe = float(cagr / vol) if vol and not np.isnan(vol) and vol > 0 else np.nan

    drawdown = curve / curve.cummax() - 1.0
    max_dd = float(drawdown.min())

    gross = equity["gross_return"].astype(float)
    cost = equity["cost"].astype(float)

    return {
        "periods": n,
        "years": round(years, 2),
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "avg_turnover": float(equity["turnover"].mean()),
        # コストが年率で何%を食っているか。週次リバランスではここが効く。
        "cost_drag_annual": float(cost.mean() * periods_per_year),
        "gross_cagr_gap": float(gross.mean() - r.mean()) * periods_per_year,
    }


def information_coefficient(
    panel: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    score_column: str = "score",
    horizon_days: int = 5,
) -> pd.DataFrame:
    """判断日ごとの Rank IC（スコア順位 vs 先行リターン順位の相関）.

    スコアに予測力があるかを、ポートフォリオ構築の仕方と切り離して測る。
    """
    closes = panel.pivot_table(index="Date", columns="Code", values="close", aggfunc="last")
    forward = closes.shift(-horizon_days) / closes - 1.0

    rows = []
    for date in decision_dates:
        if date not in forward.index:
            continue
        snapshot = panel[panel["Date"] == date]
        if "tradable" in snapshot.columns:
            snapshot = snapshot[snapshot["tradable"].astype(bool)]
        snapshot = snapshot.dropna(subset=[score_column])
        if len(snapshot) < 30:
            continue

        fwd = forward.loc[date].reindex(snapshot["Code"].values)
        pair = pd.DataFrame(
            {"score": snapshot[score_column].values, "fwd": fwd.values}
        ).dropna()
        if len(pair) < 30:
            continue
        # 順位相関 = 順位に直した上での Pearson 相関。
        # pandas の method="spearman" は scipy を要求するので自前で計算する。
        ranked = pair.rank()
        rows.append(
            {
                "date": date,
                "n": len(pair),
                "ic": float(ranked["score"].corr(ranked["fwd"])),
            }
        )

    return pd.DataFrame(rows)


def ic_summary(ic: pd.DataFrame) -> dict:
    """IC の平均と情報比。|mean IC| が 0.02〜0.05 あれば実用域."""
    if ic is None or ic.empty:
        return {}
    values = ic["ic"].dropna()
    if values.empty:
        return {}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_ir": float(mean / std) if std > 0 else np.nan,
        "positive_rate": float((values > 0).mean()),
        "n_periods": int(len(values)),
    }
