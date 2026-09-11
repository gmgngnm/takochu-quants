"""ファンダメンタルズ特徴量.

方針: バリュエーションの「水準」ではなく、業績見通しの「変化」を測る。
1週間スケールで価格に効くのは PER の絶対値ではなく、
会社予想の上方/下方修正と決算サプライズ（PEAD）だから。

出力は「開示 1 行 = 1 レコード」の facts テーブルで、`available_date` を持つ。
これを ASOF JOIN でパネルに貼ると PIT が保たれる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from takochu.pit import add_available_date

# 累計実績。四半期報告では期初からの累計値が入る。
ACTUAL_FIELDS = {
    "net_sales": "NetSales",
    "operating_profit": "OperatingProfit",
    "ordinary_profit": "OrdinaryProfit",
    "profit": "Profit",
}
# 通期の会社予想
FORECAST_FIELDS = {
    "fc_net_sales": "ForecastNetSales",
    "fc_operating_profit": "ForecastOperatingProfit",
    "fc_profit": "ForecastProfit",
    "fc_eps": "ForecastEarningsPerShare",
}
BALANCE_FIELDS = {
    "total_assets": "TotalAssets",
    "equity": "Equity",
    "equity_ratio": "EquityToAssetRatio",
    "bps": "BookValuePerShare",
    "eps": "EarningsPerShare",
}

QUARTER_NUMBER = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    """J-Quants は欠損を空文字で返すことがあるので必ず数値化を通す."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    """符号反転（赤字→黒字など）で比率が意味を失うため、分母は絶対値を使う."""
    denom = old.abs().replace(0.0, np.nan)
    return (new - old) / denom


def build_fundamental_facts(
    statements: pd.DataFrame,
    trading_days: pd.Series | None = None,
) -> pd.DataFrame:
    """決算短信から PIT な facts テーブルを作る."""
    if statements.empty:
        return pd.DataFrame()

    df = add_available_date(statements, trading_days)
    df["LocalCode"] = df["LocalCode"].astype(str)
    df["DisclosureNumber"] = pd.to_numeric(df["DisclosureNumber"], errors="coerce")
    df["period_end"] = pd.to_datetime(df.get("CurrentPeriodEndDate"), errors="coerce")
    df["fy_end"] = pd.to_datetime(df.get("CurrentFiscalYearEndDate"), errors="coerce")
    df["quarter"] = df.get("TypeOfCurrentPeriod", pd.Series(index=df.index)).map(QUARTER_NUMBER)

    for alias, column in {**ACTUAL_FIELDS, **FORECAST_FIELDS, **BALANCE_FIELDS}.items():
        df[alias] = _num(df, column)

    df = df.sort_values(["LocalCode", "available_date", "DisclosureNumber"]).reset_index(drop=True)

    df = _add_forecast_revision(df)
    df = _add_yoy(df)
    df = _add_progress(df)
    df = _add_quality(df)

    feature_cols = [
        "revision_op", "revision_sales", "revision_age_days",
        "sales_yoy", "op_yoy", "profit_yoy", "op_margin", "op_margin_delta",
        "progress_gap", "quarter",
        "roe", "equity_ratio", "eps_ttm", "bps",
        "fc_operating_profit", "fc_eps",
    ]
    # 開示の種類によって埋まる列が違う（予想修正だけの開示など）。
    # 最後に判明した値を持ち越すことで、ASOF JOIN が拾う最新行が常に完全になる。
    df[feature_cols] = df.groupby("LocalCode", sort=False)[feature_cols].ffill()

    keep = ["LocalCode", "available_date", "DisclosedDate", "period_end", "fy_end", *feature_cols]
    return df[[c for c in keep if c in df.columns]]


def _add_forecast_revision(df: pd.DataFrame) -> pd.DataFrame:
    """同一決算期における会社予想の変化率。1週間スケールで最も効く特徴量."""
    out = df.copy()
    revision_targets = [
        ("fc_operating_profit", "revision_op"),
        ("fc_net_sales", "revision_sales"),
    ]
    for alias, target in revision_targets:
        prev = out.groupby(["LocalCode", "fy_end"], sort=False)[alias].transform(
            lambda s: s.where(s.notna()).ffill().shift(1)
        )
        out[target] = _pct_change(out[alias], prev)

    # 直近の修正からの経過日数。修正直後ほどドリフトが効くので減衰に使う。
    revised = out["revision_op"].notna() & (out["revision_op"].abs() > 1e-9)
    last_revision = out["available_date"].where(revised)
    last_revision = out.assign(_lr=last_revision).groupby("LocalCode", sort=False)["_lr"].ffill()
    out["revision_age_days"] = (
        pd.to_datetime(out["available_date"]) - pd.to_datetime(last_revision)
    ).dt.days
    return out


def _add_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """前年同四半期比。累計値どうしを比較するので四半期を揃える必要がある."""
    out = df.copy()
    has_actual = out["net_sales"].notna() | out["operating_profit"].notna()
    actual = out[has_actual].sort_values(["LocalCode", "quarter", "period_end"])

    for alias, target in [
        ("net_sales", "sales_yoy"),
        ("operating_profit", "op_yoy"),
        ("profit", "profit_yoy"),
    ]:
        prev = actual.groupby(["LocalCode", "quarter"], sort=False)[alias].shift(1)
        actual[target] = _pct_change(actual[alias], prev)

    actual["op_margin"] = actual["operating_profit"] / actual["net_sales"].replace(0.0, np.nan)
    prev_margin = actual.groupby(["LocalCode", "quarter"], sort=False)["op_margin"].shift(1)
    actual["op_margin_delta"] = actual["op_margin"] - prev_margin

    cols = ["sales_yoy", "op_yoy", "profit_yoy", "op_margin", "op_margin_delta"]
    return out.join(actual[cols], how="left")


def _add_progress(df: pd.DataFrame) -> pd.DataFrame:
    """通期会社予想に対する進捗率と、四半期ペースからの乖離.

    3Q 終了時点なら進捗 75% が中立。これを上回っていれば上方修正の芽があり、
    決算発表後のドリフト（PEAD）の代理変数になる。
    """
    out = df.copy()
    progress = out["operating_profit"] / out["fc_operating_profit"].replace(0.0, np.nan)
    expected = out["quarter"] / 4.0
    # 通期決算（進捗 100% で当たり前）は情報を持たないので除外する
    gap = (progress - expected).where(out["quarter"].between(1, 3))
    # 予想が赤字のときの進捗率は解釈不能
    out["progress_gap"] = gap.where(out["fc_operating_profit"] > 0)
    return out


def _add_quality(df: pd.DataFrame) -> pd.DataFrame:
    """収益性・財務健全性。変化ほどは効かないが、質の低い銘柄を落とすのに使う."""
    out = df.copy()
    actual = out[out["profit"].notna()].sort_values(["LocalCode", "period_end"]).copy()

    # 四半期累計から TTM を作る: 通期はそのまま、四半期は
    # 「直近の通期実績 - 前年同期累計 + 当期累計」で近似する。
    fy_only = actual["profit"].where(actual["quarter"] == 4)
    last_fy = fy_only.groupby(actual["LocalCode"], sort=False).ffill()
    prev_same_q = actual.groupby(["LocalCode", "quarter"], sort=False)["profit"].shift(1)

    actual["profit_ttm"] = np.where(
        actual["quarter"] == 4,
        actual["profit"],
        last_fy - prev_same_q + actual["profit"],
    )
    actual["roe"] = actual["profit_ttm"] / actual["equity"].replace(0.0, np.nan)
    actual["eps_ttm"] = actual["eps"]

    return out.join(actual[["roe", "eps_ttm"]], how="left")
