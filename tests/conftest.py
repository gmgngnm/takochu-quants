"""合成データのフィクスチャ.

J-Quants の認証情報がなくてもテストが回るように、API と同じ列名・
同じ型（数値が文字列で来る点も含む）で偽データを組み立てる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.universe import SHARES_FIELD


@pytest.fixture
def trading_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-06", "2023-12-29")


@pytest.fixture
def codes() -> list[str]:
    return [f"{1300 + i * 7}0" for i in range(40)]


@pytest.fixture
def quotes(trading_days, codes) -> pd.DataFrame:
    """ランダムウォークの日足。銘柄ごとにドリフトを変えて順位が付くようにする."""
    rng = np.random.default_rng(42)
    frames = []
    for i, code in enumerate(codes):
        n = len(trading_days)
        drift = (i - len(codes) / 2) * 0.00012
        ret = rng.normal(drift, 0.018, n)
        close = 1000 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
        open_ = close * (1 + rng.normal(0, 0.004, n))
        volume = rng.integers(200_000, 2_000_000, n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "Date": trading_days.strftime("%Y-%m-%d"),
                    "Code": code,
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "TurnoverValue": close * volume,
                    "AdjustmentOpen": open_,
                    "AdjustmentHigh": high,
                    "AdjustmentLow": low,
                    "AdjustmentClose": close,
                    "AdjustmentVolume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def listed(trading_days, codes) -> pd.DataFrame:
    """四半期ごとの銘柄マスタ断面."""
    snapshots = trading_days[::60]
    rows = []
    for day in snapshots:
        for i, code in enumerate(codes):
            rows.append(
                {
                    "Date": day.strftime("%Y-%m-%d"),
                    "Code": code,
                    "CompanyName": f"テスト{code}",
                    "MarketCode": "0111" if i < 35 else "0112",
                    "Sector17Code": str(1 + i % 5),
                    "Sector33Code": str(3050 + i % 8),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def statements(codes) -> pd.DataFrame:
    """四半期ごとの決算短信。数値は API と同様に文字列で入れる."""
    rng = np.random.default_rng(7)
    rows = []
    number = 0
    for i, code in enumerate(codes):
        base_sales = 50_000 + i * 3_000
        for year in (2020, 2021, 2022, 2023):
            for q, (quarter, month) in enumerate(
                [("1Q", 8), ("2Q", 11), ("3Q", 2), ("FY", 5)], start=1
            ):
                disclosed_year = year + 1 if month < 6 else year
                disclosed = pd.Timestamp(f"{disclosed_year}-{month:02d}-10")
                growth = 1 + 0.05 * (year - 2020) + rng.normal(0, 0.03)
                sales = base_sales * growth * q / 4
                op = sales * (0.08 + rng.normal(0, 0.01))
                fc_op = base_sales * growth * 0.08 * (1 + 0.02 * q)
                number += 1
                rows.append(
                    {
                        "DisclosureNumber": str(number),
                        "LocalCode": code,
                        "DisclosedDate": disclosed.strftime("%Y-%m-%d"),
                        # 半分は引け後開示にして PIT ロジックを踏ませる
                        "DisclosedTime": "15:30" if i % 2 else "12:00",
                        "TypeOfDocument": f"{quarter}FinancialStatements_Consolidated_JP",
                        "TypeOfCurrentPeriod": quarter,
                        "CurrentPeriodEndDate": (
                            disclosed - pd.Timedelta(days=40)
                        ).strftime("%Y-%m-%d"),
                        "CurrentFiscalYearEndDate": f"{year + 1}-03-31",
                        "NetSales": f"{sales:.0f}",
                        "OperatingProfit": f"{op:.0f}",
                        "OrdinaryProfit": f"{op * 1.02:.0f}",
                        "Profit": f"{op * 0.68:.0f}",
                        "EarningsPerShare": f"{op * 0.68 / 100:.2f}",
                        "TotalAssets": f"{sales * 3:.0f}",
                        "Equity": f"{sales * 1.5:.0f}",
                        "EquityToAssetRatio": "0.5",
                        "BookValuePerShare": f"{sales * 1.5 / 100:.2f}",
                        "ForecastNetSales": f"{base_sales * growth:.0f}",
                        "ForecastOperatingProfit": f"{fc_op:.0f}",
                        "ForecastProfit": f"{base_sales * growth * 0.055:.0f}",
                        "ForecastEarningsPerShare": f"{base_sales * growth * 0.055 / 100:.2f}",
                        SHARES_FIELD: "100000000",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    from takochu.config import load_config

    cfg = load_config()
    # 合成データは銘柄数が少ないので閾値を緩める
    cfg.universe["min_adv_yen"] = 0
    cfg.universe["min_market_cap_yen"] = 0
    cfg.universe["min_listing_days"] = 30
    cfg.portfolio["n_positions"] = 8
    cfg.backtest["start"] = "2021-01-01"
    return cfg


@pytest.fixture
def null_quotes(trading_days, codes) -> pd.DataFrame:
    """予測不能な日足（銘柄間にドリフト差が無い）.

    このデータで IC が立つなら、それは戦略の予測力ではなく
    どこかに未来の情報が漏れている証拠になる。
    """
    rng = np.random.default_rng(20240101)
    frames = []
    for code in codes:
        n = len(trading_days)
        close = 1000 * np.exp(np.cumsum(rng.normal(0.0, 0.018, n)))
        volume = rng.integers(200_000, 2_000_000, n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "Date": trading_days.strftime("%Y-%m-%d"),
                    "Code": code,
                    "Open": close * (1 + rng.normal(0, 0.004, n)),
                    "High": close * 1.005,
                    "Low": close * 0.995,
                    "Close": close,
                    "Volume": volume,
                    "TurnoverValue": close * volume,
                    "AdjustmentClose": close,
                    "AdjustmentVolume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
