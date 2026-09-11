"""チャート特徴量.

役割は「銘柄選定」ではなく「タイミングの確認とリスク推定」。
ファンダの変化で選んだ銘柄が、需給的に逆行していないかを見る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 245
TRADING_DAYS_MONTH = 21


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_pct(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean()
    return atr / df["close"]


def add_price_features(panel: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """ユニバーステーブルに価格系特徴量を足す（銘柄ごとの時系列計算）."""
    config = config or {}
    atr_window = int(config.get("atr_window", 14))
    vol_window = int(config.get("vol_window", 20))

    df = panel.sort_values(["Code", "Date"]).copy()
    out = []

    for _, g in df.groupby("Code", sort=False):
        g = g.copy()
        close = g["close"]
        ret = close.pct_change()

        g["ret_1w"] = close.pct_change(5)
        g["ret_1m"] = close.pct_change(TRADING_DAYS_MONTH)
        # 12-1 モメンタム。直近1ヶ月を除くのは短期リバーサルを拾わないため。
        g["mom_12_1"] = close.shift(TRADING_DAYS_MONTH) / close.shift(TRADING_DAYS_YEAR) - 1

        ma20 = close.rolling(20, min_periods=20).mean()
        ma60 = close.rolling(60, min_periods=60).mean()
        ma200 = close.rolling(200, min_periods=200).mean()
        g["ma_gap_20"] = close / ma20 - 1
        g["ma_gap_60"] = close / ma60 - 1
        g["above_ma200"] = (close > ma200).astype("float").where(ma200.notna())

        high_52w = close.rolling(TRADING_DAYS_YEAR, min_periods=120).max()
        g["dist_52w_high"] = close / high_52w - 1

        g["vol_20"] = ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(
            TRADING_DAYS_YEAR
        )
        g["atr_pct"] = _atr_pct(g, atr_window)
        g["rsi_14"] = _rsi(close, 14)

        turnover_ma = g["turnover"].rolling(60, min_periods=20).mean()
        turnover_sd = g["turnover"].rolling(60, min_periods=20).std()
        g["turnover_z"] = (g["turnover"] - turnover_ma) / turnover_sd.replace(0.0, np.nan)

        # 翌営業日の寄り値。バックテストの約定価格に使う（判断は当日引け後）。
        g["next_open"] = g["open"].shift(-1)
        g["next_date"] = g["Date"].shift(-1)

        out.append(g)

    return pd.concat(out, ignore_index=True)
