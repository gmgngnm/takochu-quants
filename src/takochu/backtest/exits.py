"""保有中の手仕舞い判定.

週次リバランスの合間でも、利確・損切りの水準に触れたらそこで降りる。

日足しか無い以上、ザラ場中の正確な約定は再現できない。ここでは
**その日の高値・安値がトリガー価格に届いたら、その価格で約定した**
とみなす近似を使う。同じ日に利確と損切りの両方に触れた場合、
どちらが先だったかは日足から分からないので、保守的に損切りを採用する。

実運用でこの近似が崩れるのは、ギャップで飛んだとき。寄り付きが
すでにトリガーを超えていれば、約定するのは寄り値であってトリガー価格
ではない。その分は下でギャップとして扱っている。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HELD = "held"           # 最後まで持ち切った
TAKE_PROFIT = "take_profit"
STOP_LOSS = "stop_loss"


@dataclass
class ExitResult:
    returns: pd.Series          # 銘柄ごとの期間リターン
    reasons: pd.Series          # HELD / TAKE_PROFIT / STOP_LOSS
    exit_dates: pd.Series

    @property
    def triggered(self) -> pd.Series:
        return self.reasons != HELD


def simulate_exits(
    codes: pd.Index,
    entry_price: pd.Series,
    final_price: pd.Series,
    final_date: pd.Timestamp,
    window_dates: pd.DatetimeIndex,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    take_profit: float | pd.Series | None = None,
    stop_loss: float | pd.Series | None = None,
) -> ExitResult:
    """保有期間中の手仕舞いを判定する.

    Args:
        entry_price: 執行日の寄り値（エントリー価格）。
        final_price: 手仕舞い予定日の価格（次回執行日の寄り、または金曜引け）。
        window_dates: エントリー翌日から手仕舞い予定日までの営業日。
        take_profit / stop_loss: エントリー価格からの変化率。銘柄ごとに
            変えたい場合は codes を index とする Series でも渡せる。None で無効。
    """
    entry = entry_price.astype("float64")
    valid = entry.notna() & (entry > 0)

    returns = (final_price.astype("float64") / entry - 1.0)
    reasons = pd.Series(HELD, index=codes, dtype="object")
    exit_dates = pd.Series(final_date, index=codes, dtype="object")

    if take_profit is None and stop_loss is None:
        return ExitResult(returns.where(valid, 0.0), reasons, exit_dates)

    tp_price = entry * (1.0 + take_profit) if take_profit is not None else None
    sl_price = entry * (1.0 - stop_loss) if stop_loss is not None else None
    open_positions = valid.copy()

    for date in window_dates:
        if not open_positions.any():
            break
        day_open = opens.reindex(index=[date], columns=codes).iloc[0].astype("float64")
        day_high = highs.reindex(index=[date], columns=codes).iloc[0].astype("float64")
        day_low = lows.reindex(index=[date], columns=codes).iloc[0].astype("float64")
        tradeable = open_positions & day_high.notna() & day_low.notna()
        if not tradeable.any():
            continue

        # 損切りを先に見る。同日に両方へ触れた場合は保守的に損切り側を採用する。
        if sl_price is not None:
            hit = tradeable & (day_low <= sl_price)
            if hit.any():
                # ギャップダウンで寄り付きが既に下なら、約定するのは寄り値
                fill = np.minimum(sl_price[hit], day_open[hit].fillna(sl_price[hit]))
                returns.loc[hit] = fill / entry[hit] - 1.0
                reasons.loc[hit] = STOP_LOSS
                exit_dates.loc[hit] = date
                open_positions &= ~hit
                tradeable &= ~hit

        if tp_price is not None and tradeable.any():
            hit = tradeable & (day_high >= tp_price)
            if hit.any():
                # ギャップアップなら寄り値で約定する（有利側も同様に扱う）
                fill = np.maximum(tp_price[hit], day_open[hit].fillna(tp_price[hit]))
                returns.loc[hit] = fill / entry[hit] - 1.0
                reasons.loc[hit] = TAKE_PROFIT
                exit_dates.loc[hit] = date
                open_positions &= ~hit

    return ExitResult(returns.where(valid, 0.0), reasons, exit_dates)


def resolve_stop_loss(
    config: dict, snapshot: pd.DataFrame, codes: pd.Index
) -> tuple[float | None, pd.Series | None]:
    """設定から損切り幅を決める.

    固定率（stop_loss）と ATR 倍率（stop_loss_atr）の両方が指定された場合、
    銘柄ごとに**浅いほう**を採用する。深いほうを選ぶと、意図した最大損失を
    超えてしまうため。
    """
    fixed = config.get("stop_loss")
    atr_mult = config.get("stop_loss_atr")
    if atr_mult is None:
        return (float(fixed) if fixed is not None else None), None

    atr = (
        snapshot.set_index("Code")["atr_pct"].reindex(codes).astype("float64")
        if "atr_pct" in snapshot.columns
        else pd.Series(np.nan, index=codes)
    )
    per_stock = atr * float(atr_mult)
    if fixed is not None:
        per_stock = per_stock.clip(upper=float(fixed)).fillna(float(fixed))
    return (float(fixed) if fixed is not None else None), per_stock
