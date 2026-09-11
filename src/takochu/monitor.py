"""保有中の手仕舞い監視.

週次リバランスの合間に、利確・損切りの水準へ到達した銘柄を洗い出す。
毎日引け後に回す想定。

**日足ベースであることの意味**: J-Quants は日足しか返さないので、
到達を知るのは最短でも当日の引け後になる。実際に手仕舞えるのは翌営業日。
ザラ場中にラインへ触れた瞬間に発注したいなら、板と約定を受け取れる
証券会社の API（kabuステーション / 立花証券e支店）が要る。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from takochu.backtest.exits import STOP_LOSS, TAKE_PROFIT

log = logging.getLogger(__name__)


@dataclass
class Alert:
    code: str
    reason: str                 # take_profit / stop_loss
    entry_date: str
    entry_price: float
    trigger_price: float
    last_close: float
    change: float               # エントリーからの変化率（直近終値ベース）
    touched_on: str             # ラインに到達した日
    weight: float | None = None


def check_holdings(
    panel: pd.DataFrame,
    holdings: pd.DataFrame,
    decision_date: pd.Timestamp,
    holding_config: dict,
) -> tuple[list[Alert], pd.DataFrame]:
    """保有銘柄を点検し、手仕舞い推奨と現況テーブルを返す.

    Args:
        panel: 特徴量パネル（最新の日足まで入っていること）。
        holdings: picks_history の最新レコード（Code, weight）。
        decision_date: その保有を決めた判断日。エントリーは翌営業日の寄り。
    """
    take_profit = holding_config.get("take_profit")
    stop_loss = holding_config.get("stop_loss")
    atr_mult = holding_config.get("stop_loss_atr")

    dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    idx = dates.searchsorted(decision_date, side="right")
    if idx >= len(dates):
        log.warning("判断日の翌営業日がまだ来ていません")
        return [], pd.DataFrame()

    entry_date = dates[idx]
    last_date = dates[-1]
    codes = holdings["Code"].astype(str).tolist()

    held = panel[panel["Code"].astype(str).isin(codes)].copy()
    held["Code"] = held["Code"].astype(str)
    window = held[(held["Date"] >= entry_date) & (held["Date"] <= last_date)]
    if window.empty:
        return [], pd.DataFrame()

    weights = holdings.set_index(holdings["Code"].astype(str))["weight"].to_dict()
    alerts: list[Alert] = []
    rows = []

    for code, group in window.groupby("Code", sort=False):
        group = group.sort_values("Date")
        entry_row = group.iloc[0]
        entry_price = float(entry_row.get("open") or entry_row["close"])
        if not entry_price or pd.isna(entry_price):
            continue

        last = group.iloc[-1]
        last_close = float(last["close"])
        change = last_close / entry_price - 1.0

        stop = _stop_width(stop_loss, atr_mult, entry_row)
        tp_price = entry_price * (1 + take_profit) if take_profit is not None else None
        sl_price = entry_price * (1 - stop) if stop is not None else None

        # 到達判定は終値ではなく高値・安値で見る。ザラ場で触れていれば
        # 逆指値なら約定しているはずなので、それを見落とさないため。
        hit_stop = group[group["low"] <= sl_price] if sl_price else group.iloc[0:0]
        hit_tp = group[group["high"] >= tp_price] if tp_price else group.iloc[0:0]

        reason = touched_on = trigger = None
        if not hit_stop.empty:
            reason, touched_on, trigger = STOP_LOSS, hit_stop.iloc[0]["Date"], sl_price
        elif not hit_tp.empty:
            reason, touched_on, trigger = TAKE_PROFIT, hit_tp.iloc[0]["Date"], tp_price

        rows.append(
            {
                "Code": code,
                "weight": weights.get(code),
                "entry_price": entry_price,
                "last_close": last_close,
                "change": change,
                "stop_price": sl_price,
                "take_price": tp_price,
                "reason": reason or "",
            }
        )
        if reason:
            alerts.append(
                Alert(
                    code=code,
                    reason=reason,
                    entry_date=f"{entry_date:%Y-%m-%d}",
                    entry_price=entry_price,
                    trigger_price=float(trigger),
                    last_close=last_close,
                    change=change,
                    touched_on=f"{pd.Timestamp(touched_on):%Y-%m-%d}",
                    weight=weights.get(code),
                )
            )

    status = pd.DataFrame(rows).sort_values("change", ascending=False).reset_index(drop=True)
    # 損切り -> 利確 の順。先に見るべきものを上に。
    alerts.sort(key=lambda a: (a.reason != STOP_LOSS, a.change))
    return alerts, status


def _stop_width(stop_loss, atr_mult, row) -> float | None:
    """固定率と ATR 倍率の浅いほうを採る（意図した最大損失を超えないように）."""
    widths = []
    if stop_loss is not None:
        widths.append(float(stop_loss))
    if atr_mult is not None:
        atr = row.get("atr_pct")
        if atr is not None and not pd.isna(atr):
            widths.append(float(atr) * float(atr_mult))
    return min(widths) if widths else None


def alerts_to_records(alerts: list[Alert]) -> list[dict]:
    return [asdict(a) for a in alerts]
