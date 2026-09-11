"""週次リバランスのバックテスト.

前提を明示しておく:
  - 判断は「週の最終営業日の引け後」。使う特徴量はその時点で入手済みのものだけ。
  - 執行は「翌営業日の寄り付き」。判断日の引け値では約定できない。
  - コストは片道 bps で売買代金に比例させる。週次リバランスは年 50 回転する
    ので、ここを甘く見積もると机上だけ勝つ曲線になる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from takochu.strategy.score import build_scores, select_portfolio

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.DataFrame          # 期ごとの資産推移
    trades: pd.DataFrame          # 期ごとの保有銘柄とウェイト
    diagnostics: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.set_index("date")["return"]


def _price_matrix(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return panel.pivot_table(index="Date", columns="Code", values=column, aggfunc="last")


def run_backtest(
    panel: pd.DataFrame,
    config,
    decision_dates: pd.DatetimeIndex | None = None,
) -> BacktestResult:
    """特徴量パネルから週次リバランス戦略をシミュレートする."""
    bt_cfg = config.backtest if hasattr(config, "backtest") else config["backtest"]
    pf_cfg = config.portfolio if hasattr(config, "portfolio") else config["portfolio"]
    weights_cfg = (
        config.score_weights if hasattr(config, "score_weights") else config["score_weights"]
    )

    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])

    start = bt_cfg.get("start")
    end = bt_cfg.get("end")
    if start:
        panel = panel[panel["Date"] >= pd.Timestamp(start)]
    if end:
        panel = panel[panel["Date"] <= pd.Timestamp(end)]
    if panel.empty:
        raise ValueError("バックテスト期間にデータがありません")

    # すでに score 列があればそれを使う。LLM スコアを足したパネルや、
    # 重みを変えて試したパネルをそのまま流し込めるようにするため。
    if "score" not in panel.columns:
        log.info("スコアを計算中... (%d 行)", len(panel))
        panel = build_scores(panel, weights_cfg, pf_cfg.get("sector_field", "sector33"))

    if decision_dates is None:
        from takochu.features.build import weekly_decision_dates

        decision_dates = weekly_decision_dates(panel)

    all_dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    opens = _price_matrix(panel, "open")
    closes = _price_matrix(panel, "close")
    # 寄り値が欠ける日は同日終値で代用する（薄商いの銘柄で起きる）
    exec_px = opens.combine_first(closes)

    cost_rate = float(bt_cfg.get("cost_bps_oneway", 20)) / 10_000.0
    capital = float(bt_cfg.get("initial_capital", 10_000_000))

    rows: list[dict] = []
    trade_rows: list[pd.DataFrame] = []
    # 値動きを反映した「現在の実ウェイト」。目標ウェイトと比較して回転率を出す。
    held_weights = pd.Series(dtype="float64")
    equity = capital
    missing_price_events = 0

    # 判断日 -> 執行日（翌営業日）の対応を作る
    schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for d in decision_dates:
        idx = all_dates.searchsorted(d, side="right")
        if idx >= len(all_dates):
            continue  # 執行日がまだ来ていない最終週は捨てる
        schedule.append((d, all_dates[idx]))

    for i, (decision_date, exec_date) in enumerate(schedule):
        next_exec = schedule[i + 1][1] if i + 1 < len(schedule) else all_dates[-1]
        if next_exec <= exec_date:
            break

        snapshot = panel[panel["Date"] == decision_date]
        target = select_portfolio(snapshot, pf_cfg)
        target_w = (
            target.set_index("Code")["weight"] if not target.empty else pd.Series(dtype="float64")
        )
        target_w = target_w * float(bt_cfg.get("max_gross", 1.0))

        # 売買回転は「今の実ウェイト」と「目標ウェイト」の差。前期の"目標"と
        # 比べるとドリフト分を無視してコストを過小評価する。
        turnover = float(target_w.subtract(held_weights, fill_value=0.0).abs().sum())
        cost = turnover * cost_rate

        if len(target_w) == 0:
            period_return = 0.0
            held_weights = pd.Series(dtype="float64")
        else:
            codes = target_w.index
            p0 = exec_px.reindex(index=[exec_date], columns=codes).iloc[0]
            p1 = exec_px.reindex(index=[next_exec], columns=codes).iloc[0]
            # 値が取れない銘柄（売買停止・上場廃止）はリターン 0 として扱い、
            # 次回リバランスで自然に外れる。件数は diagnostics に残す。
            bad = p0.isna() | p1.isna() | (p0 <= 0)
            missing_price_events += int(bad.sum())
            stock_return = (p1 / p0 - 1.0).where(~bad, 0.0).astype("float64")
            period_return = float((target_w * stock_return).sum())

            grown = target_w * (1.0 + stock_return)
            total = grown.sum()
            held_weights = grown / total if total else grown

        net_return = period_return - cost
        equity *= 1.0 + net_return

        rows.append(
            {
                "decision_date": decision_date,
                "exec_date": exec_date,
                "date": next_exec,
                "n_positions": int(len(target_w)),
                "gross_return": period_return,
                "turnover": turnover,
                "cost": cost,
                "return": net_return,
                "equity": equity,
            }
        )
        if not target.empty:
            trade_rows.append(target.assign(decision_date=decision_date, exec_date=exec_date))

    equity_df = pd.DataFrame(rows)
    trades_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    diagnostics = {
        "periods": len(equity_df),
        "missing_price_events": missing_price_events,
        "avg_positions": float(equity_df["n_positions"].mean()) if len(equity_df) else 0.0,
        "avg_turnover": float(equity_df["turnover"].mean()) if len(equity_df) else 0.0,
        "total_cost_paid": float(equity_df["cost"].sum()) if len(equity_df) else 0.0,
    }
    return BacktestResult(equity=equity_df, trades=trades_df, diagnostics=diagnostics)
