"""特徴量パネルの組み立て.

ユニバース判定 + 価格特徴量 + PIT なファンダ facts を 1 枚の
(Date, Code) パネルに束ねる。バックテストもレポート生成も
このパネルだけを入力にする。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from takochu.features.fundamental import build_fundamental_facts
from takochu.features.price import add_price_features
from takochu.pit import asof_latest
from takochu.universe import build_universe

log = logging.getLogger(__name__)

FUNDAMENTAL_COLUMNS = [
    "revision_op", "revision_sales", "revision_age_days",
    "sales_yoy", "op_yoy", "profit_yoy", "op_margin", "op_margin_delta",
    "progress_gap", "roe", "equity_ratio", "eps_ttm", "bps", "fc_eps",
]


def _add_days_to_next_earnings(
    panel: pd.DataFrame,
    statements: pd.DataFrame,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """次の決算発表までの日数.

    過去分は実際の開示日を前方参照で引く。決算発表"予定日"は事前に公表される
    公開情報なので、日付を知っていること自体は look-ahead ではない（中身は当然使わない）。

    ただしパネル最終日の先には開示実績が無いため、過去データだけだと
    ライブのスクリーニング時にブラックアウトが一切効かなくなる。
    そこを J-Quants /fins/announcement（決算発表予定）で埋める。
    """
    sources = []
    if not statements.empty:
        sources.append(
            statements[["LocalCode", "DisclosedDate"]].rename(
                columns={"LocalCode": "Code", "DisclosedDate": "next_earnings_date"}
            )
        )
    if announcements is not None and not announcements.empty and "Date" in announcements.columns:
        sources.append(
            announcements[["Code", "Date"]].rename(columns={"Date": "next_earnings_date"})
        )

    if not sources:
        panel["days_to_next_earnings"] = np.nan
        return panel

    events = (
        pd.concat(sources, ignore_index=True)
        .assign(
            Code=lambda d: d["Code"].astype(str),
            next_earnings_date=lambda d: pd.to_datetime(d["next_earnings_date"], errors="coerce"),
        )
        .dropna()
        .drop_duplicates()
        .sort_values("next_earnings_date")
    )

    merged = pd.merge_asof(
        panel.sort_values("Date"),
        events,
        left_on="Date",
        right_on="next_earnings_date",
        by="Code",
        direction="forward",
        allow_exact_matches=False,
    )
    merged["days_to_next_earnings"] = (
        merged["next_earnings_date"] - merged["Date"]
    ).dt.days
    return merged


def build_feature_panel(
    quotes: pd.DataFrame,
    listed: pd.DataFrame,
    statements: pd.DataFrame,
    config,
    trading_days: pd.Series | None = None,
    announcements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """(Date, Code) の特徴量パネルを返す."""
    universe_cfg = config.universe if hasattr(config, "universe") else config["universe"]
    feature_cfg = config.features if hasattr(config, "features") else config["features"]

    log.info("ユニバースを構築中...")
    panel = build_universe(quotes, listed, statements, universe_cfg, trading_days)

    log.info("価格特徴量を計算中... (%d 行)", len(panel))
    panel = add_price_features(panel, feature_cfg)

    log.info("ファンダ facts を構築中...")
    facts = build_fundamental_facts(statements, trading_days)
    if facts.empty:
        for col in FUNDAMENTAL_COLUMNS:
            panel[col] = np.nan
    else:
        available = [c for c in FUNDAMENTAL_COLUMNS if c in facts.columns]
        panel = asof_latest(panel, facts, value_columns=[*available, "available_date"])

    panel = _add_days_to_next_earnings(panel, statements, announcements)

    # バリュエーション。絶対水準は使わず、後段で業種内 z-score にする。
    panel["per"] = panel["raw_close"] / panel["eps_ttm"].replace(0.0, np.nan)
    panel["pbr"] = panel["raw_close"] / panel["bps"].replace(0.0, np.nan)
    panel.loc[panel["per"] <= 0, "per"] = np.nan  # 赤字企業の PER は比較不能

    # 保有期間中に決算を跨ぐ銘柄はエントリ対象から外す
    blackout = int(feature_cfg.get("earnings_blackout_days", 10))
    days_to = pd.to_numeric(panel["days_to_next_earnings"], errors="coerce")
    panel["earnings_blackout"] = (days_to <= blackout).fillna(False).astype(bool)
    panel["tradable"] = panel["in_universe"].astype(bool) & ~panel["earnings_blackout"]

    return panel.sort_values(["Date", "Code"]).reset_index(drop=True)


def weekly_decision_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """各週の最終営業日（= 判断日）を返す。翌営業日の寄りで執行する."""
    dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W")
    return pd.DatetimeIndex(frame.groupby("week")["date"].max().values)
