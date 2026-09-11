"""ユニバース構築: プライム × 流動性 × 規模.

「その日時点で」条件を満たしていたかを日次で判定する。上場廃止銘柄も
データに残っている限りそのまま扱うので、生存バイアスが入らない。
"""

from __future__ import annotations

import pandas as pd

from takochu.pit import add_available_date, asof_latest

SHARES_FIELD = "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"


def prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """日足を扱いやすい形に整える.

    株式分割の影響を消すため、価格・出来高は調整済み系列を使う。
    生の Close は発注数量の計算に必要なので残しておく。
    """
    df = quotes.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)

    for adj, raw in [
        ("AdjustmentOpen", "Open"),
        ("AdjustmentHigh", "High"),
        ("AdjustmentLow", "Low"),
        ("AdjustmentClose", "Close"),
        ("AdjustmentVolume", "Volume"),
    ]:
        target = raw.lower()
        if adj in df.columns:
            df[target] = pd.to_numeric(df[adj], errors="coerce")
            # 調整値が欠けている日は生値で埋める
            df[target] = df[target].fillna(pd.to_numeric(df.get(raw), errors="coerce"))
        else:
            df[target] = pd.to_numeric(df.get(raw), errors="coerce")

    df["raw_close"] = pd.to_numeric(df.get("Close"), errors="coerce")
    df["turnover"] = pd.to_numeric(df.get("TurnoverValue"), errors="coerce")

    keep = ["Date", "Code", "open", "high", "low", "close", "volume", "raw_close", "turnover"]
    df = df[keep].dropna(subset=["close"])
    return df.sort_values(["Code", "Date"]).reset_index(drop=True)


def latest_listed_info(listed: pd.DataFrame) -> pd.DataFrame:
    """銘柄マスタの履歴から、各銘柄の基準日ごとのレコードを整える."""
    df = listed.copy()
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Code", "Date"])
    return df


def build_universe(
    quotes: pd.DataFrame,
    listed: pd.DataFrame,
    statements: pd.DataFrame,
    config: dict,
    trading_days: pd.Series | None = None,
) -> pd.DataFrame:
    """日次のユニバース判定テーブルを返す.

    戻り値の列:
        Date, Code, close, turnover, adv, market_cap, sector17, sector33,
        market_code, listing_days, in_universe
    """
    px = prepare_quotes(quotes)
    adv_window = int(config.get("adv_window", 20))

    grouped = px.groupby("Code", sort=False)
    px["adv"] = grouped["turnover"].transform(
        lambda s: s.rolling(adv_window, min_periods=adv_window).mean()
    )
    # 上場からの経過日数。IPO 直後の値動きは別物なので除外するために使う。
    px["listing_days"] = grouped.cumcount()

    # --- 市場区分・業種を PIT で貼る -------------------------------------
    info = latest_listed_info(listed)
    info_cols = [c for c in ["MarketCode", "Sector17Code", "Sector33Code"] if c in info.columns]
    px = asof_latest(
        px,
        info.rename(columns={"Date": "available_date"}),
        value_columns=info_cols,
        facts_key="Code",
    )

    # --- 発行済株式数を PIT で貼り、時価総額を出す -----------------------
    if not statements.empty and SHARES_FIELD in statements.columns:
        st = add_available_date(statements, trading_days)
        st["LocalCode"] = st["LocalCode"].astype(str)
        st[SHARES_FIELD] = pd.to_numeric(st[SHARES_FIELD], errors="coerce")
        st = st.dropna(subset=[SHARES_FIELD])
        px = asof_latest(px, st, value_columns=[SHARES_FIELD])
        px["shares"] = px[SHARES_FIELD]
    else:
        px["shares"] = pd.NA

    px["market_cap"] = pd.to_numeric(px["shares"], errors="coerce") * px["raw_close"]

    # --- 条件判定 ---------------------------------------------------------
    market_codes = set(config.get("market_codes", []))
    cond = pd.Series(True, index=px.index)
    if market_codes and "MarketCode" in px.columns:
        cond &= px["MarketCode"].astype(str).isin(market_codes)
    cond &= px["adv"] >= float(config.get("min_adv_yen", 0))
    cond &= px["listing_days"] >= int(config.get("min_listing_days", 0))

    min_cap = float(config.get("min_market_cap_yen", 0))
    if min_cap > 0:
        # 株式数が取れない銘柄を時価総額で落とすと、財務データの
        # 取り込み漏れがそのままユニバースの欠落になる。判定は保守的に
        # 「取れている場合のみ弾く」とし、欠損は別途 doctor で検出する。
        cond &= (px["market_cap"] >= min_cap) | px["market_cap"].isna()

    px["in_universe"] = cond

    px = px.rename(
        columns={
            "Sector17Code": "sector17",
            "Sector33Code": "sector33",
            "MarketCode": "market_code",
        }
    )
    keep = [
        "Date", "Code", "open", "high", "low", "close", "raw_close", "volume", "turnover",
        "adv", "market_cap", "listing_days", "in_universe",
    ]
    keep += [c for c in ["sector17", "sector33", "market_code"] if c in px.columns]
    return px[keep].sort_values(["Date", "Code"]).reset_index(drop=True)
