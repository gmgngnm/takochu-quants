"""J-Quants からのデータ取り込み.

生データをそのまま Parquet に落とすだけで、加工は一切しない。
主キーで upsert するので、途中で失敗しても同じコマンドを再実行すれば
続きから埋まる（訂正開示は後から来たものが正になる）。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from takochu.io.jquants import JQuantsClient
from takochu.io.store import Store

log = logging.getLogger(__name__)

# 銘柄マスタは毎日取る必要がない。市場区分の変更を捉えられれば十分。
LISTED_INFO_INTERVAL_DAYS = 28


def _trading_days(store: Store, client: JQuantsClient, start: str, end: str) -> list[str]:
    calendar = store.read("trading_calendar")
    if calendar.empty:
        log.info("営業日カレンダーを取得中 %s ~ %s", start, end)
        calendar = client.trading_calendar(start=start, end=end)
        store.upsert("trading_calendar", calendar)

    calendar["Date"] = pd.to_datetime(calendar["Date"])
    mask = (calendar["Date"] >= pd.Timestamp(start)) & (calendar["Date"] <= pd.Timestamp(end))
    # HolidayDivision: 0=非営業日, 1=営業日, 2=半日立会
    if "HolidayDivision" in calendar.columns:
        mask &= calendar["HolidayDivision"].astype(str).isin(["1", "2"])
    days = calendar.loc[mask, "Date"].sort_values()
    return [d.strftime("%Y-%m-%d") for d in days]


def ingest_all(
    store: Store,
    client: JQuantsClient,
    start: str | None = None,
    end: str | None = None,
    datasets: list[str] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """指定期間のデータを取り込み、取り込み後のサマリを返す."""
    end = end or date.today().strftime("%Y-%m-%d")
    start = start or (date.today() - timedelta(days=365 * 8)).strftime("%Y-%m-%d")
    wanted = set(
        datasets
        or [
            "daily_quotes",
            "statements",
            "listed_info",
            "topix",
            "weekly_margin",
            "announcement",
        ]
    )

    days = _trading_days(store, client, start, end)
    if not days:
        log.warning("対象期間に営業日がありません: %s ~ %s", start, end)
        return store.summary()

    log.info("対象営業日: %d 日 (%s ~ %s)", len(days), days[0], days[-1])

    if "topix" in wanted:
        log.info("TOPIX を取得中")
        store.upsert("topix", client.topix(start=start, end=end))

    if "listed_info" in wanted:
        _ingest_listed_info(store, client, days, resume)

    for name, fetch in [
        ("daily_quotes", lambda d: client.daily_quotes(on=d)),
        ("statements", lambda d: client.statements(on=d)),
    ]:
        if name not in wanted:
            continue
        _ingest_by_day(store, name, fetch, days, resume)

    if "weekly_margin" in wanted:
        log.info("信用残を取得中")
        store.upsert("weekly_margin", client.weekly_margin_interest(start=start, end=end))

    if "announcement" in wanted:
        # 翌営業日分しか返らないので、毎日の実行で少しずつ貯まる。
        log.info("決算発表予定を取得中")
        store.upsert("announcement", client.announcement())

    return store.summary()


def _ingest_listed_info(
    store: Store, client: JQuantsClient, days: list[str], resume: bool
) -> None:
    existing: set[str] = set()
    if resume and store.row_count("listed_info"):
        known = store.read("listed_info", columns=["Date"])
        existing = set(pd.to_datetime(known["Date"]).dt.strftime("%Y-%m-%d"))

    targets = days[::LISTED_INFO_INTERVAL_DAYS]
    if days[-1] not in targets:
        targets.append(days[-1])  # 最新断面は必ず押さえる

    for i, day in enumerate(targets, 1):
        if day in existing:
            continue
        log.info("銘柄マスタ %s (%d/%d)", day, i, len(targets))
        store.upsert("listed_info", client.listed_info(on=day))


def _ingest_by_day(store: Store, name: str, fetch, days: list[str], resume: bool) -> None:
    date_column = "DisclosedDate" if name == "statements" else "Date"
    done: set[str] = set()
    if resume and store.row_count(name):
        existing = store.read(name, columns=[date_column])
        done = set(pd.to_datetime(existing[date_column]).dt.strftime("%Y-%m-%d"))
        log.info("%s: 取り込み済み %d 日分をスキップします", name, len(done))

    pending = [d for d in days if d not in done]
    log.info("%s: %d 日分を取得します", name, len(pending))

    buffer: list[pd.DataFrame] = []
    for i, day in enumerate(pending, 1):
        frame = fetch(day)
        if not frame.empty:
            buffer.append(frame)
        if i % 50 == 0 or i == len(pending):
            if buffer:
                store.upsert(name, pd.concat(buffer, ignore_index=True))
                buffer.clear()
            log.info("%s: %d/%d 日完了 (%s)", name, i, len(pending), day)

    # 開示が 1 件も無い日は done に記録されないため、毎回再取得になる。
    # 決算がまったく無い営業日は稀なので実害はない。
