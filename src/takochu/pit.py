"""Point-in-Time (PIT) 処理.

バックテストの成否はここで決まる。決算は「決算期末日」ではなく
「開示された瞬間」から使えるようになるので、両者を取り違えると
未来の情報で過去を判断する look-ahead bias が入り、
バックテストだけが綺麗に勝つシステムが出来上がる。

このモジュールの規約:
  - `available_date` = その情報を使って「引け値ベースの判断」ができる最初の営業日。
  - 判断時刻 D の引け後に意思決定するとき、使ってよいのは
    `available_date <= D` の行だけ。
  - 引け後に開示された情報はその日の引け値では取引できないので、
    翌営業日を `available_date` とする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 大引け。これ以降の開示はその日の引けでは取引できない。
MARKET_CLOSE = pd.Timedelta(hours=15)


def _parse_disclosed_time(series: pd.Series) -> pd.Series:
    """"15:30" / "15:30:00" 形式の開示時刻を Timedelta にする。欠損は引け後扱い.

    pandas は "15:30" のような秒なし表記を解釈しないので、先に秒を補う。
    """
    text = series.astype("string").str.strip()
    text = text.where(~text.str.fullmatch(r"\d{1,2}:\d{2}", na=False), text + ":00")
    parsed = pd.to_timedelta(text, errors="coerce")
    # 時刻が取れない行を「引け前」に倒すと look-ahead になるので、保守側（引け後）に倒す
    return parsed.fillna(MARKET_CLOSE + pd.Timedelta(minutes=1))


def add_available_date(
    statements: pd.DataFrame,
    trading_days: pd.Series | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """決算短信に `available_date` を付与する.

    Args:
        statements: J-Quants /fins/statements の生データ。
        trading_days: 営業日の並び。与えると引け後開示を「次の営業日」に送る。
            省略時は暦日で +1 日（週末を跨ぐと保守的すぎる方向にずれるが安全側）。
    """
    if statements.empty:
        return statements.assign(available_date=pd.Series(dtype="datetime64[ns]"))

    df = statements.copy()
    disclosed = pd.to_datetime(df["DisclosedDate"], errors="coerce")
    after_close = _parse_disclosed_time(df.get("DisclosedTime", pd.Series(index=df.index))) > (
        MARKET_CLOSE
    )

    available = disclosed.copy()
    if trading_days is not None:
        days = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading_days)).unique()).sort_values()
        # searchsorted("right") で「その日より後の最初の営業日」を引く
        idx = days.searchsorted(disclosed[after_close], side="right")
        idx = np.clip(idx, 0, len(days) - 1)
        next_day = pd.Series(days[idx], index=disclosed[after_close].index)
        # カレンダー末尾を超える開示は NaT にして落とす（使えないことを明示する）
        out_of_range = disclosed[after_close] >= days[-1]
        next_day[out_of_range] = pd.NaT
        available.loc[after_close] = next_day
    else:
        available.loc[after_close] = disclosed[after_close] + pd.Timedelta(days=1)

    df["available_date"] = available
    return df


def asof_latest(
    panel: pd.DataFrame,
    facts: pd.DataFrame,
    value_columns: list[str],
    *,
    panel_key: str = "Code",
    facts_key: str = "LocalCode",
    panel_date: str = "Date",
    facts_date: str = "available_date",
    suffix: str = "",
) -> pd.DataFrame:
    """パネルの各 (銘柄, 日付) に、その時点で入手済みの最新の facts を貼る.

    DuckDB の ASOF JOIN を使う。`facts_date <= panel_date` を満たす最新行だけが
    結合されるので、構造的に look-ahead が起きない。
    """
    if panel.empty:
        return panel
    if facts.empty:
        for col in value_columns:
            panel[f"{col}{suffix}"] = np.nan
        return panel

    import duckdb

    left = panel.copy()
    left[panel_date] = pd.to_datetime(left[panel_date])

    # value_columns に facts_date 自身が含まれることがあるので重複を潰す
    right_columns = list(dict.fromkeys([facts_key, facts_date, *value_columns]))
    right = facts[right_columns].copy()
    right[facts_date] = pd.to_datetime(right[facts_date])
    right = right.dropna(subset=[facts_date]).sort_values([facts_key, facts_date])

    selected = ", ".join(f'r."{c}" AS "{c}{suffix}"' for c in value_columns)
    sql = f"""
        SELECT l.*, {selected}
        FROM left_panel AS l
        ASOF LEFT JOIN right_facts AS r
          ON l."{panel_key}" = r."{facts_key}"
         AND l."{panel_date}" >= r."{facts_date}"
    """
    con = duckdb.connect()
    try:
        con.register("left_panel", left)
        con.register("right_facts", right)
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def assert_no_lookahead(
    df: pd.DataFrame, decision_date: str = "Date", source_date: str = "available_date"
) -> None:
    """結合結果に未来の情報が混ざっていないことを検証する（テストと取り込み後の点検用）."""
    if df.empty or source_date not in df.columns:
        return
    left = pd.to_datetime(df[decision_date])
    right = pd.to_datetime(df[source_date])
    violations = (right > left) & right.notna()
    if violations.any():
        sample = df.loc[violations, [decision_date, source_date]].head()
        raise AssertionError(f"未来の情報が {violations.sum()} 行混入しています:\n{sample}")
