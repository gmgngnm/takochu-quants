"""コマンドラインインターフェース."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from takochu.config import load_config
from takochu.io.jquants import JQuantsClient
from takochu.io.store import Store


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _context(args):
    config = load_config(args.config)
    store = Store(config.data_dir)
    return config, store


# --------------------------------------------------------------------- ingest


def cmd_ingest(args) -> int:
    from takochu.ingest import ingest_all

    config, store = _context(args)
    client = JQuantsClient(cache_dir=config.data_dir)
    summary = ingest_all(
        store,
        client,
        start=args.start,
        end=args.end,
        datasets=args.datasets,
        resume=not args.no_resume,
    )
    print(summary.to_string(index=False))
    return 0


# ------------------------------------------------------------------- features


def cmd_features(args) -> int:
    from takochu.features import build_feature_panel

    config, store = _context(args)
    quotes = store.read("daily_quotes")
    if quotes.empty:
        print("日足がありません。先に `takochu ingest` を実行してください。", file=sys.stderr)
        return 1

    listed = store.read("listed_info")
    statements = store.read("statements")
    calendar = store.read("trading_calendar")
    trading_days = pd.to_datetime(calendar["Date"]) if not calendar.empty else None

    announcements = store.read("announcement")
    panel = build_feature_panel(
        quotes, listed, statements, config, trading_days, announcements
    )
    path = store.write_derived("panel", panel)

    print(f"特徴量パネルを書き出しました: {path}")
    print(f"  行数        : {len(panel):,}")
    print(f"  期間        : {panel['Date'].min():%Y-%m-%d} ~ {panel['Date'].max():%Y-%m-%d}")
    print(f"  銘柄数      : {panel['Code'].nunique():,}")
    print(f"  ユニバース内: {int(panel['in_universe'].sum()):,} 行")
    print(f"  売買可能    : {int(panel['tradable'].sum()):,} 行")
    return 0


# ------------------------------------------------------------------- backtest


def cmd_backtest(args) -> int:
    from takochu.backtest import information_coefficient, run_backtest, summarize
    from takochu.backtest.metrics import ic_summary
    from takochu.features.build import weekly_decision_dates

    config, store = _context(args)
    panel = store.read_derived("panel")

    if args.start:
        config.backtest["start"] = args.start
    if args.end:
        config.backtest["end"] = args.end

    result = run_backtest(panel, config)
    stats = summarize(result.equity)

    print("\n=== バックテスト結果 ===")
    for key, value in stats.items():
        formatted = f"{value:,.4f}" if isinstance(value, float) else value
        print(f"  {key:<20}: {formatted}")

    print("\n=== 診断 ===")
    for key, value in result.diagnostics.items():
        formatted = f"{value:,.4f}" if isinstance(value, float) else value
        print(f"  {key:<20}: {formatted}")

    scored = panel
    if "score" not in scored.columns:
        from takochu.strategy.score import build_scores

        scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])
    ic = information_coefficient(scored, weekly_decision_dates(scored))
    print("\n=== Information Coefficient ===")
    summary = ic_summary(ic)
    if summary:
        for key, value in summary.items():
            formatted = f"{value:,.4f}" if isinstance(value, float) else value
            print(f"  {key:<20}: {formatted}")
        print("\n  |mean_ic| が 0.02 を下回るなら、スコアに予測力はほぼ無い。")
    else:
        print("  IC を計算できるだけの断面がありません。")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.equity.to_csv(out / "equity.csv", index=False)
        result.trades.to_csv(out / "trades.csv", index=False)
        ic.to_csv(out / "ic.csv", index=False)
        (out / "summary.json").write_text(
            json.dumps({"stats": stats, "diagnostics": result.diagnostics, "ic": summary},
                       ensure_ascii=False, indent=2, default=float)
        )
        print(f"\n結果を {out} に保存しました。")
    return 0


# --------------------------------------------------------------------- screen


def cmd_screen(args) -> int:
    """最新の判断日でのポートフォリオ候補を表示する（発注はしない）."""
    from takochu.strategy.score import build_scores, select_portfolio

    config, store = _context(args)
    panel = store.read_derived("panel")
    scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])

    as_of = pd.Timestamp(args.date) if args.date else scored["Date"].max()
    snapshot = scored[scored["Date"] == as_of]
    if snapshot.empty:
        print(f"{as_of:%Y-%m-%d} のデータがありません。", file=sys.stderr)
        return 1

    picks = select_portfolio(snapshot, config.portfolio)
    detail = picks.merge(
        snapshot[
            ["Code", "raw_close", "score_revision", "score_quality", "score_momentum",
             "score_value", "revision_op", "op_yoy", "progress_gap", "roe", "per",
             "days_to_next_earnings", "vol_20"]
        ],
        on="Code",
        how="left",
    )

    print(f"\n=== 判断日 {as_of:%Y-%m-%d} の候補 ({len(detail)} 銘柄) ===")
    print("※ これは提案であって発注ではありません。内容を必ず自分で確認してください。\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(detail.round(4).to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.out, index=False)
        print(f"\n{args.out} に保存しました。")
    return 0


# --------------------------------------------------------------------- doctor


def cmd_doctor(args) -> int:
    """データ健全性の点検。look-ahead とカバレッジの欠落を探す."""
    from takochu.pit import assert_no_lookahead

    config, store = _context(args)
    print("=== データセット ===")
    print(store.summary().to_string(index=False))

    try:
        panel = store.read_derived("panel")
    except FileNotFoundError:
        print("\nパネル未作成。`takochu features` を先に実行してください。")
        return 0

    print("\n=== パネル点検 ===")
    issues = 0

    try:
        assert_no_lookahead(panel)
        print("  [OK] ファンダ facts に未来の情報は混入していません")
    except AssertionError as exc:
        print(f"  [NG] {exc}")
        issues += 1

    universe = panel[panel["in_universe"]]
    for column in ["revision_op", "op_yoy", "roe", "per", "mom_12_1", "vol_20"]:
        if column not in universe.columns:
            continue
        coverage = float(universe[column].notna().mean())
        flag = "OK" if coverage > 0.5 else "警告"
        print(f"  [{flag}] {column:<16} 充足率 {coverage:6.1%}")
        if coverage <= 0.5:
            issues += 1

    per_date = universe.groupby("Date").size()
    if not per_date.empty:
        print(f"\n  ユニバース銘柄数: 中央値 {per_date.median():.0f} / "
              f"最小 {per_date.min():.0f} / 最大 {per_date.max():.0f}")

    print(f"\n問題 {issues} 件" if issues else "\n問題は見つかりませんでした。")
    return 0


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(
        prog="takochu", description="国内プライム株の週次スイング戦略基盤"
    )
    parser.add_argument("--config", help="設定 YAML のパス")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="J-Quants からデータを取り込む")
    p.add_argument("--start", help="開始日 YYYY-MM-DD")
    p.add_argument("--end", help="終了日 YYYY-MM-DD")
    p.add_argument("--datasets", nargs="*", help="対象データセット")
    p.add_argument("--no-resume", action="store_true", help="取り込み済みも再取得する")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("features", help="特徴量パネルを構築する")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("backtest", help="週次リバランスを検証する")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", help="結果の出力ディレクトリ")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("screen", help="最新断面の候補銘柄を表示する")
    p.add_argument("--date", help="判断日 YYYY-MM-DD（省略時は最新）")
    p.add_argument("--out", help="CSV 出力先")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("doctor", help="データ健全性を点検する")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
