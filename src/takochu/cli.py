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
    try:
        llm_facts = store.read_derived("llm_facts")
    except FileNotFoundError:
        llm_facts = None
    panel = build_feature_panel(
        quotes, listed, statements, config, trading_days, announcements, llm_facts
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



# -------------------------------------------------------------------- analyze


def cmd_analyze(args) -> int:
    """決算開示の定性情報を Claude で構造化スコアに変換する."""
    from takochu.llm.analyzer import ClaudeAnalyzer
    from takochu.llm.sources import iter_disclosure_inputs
    from takochu.llm.store import AnalysisCache, build_llm_facts, select_for_detail

    config, store = _context(args)
    statements = store.read("statements")
    if statements.empty:
        print("決算短信がありません。先に `takochu ingest` を実行してください。", file=sys.stderr)
        return 1

    text_dir = Path(args.text_dir) if args.text_dir else config.data_dir / "disclosures"
    cache = AnalysisCache(config.data_dir / "derived" / "llm_analyses.jsonl")

    detail_codes: set[str] = set()
    if args.detail_top:
        try:
            from takochu.strategy.score import build_scores

            panel = store.read_derived("panel")
            scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])
            detail_codes = select_for_detail(scored, args.detail_top)
            print(f"精査対象（上位 {args.detail_top} 銘柄）: {len(detail_codes)} 件")
        except FileNotFoundError:
            print("パネル未作成のため、精査対象を選べません。全件スクリーニングのみ行います。")

    items = list(
        iter_disclosure_inputs(statements, text_dir, codes=args.codes, since=args.since)
    )
    print(f"テキストが揃っている開示: {len(items)} 件")

    analyzer = ClaudeAnalyzer(use_fallbacks=not args.no_fallbacks, effort=args.effort)
    pending = []
    for item in items:
        detailed = item.code in detail_codes
        model = analyzer.detail_model if detailed else analyzer.screen_model
        if item.cache_key(model) in cache:
            continue
        pending.append((item, detailed))

    print(f"未分析（今回 API を呼ぶ件数）: {len(pending)} 件")
    if args.dry_run:
        print("\n--dry-run のため API は呼びません。")
        return 0
    if not pending:
        print("すべてキャッシュ済みです。")

    for i, (item, detailed) in enumerate(pending, 1):
        if args.limit and i > args.limit:
            print(f"--limit {args.limit} に達したので打ち切ります。")
            break
        model = analyzer.detail_model if detailed else analyzer.screen_model
        analysis = analyzer.analyze(item, detailed=detailed)
        if analysis is not None:
            cache.put(item.cache_key(model), item.disclosure_number, item.code, model, analysis)
        if i % 25 == 0:
            print(f"  {i}/{len(pending)} 件完了")

    calendar = store.read("trading_calendar")
    trading_days = pd.to_datetime(calendar["Date"]) if not calendar.empty else None
    facts = build_llm_facts(cache.to_frame(), statements, trading_days)
    if not facts.empty:
        path = store.write_derived("llm_facts", facts)
        print(f"\nLLM facts を書き出しました: {path} ({len(facts):,} 件)")
        print("`takochu features` を再実行するとパネルに反映されます。")

    print("\n=== API 使用量 ===")
    print(json.dumps(analyzer.cost_report(), ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------- report


def cmd_report(args) -> int:
    """週次レポート（HTML）を生成する."""
    from takochu.report import build_report, write_report
    from takochu.strategy.score import build_scores, select_portfolio

    config, store = _context(args)
    panel = store.read_derived("panel")
    scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])

    all_dates = pd.DatetimeIndex(sorted(scored["Date"].unique()))
    as_of = pd.Timestamp(args.date) if args.date else all_dates[-1]
    snapshot = scored[scored["Date"] == as_of]
    if snapshot.empty:
        print(f"{as_of:%Y-%m-%d} のデータがありません。", file=sys.stderr)
        return 1

    idx = all_dates.searchsorted(as_of, side="right")
    exec_date = all_dates[idx] if idx < len(all_dates) else None

    picks = select_portfolio(snapshot, config.portfolio)

    history_path = config.data_dir / "derived" / "picks_history.parquet"
    previous = _previous_picks(history_path, as_of)

    analyses = _analysis_details(config.data_dir)
    if analyses is not None:
        snapshot = snapshot.merge(analyses, on="Code", how="left")

    capital = args.capital or config.backtest.get("initial_capital")
    content = build_report(
        picks=picks,
        snapshot=snapshot,
        decision_date=as_of,
        exec_date=exec_date,
        previous_picks=previous,
        capital=capital,
        diagnostics={"パネル最終日": f"{panel['Date'].max():%Y-%m-%d}"},
    )

    out = Path(args.out) if args.out else config.data_dir / "reports" / f"{as_of:%Y%m%d}.html"
    write_report(out, content)
    print(f"レポートを書き出しました: {out}")
    n_exit = 0 if previous is None else len(set(previous["Code"]) - set(picks["Code"]))
    print(f"  判断日 {as_of:%Y-%m-%d} / 買い {len(picks)} 銘柄 / 売り {n_exit} 銘柄")

    if not args.no_save:
        _save_picks(history_path, picks, as_of)
        print(f"  保有履歴を更新しました: {history_path}")
    return 0


def _previous_picks(path: Path, as_of: pd.Timestamp) -> pd.DataFrame | None:
    if not path.exists():
        return None
    history = pd.read_parquet(path)
    history = history[pd.to_datetime(history["decision_date"]) < as_of]
    if history.empty:
        return None
    latest = history["decision_date"].max()
    return history[history["decision_date"] == latest]


def _save_picks(path: Path, picks: pd.DataFrame, as_of: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = picks.assign(decision_date=as_of)
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[pd.to_datetime(existing["decision_date"]) != as_of]
        record = pd.concat([existing, record], ignore_index=True)
    record.to_parquet(path, index=False)


def _analysis_details(data_dir: Path) -> pd.DataFrame | None:
    """レポートに載せる根拠テキストを、分析キャッシュから拾う."""
    path = data_dir / "derived" / "llm_analyses.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["Code"] = df["code"].astype(str)
    if "llm_confidence" not in df.columns:
        df = df.rename(columns={"confidence": "llm_confidence"})
    # 同じ銘柄に複数の開示があれば、最後に分析したものを載せる
    df = df.drop_duplicates("Code", keep="last")
    return df[["Code", "rationale", "risk_flags", "llm_confidence"]]



# ------------------------------------------------------------------------ pwa


def cmd_pwa(args) -> int:
    """週次レポートを iPhone に入れられる PWA として書き出す."""
    from takochu.pwa import build_pwa
    from takochu.strategy.score import build_scores, select_portfolio

    config, store = _context(args)
    panel = store.read_derived("panel")
    scored = build_scores(panel, config.score_weights, config.portfolio["sector_field"])

    all_dates = pd.DatetimeIndex(sorted(scored["Date"].unique()))
    as_of = pd.Timestamp(args.date) if args.date else all_dates[-1]
    snapshot = scored[scored["Date"] == as_of]
    if snapshot.empty:
        print(f"{as_of:%Y-%m-%d} のデータがありません。", file=sys.stderr)
        return 1

    idx = all_dates.searchsorted(as_of, side="right")
    exec_date = all_dates[idx] if idx < len(all_dates) else None

    picks = select_portfolio(snapshot, config.portfolio)
    analyses = _analysis_details(config.data_dir)
    if analyses is not None:
        snapshot = snapshot.merge(analyses, on="Code", how="left")

    previous = _previous_picks(config.data_dir / "derived" / "picks_history.parquet", as_of)

    alerts = None
    alerts_path = config.data_dir / "derived" / "alerts.json"
    if alerts_path.exists():
        alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
        # 古い週のアラートを混ぜない
        if alerts.get("decisionDate") != f"{as_of:%Y-%m-%d}":
            alerts = None

    out_dir = Path(args.out) if args.out else config.data_dir / "pwa"
    index = build_pwa(
        out_dir=out_dir,
        picks=picks,
        snapshot=snapshot,
        decision_date=as_of,
        exec_date=exec_date,
        previous_picks=previous,
        capital=args.capital or config.backtest.get("initial_capital"),
        alerts=alerts,
    )

    print(f"PWA を書き出しました: {index.parent}")
    print(f"  判断日 {as_of:%Y-%m-%d} / 買い {len(picks)} 銘柄")
    print("\n手元で確認する:")
    print(f"  python -m http.server 8000 --directory {index.parent}")
    print("\niPhone のホーム画面に入れるには HTTPS が必要です。")
    print("  GitHub Pages などに置き、Safari で開いて 共有 → ホーム画面に追加")
    print("  ※ 保有銘柄が載るので、公開リポジトリには置かないこと。")
    return 0



# ------------------------------------------------------------------- monitor


def cmd_monitor(args) -> int:
    """保有中の銘柄が利確・損切りの水準に達していないか点検する."""
    from takochu.monitor import alerts_to_records, check_holdings

    config, store = _context(args)
    panel = store.read_derived("panel")

    history_path = config.data_dir / "derived" / "picks_history.parquet"
    if not history_path.exists():
        print("保有履歴がありません。先に `takochu report` を実行してください。", file=sys.stderr)
        return 1

    history = pd.read_parquet(history_path)
    history["decision_date"] = pd.to_datetime(history["decision_date"])
    decision_date = history["decision_date"].max()
    holdings = history[history["decision_date"] == decision_date]

    holding_cfg = dict(config.holding)
    for key in ("take_profit", "stop_loss", "stop_loss_atr"):
        value = getattr(args, key)
        if value is not None:
            holding_cfg[key] = value

    if all(holding_cfg.get(k) is None for k in ("take_profit", "stop_loss", "stop_loss_atr")):
        print("利確・損切りの水準が設定されていません。", file=sys.stderr)
        print("config/default.yaml の holding か、--stop-loss / --take-profit で指定してください。",
              file=sys.stderr)
        return 1

    alerts, status = check_holdings(panel, holdings, decision_date, holding_cfg)

    print(f"=== 保有 {len(holdings)} 銘柄 / {decision_date:%Y-%m-%d} 判断 "
          f"/ 最終データ {panel['Date'].max():%Y-%m-%d} ===\n")
    if status.empty:
        print("点検できる保有がありません。")
        return 0

    display = status.assign(
        change=lambda d: (d["change"] * 100).round(2).astype(str) + "%",
        reason=lambda d: d["reason"].map({"stop_loss": "損切り", "take_profit": "利確"}).fillna(""),
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(display.round(1).to_string(index=False))

    if alerts:
        print(f"\n!! 手仕舞い推奨 {len(alerts)} 件")
        for a in alerts:
            label = "損切り" if a.reason == "stop_loss" else "利確"
            print(f"  {a.code}  {label}  {a.touched_on} に {a.trigger_price:,.0f} 到達"
                  f"  (エントリー {a.entry_price:,.0f} / 直近 {a.last_close:,.0f}"
                  f" / {a.change:+.1%})")
        print("\n  日足ベースの判定です。実際に手仕舞えるのは翌営業日以降。")
        print("  ザラ場で即座に反応したい場合は、証券会社の API と逆指値注文が必要です。")
    else:
        print("\n水準に達した銘柄はありません。")

    out = config.data_dir / "derived" / "alerts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"decisionDate": f"{decision_date:%Y-%m-%d}",
             "asOf": f"{panel['Date'].max():%Y-%m-%d}",
             "alerts": alerts_to_records(alerts)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{out} に保存しました（PWA に表示されます）。")
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

    p = sub.add_parser("analyze", help="決算開示を Claude で定性分析する")
    p.add_argument("--text-dir", help="開示テキストの置き場（既定: data/disclosures）")
    p.add_argument("--detail-top", type=int, default=100,
                   help="上位何銘柄を上位モデルで精査するか（0 で精査なし）")
    p.add_argument("--since", help="この日付以降の開示のみ YYYY-MM-DD")
    p.add_argument("--codes", nargs="*", help="対象銘柄コードを限定する")
    p.add_argument("--limit", type=int, help="API 呼び出しの上限件数")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--no-fallbacks", action="store_true",
                   help="拒否時のサーバ側フォールバックを使わない")
    p.add_argument("--dry-run", action="store_true", help="件数だけ数えて API を呼ばない")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="週次レポート(HTML)を生成する")
    p.add_argument("--date", help="判断日 YYYY-MM-DD（省略時は最新）")
    p.add_argument("--out", help="出力先 HTML")
    p.add_argument("--capital", type=float, help="投下資金。株数の算出に使う")
    p.add_argument("--no-save", action="store_true", help="保有履歴を更新しない")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("monitor", help="保有中の利確・損切り水準を点検する")
    p.add_argument("--take-profit", type=float, help="利確ライン 例: 0.08")
    p.add_argument("--stop-loss", type=float, help="損切りライン 例: 0.05")
    p.add_argument("--stop-loss-atr", type=float, help="ATR 倍率での損切り 例: 2.0")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("pwa", help="iPhone 用の PWA を書き出す")
    p.add_argument("--date", help="判断日 YYYY-MM-DD（省略時は最新）")
    p.add_argument("--out", help="出力先ディレクトリ")
    p.add_argument("--capital", type=float, help="投下資金。株数の算出に使う")
    p.set_defaults(func=cmd_pwa)

    p = sub.add_parser("doctor", help="データ健全性を点検する")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
