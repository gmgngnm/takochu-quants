"""Claude による定性分析層.

四季報の【業績記事】に相当する定性評価を、決算短信の定性情報から生成する。
記者の取材部分は再現できないが、公開情報の要約・評価は自動化できるうえ、
年 4 回更新の四季報と違って**開示のたびに更新される**ぶん鮮度で上回る。

2 段構えでコストを抑える:
  1. 全開示を Claude Haiku 4.5 でスクリーニング
  2. 上位だけを Claude Opus 5 で精査

システムプロンプトは凍結されているので prompt caching が効く。
`usage.cache_read_input_tokens` が 0 のまま増えない場合はキャッシュが
壊れている（prompts.py の注意書きを参照）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from takochu.llm.prompts import (
    PROMPT_VERSION,
    SCREEN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_message,
)
from takochu.llm.schema import (
    ANALYSIS_SCHEMA,
    AnalysisValidationError,
    DisclosureAnalysis,
    parse_analysis,
)

log = logging.getLogger(__name__)

SCREEN_MODEL = "claude-haiku-4-5"
DETAIL_MODEL = "claude-opus-5"

# $/1M tokens (input, output)。コスト見積り用。
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_RATIO = 0.1
CACHE_WRITE_RATIO = 1.25


@dataclass(frozen=True)
class DisclosureInput:
    """分析対象の開示 1 件."""

    disclosure_number: str
    code: str
    period: str
    document_type: str
    text: str

    def cache_key(self, model: str) -> str:
        """同じ開示・同じプロンプト・同じモデルなら再実行しない."""
        return f"{self.disclosure_number}|{PROMPT_VERSION}|{model}"


@dataclass
class Usage:
    """課金対象トークンの累計."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    refusals: int = 0
    failures: int = 0

    def add(self, usage, model: str) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def estimated_cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
        return (
            self.input_tokens * rate_in
            + self.cache_read_tokens * rate_in * CACHE_READ_RATIO
            + self.cache_write_tokens * rate_in * CACHE_WRITE_RATIO
            + self.output_tokens * rate_out
        ) / 1_000_000


class ClaudeAnalyzer:
    """開示文書 -> 構造化された定性スコア.

    Args:
        client: anthropic.Anthropic 互換のクライアント。省略時に生成する。
            テストでは偽クライアントを差し込める。
        screen_model / detail_model: 2 段構えのモデル。
        use_fallbacks: Opus 5 の安全分類器が判断を拒否した場合に、
            サーバ側で別モデルに回す（`fallbacks="default"`）。
    """

    def __init__(
        self,
        client=None,
        screen_model: str = SCREEN_MODEL,
        detail_model: str = DETAIL_MODEL,
        use_fallbacks: bool = True,
        effort: str = "medium",
    ) -> None:
        self._client = client
        self.screen_model = screen_model
        self.detail_model = detail_model
        self.use_fallbacks = use_fallbacks
        self.effort = effort
        self.usage: dict[str, Usage] = {}

    @property
    def client(self):
        if self._client is None:
            import anthropic  # 遅延 import: LLM 層を使わないなら依存しない

            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------ 1 件

    def analyze(self, item: DisclosureInput, detailed: bool = False) -> DisclosureAnalysis | None:
        """開示 1 件を分析する。失敗時は None（呼び出し側で欠損として扱う）."""
        model = self.detail_model if detailed else self.screen_model
        usage = self.usage.setdefault(model, Usage())

        try:
            response = self._call(item, detailed=detailed)
        except Exception as exc:  # noqa: BLE001 - 1 件の失敗で全体を止めない
            if not self._is_api_error(exc):
                raise
            log.warning("%s の分析に失敗しました: %s", item.disclosure_number, exc)
            usage.failures += 1
            return None

        usage.add(response.usage, model)

        # 安全分類器が判断を拒否した場合、content は空になりうる。
        # content[0] を無条件に読むとここで落ちる。
        if response.stop_reason == "refusal":
            log.warning("%s は分類器により拒否されました", item.disclosure_number)
            usage.refusals += 1
            return None

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            log.warning("%s: テキストブロックがありません", item.disclosure_number)
            usage.failures += 1
            return None

        try:
            return parse_analysis(json.loads(text))
        except (json.JSONDecodeError, AnalysisValidationError) as exc:
            log.warning("%s: 出力の検証に失敗しました: %s", item.disclosure_number, exc)
            usage.failures += 1
            return None

    def _call(self, item: DisclosureInput, detailed: bool):
        user_message = build_user_message(
            item.code, item.period, item.document_type, item.text
        )
        # システムプロンプトは凍結されているのでここでキャッシュを切る。
        # 開示本文（毎回変わる）はブレークポイントより後ろに置く。
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT if detailed else SCREEN_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [{"role": "user", "content": user_message}]

        if not detailed:
            # Haiku 4.5 は effort を受け付けない。思考も使わない。
            return self.client.messages.create(
                model=self.screen_model,
                max_tokens=2048,
                system=system,
                messages=messages,
                output_config={"format": _json_schema_format()},
            )

        request = {
            "model": self.detail_model,
            "max_tokens": 8000,  # 適応思考のぶんの余裕を含む
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"format": _json_schema_format(), "effort": self.effort},
        }
        if self.use_fallbacks:
            request["betas"] = ["server-side-fallback-2026-07-01"]
            request["fallbacks"] = "default"
            return self.client.beta.messages.create(**request)
        return self.client.messages.create(**request)

    @staticmethod
    def _is_api_error(exc: Exception) -> bool:
        """anthropic の API 例外だけを握りつぶす（バグを隠さないため）."""
        try:
            import anthropic
        except ImportError:
            return False
        return isinstance(exc, (anthropic.APIError, anthropic.APIStatusError))

    # ---------------------------------------------------------------- バッチ

    def analyze_many(
        self,
        items: Iterable[DisclosureInput],
        detailed: bool = False,
        progress_every: int = 25,
    ) -> Iterator[tuple[DisclosureInput, DisclosureAnalysis | None]]:
        """複数の開示を順に分析する（1 件ごとに結果を yield）."""
        for i, item in enumerate(items, 1):
            yield item, self.analyze(item, detailed=detailed)
            if progress_every and i % progress_every == 0:
                log.info("%d 件を分析しました", i)

    def cost_report(self) -> dict:
        """モデル別のトークン使用量と概算コスト."""
        report = {}
        total = 0.0
        for model, usage in self.usage.items():
            cost = usage.estimated_cost_usd(model)
            total += cost
            report[model] = {
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "refusals": usage.refusals,
                "failures": usage.failures,
                "estimated_cost_usd": round(cost, 4),
            }
        report["total_estimated_cost_usd"] = round(total, 4)
        return report


def _json_schema_format() -> dict:
    return {"type": "json_schema", "schema": ANALYSIS_SCHEMA}
