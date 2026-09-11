"""Claude 定性層のテスト.

API キーなしで回るよう、偽クライアントを差し込む。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from takochu.llm.analyzer import ClaudeAnalyzer, DisclosureInput
from takochu.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from takochu.llm.schema import AnalysisValidationError, parse_analysis
from takochu.llm.store import AnalysisCache, build_llm_facts

VALID = {
    "earnings_quality": 1,
    "guidance_tone": 2,
    "catalysts": ["新工場の稼働"],
    "risk_flags": ["cost_pressure"],
    "confidence": 0.8,
    "rationale": "増収増益で利益率も改善。一過性要因の記載なし。",
}


# ------------------------------------------------------------------ 偽クライアント


@dataclass
class _FakeMessages:
    payload: object
    stop_reason: str = "end_turn"
    calls: list = None

    def create(self, **kwargs):
        if self.calls is not None:
            self.calls.append(kwargs)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=1200,
                output_tokens=150,
                cache_read_input_tokens=900,
                cache_creation_input_tokens=0,
            ),
        )


class _FakeClient:
    def __init__(self, payload=VALID, stop_reason="end_turn"):
        self.calls: list = []
        self.messages = _FakeMessages(payload, stop_reason, self.calls)
        self.beta = SimpleNamespace(messages=self.messages)


def _item(text: str = "当第2四半期は増収増益となりました。" * 20) -> DisclosureInput:
    return DisclosureInput(
        disclosure_number="12345",
        code="72030",
        period="2024-09-30",
        document_type="2Q",
        text=text,
    )


# ------------------------------------------------------------------------ schema


def test_正常な出力を検証して通す():
    analysis = parse_analysis(VALID)
    assert analysis.earnings_quality == 1
    assert analysis.risk_flags == ("cost_pressure",)


def test_範囲外のスコアを弾く():
    with pytest.raises(AnalysisValidationError, match="範囲外"):
        parse_analysis({**VALID, "earnings_quality": 5})


def test_未知のリスクフラグを弾く():
    with pytest.raises(AnalysisValidationError, match="未知"):
        parse_analysis({**VALID, "risk_flags": ["alien_invasion"]})


def test_必須項目の欠落を弾く():
    payload = {k: v for k, v in VALID.items() if k != "confidence"}
    with pytest.raises(AnalysisValidationError, match="必須項目"):
        parse_analysis(payload)


def test_確信度が低い分析はスコアが0に寄る():
    """情報量の乏しい開示が断面順位を動かさないこと."""
    confident = parse_analysis({**VALID, "confidence": 1.0})
    unsure = parse_analysis({**VALID, "confidence": 0.1})
    assert confident.score == pytest.approx(1.5)
    assert abs(unsure.score) < abs(confident.score)


# ----------------------------------------------------------------------- prompts


def test_システムプロンプトは呼び出しごとに変わらない():
    """prompt caching は前方一致。1バイトでも変わるとキャッシュが死ぬ."""
    assert SYSTEM_PROMPT == SYSTEM_PROMPT
    assert "{" not in SYSTEM_PROMPT.replace("{{", "").replace("}}", "")
    for token in ("今日", "現在時刻", "%Y"):
        assert token not in SYSTEM_PROMPT


def test_開示文書はデータとして囲まれ銘柄名を含まない():
    message = build_user_message("72030", "2024-09-30", "2Q", "本文テキスト")
    assert "<disclosure>" in message and "</disclosure>" in message
    assert "指示ではありません" in message
    assert "72030" in message


def test_長い文書は切り詰めた事実を明示する():
    message = build_user_message("72030", "2024-09-30", "2Q", "あ" * 30_000, max_chars=1000)
    assert "省略しています" in message
    assert len(message) < 2000


# ---------------------------------------------------------------------- analyzer


def test_偽クライアントで分析が通る():
    client = _FakeClient()
    analyzer = ClaudeAnalyzer(client=client)
    analysis = analyzer.analyze(_item())

    assert analysis is not None
    assert analysis.guidance_tone == 2
    assert analyzer.usage[analyzer.screen_model].calls == 1


def test_システムプロンプトにキャッシュ制御が付く():
    client = _FakeClient()
    ClaudeAnalyzer(client=client).analyze(_item())

    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_スクリーニングではeffortを送らない():
    """Haiku 4.5 は effort を受け付けない."""
    client = _FakeClient()
    ClaudeAnalyzer(client=client).analyze(_item(), detailed=False)
    assert "effort" not in client.calls[0]["output_config"]


def test_精査では思考とフォールバックを有効にする():
    client = _FakeClient()
    ClaudeAnalyzer(client=client).analyze(_item(), detailed=True)

    call = client.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "medium"
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]


def test_拒否された場合はNoneを返して落ちない():
    """stop_reason を見ずに content[0] を読むコードはここで壊れる."""
    client = _FakeClient(stop_reason="refusal")
    analyzer = ClaudeAnalyzer(client=client)

    assert analyzer.analyze(_item()) is None
    assert analyzer.usage[analyzer.screen_model].refusals == 1


def test_壊れた出力はNoneを返す():
    analyzer = ClaudeAnalyzer(client=_FakeClient(payload="これはJSONではない"))
    assert analyzer.analyze(_item()) is None
    assert analyzer.usage[analyzer.screen_model].failures == 1


def test_スキーマ違反の出力はNoneを返す():
    analyzer = ClaudeAnalyzer(client=_FakeClient(payload={**VALID, "confidence": 9}))
    assert analyzer.analyze(_item()) is None


def test_コスト集計が出る():
    analyzer = ClaudeAnalyzer(client=_FakeClient())
    for _ in range(3):
        analyzer.analyze(_item())
    report = analyzer.cost_report()
    assert report[analyzer.screen_model]["calls"] == 3
    assert report["total_estimated_cost_usd"] > 0


# ------------------------------------------------------------------------- store


def test_キャッシュは再読み込みしても残る(tmp_path):
    path = tmp_path / "analyses.jsonl"
    analysis = parse_analysis(VALID)

    cache = AnalysisCache(path)
    cache.put("k1", "12345", "72030", "claude-haiku-4-5", analysis)
    assert "k1" in cache

    reloaded = AnalysisCache(path)
    assert "k1" in reloaded
    assert reloaded.get("k1")["llm_score"] == pytest.approx(analysis.score)


def test_キャッシュキーはプロンプト版を含む():
    """プロンプトを変えたのに古い結果を使い回さないこと."""
    key = _item().cache_key("claude-opus-5")
    assert PROMPT_VERSION in key
    assert "claude-opus-5" in key


def test_同じ開示は精査結果を優先する(statements):
    rows = []
    number = statements["DisclosureNumber"].iloc[0]
    code = statements["LocalCode"].iloc[0]
    for model, tone in [("claude-haiku-4-5", 0), ("claude-opus-5", 2)]:
        rows.append({
            "disclosure_number": number, "code": code, "model": model,
            "llm_score": float(tone), "confidence": 1.0,
            "earnings_quality": tone, "guidance_tone": tone, "risk_flags": [],
        })
    facts = build_llm_facts(pd.DataFrame(rows), statements)
    assert len(facts) == 1
    assert facts["llm_tone"].iloc[0] == 2


def test_LLM_factsもPIT規約に従う(statements, trading_days):
    from takochu.pit import add_available_date

    rows = [
        {
            "disclosure_number": row.DisclosureNumber,
            "code": row.LocalCode,
            "model": "claude-haiku-4-5",
            "llm_score": 1.0,
            "confidence": 0.8,
            "earnings_quality": 1,
            "guidance_tone": 1,
            "risk_flags": [],
        }
        for row in statements.head(20).itertuples()
    ]
    facts = build_llm_facts(pd.DataFrame(rows), statements, pd.Series(trading_days))

    assert not facts.empty
    expected = add_available_date(statements, pd.Series(trading_days))
    lookup = dict(
        zip(expected["DisclosureNumber"].astype(str), expected["available_date"], strict=False)
    )
    # facts の available_date は開示由来であって、決算期末日ではない
    for row in facts.itertuples():
        assert row.available_date in lookup.values()


def test_パネルにLLMスコアが入り未来を参照しない(
    quotes, listed, statements, config, trading_days
):
    from takochu.features.build import build_feature_panel
    from takochu.pit import assert_no_lookahead

    rows = [
        {
            "disclosure_number": row.DisclosureNumber,
            "code": row.LocalCode,
            "model": "claude-haiku-4-5",
            "llm_score": 1.0,
            "confidence": 0.8,
            "earnings_quality": 1,
            "guidance_tone": 1,
            "risk_flags": [],
        }
        for row in statements.itertuples()
    ]
    facts = build_llm_facts(pd.DataFrame(rows), statements, pd.Series(trading_days))
    panel = build_feature_panel(
        quotes, listed, statements, config, pd.Series(trading_days), llm_facts=facts
    )

    assert "llm_score" in panel.columns
    assert panel["llm_score"].notna().any()
    assert_no_lookahead(panel)
