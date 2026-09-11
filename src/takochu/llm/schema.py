"""Claude に返させる構造化出力の定義.

自由文で「この銘柄は買いです」と言わせない。抽出タスクとして
限定した構造を強制し、スコアに変換できる形だけを受け取る。

pydantic には依存しない（LLM 層を入れなくても本体が動くように）。
JSON Schema を API に渡し、返ってきた dict を自前で検証する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# 決算開示から抽出させるリスク要因の閉じた集合。
# 自由記述を許すと表記揺れで集計できなくなるので enum で固定する。
RISK_FLAGS = [
    "one_time_gain",           # 一過性利益（資産売却益など）による嵩上げ
    "one_time_loss",           # 一過性損失。裏を返せば来期は改善しうる
    "accounting_change",       # 会計基準・見積りの変更
    "guidance_cut_risk",       # 下方修正リスクを示唆する記述
    "demand_weakness",         # 需要減速・受注減
    "cost_pressure",           # 原価・人件費の圧迫
    "fx_dependency",           # 為替前提への依存が大きい
    "customer_concentration",  # 特定顧客への依存
    "dilution",                # 増資・転換社債による希薄化
    "litigation",              # 訴訟・行政処分
]

ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "earnings_quality": {
            "type": "integer",
            "minimum": -2,
            "maximum": 2,
            "description": (
                "利益の質。本業の実力で稼いだ利益か、一過性要因や会計処理による"
                "嵩上げか。+2=極めて質が高い, 0=判断材料なし, -2=実態は大幅に悪い。"
            ),
        },
        "guidance_tone": {
            "type": "integer",
            "minimum": -2,
            "maximum": 2,
            "description": (
                "会社が示す見通しのトーン。数値そのものではなく、"
                "表現の強気・弱気の度合い。+2=明確に自信を強めた, "
                "0=前回から変化なし, -2=明確に慎重化した。"
            ),
        },
        "catalysts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 60},
            "maxItems": 3,
            "description": "今後1〜3ヶ月で業績に効きうる具体的な材料。無ければ空配列。",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string", "enum": RISK_FLAGS},
            "maxItems": 5,
            "description": "文書中に根拠がある懸念のみ。推測で付けない。",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "判断の確からしさ。記述が定型文ばかりで情報量が乏しい場合は"
                "低くすること。低い確信度は減点ではなく、正直な申告として扱う。"
            ),
        },
        "rationale": {
            "type": "string",
            "maxLength": 300,
            "description": "スコアの根拠。文書中の具体的な記述を挙げる。",
        },
    },
    "required": [
        "earnings_quality",
        "guidance_tone",
        "catalysts",
        "risk_flags",
        "confidence",
        "rationale",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class DisclosureAnalysis:
    """1 開示に対する定性分析の結果."""

    earnings_quality: int
    guidance_tone: int
    catalysts: tuple[str, ...]
    risk_flags: tuple[str, ...]
    confidence: float
    rationale: str

    @property
    def score(self) -> float:
        """スコア合成に渡す単一の値.

        利益の質と見通しトーンの平均を、確信度で減衰させる。
        確信度が低い分析を 0 に寄せることで、情報量の乏しい開示が
        断面順位を動かさないようにする。
        """
        base = (self.earnings_quality + self.guidance_tone) / 2.0
        return base * self.confidence

    def to_dict(self) -> dict:
        data = asdict(self)
        data["catalysts"] = list(self.catalysts)
        data["risk_flags"] = list(self.risk_flags)
        return data


class AnalysisValidationError(ValueError):
    pass


def parse_analysis(payload: dict) -> DisclosureAnalysis:
    """API の戻り値を検証して dataclass に変換する.

    structured outputs はスキーマ適合を保証するが、これは外部システムとの
    境界なので明示的に検証する（スキーマを緩めたときに静かに壊れないように）。
    """
    if not isinstance(payload, dict):
        raise AnalysisValidationError(f"dict ではありません: {type(payload)}")

    missing = [k for k in ANALYSIS_SCHEMA["required"] if k not in payload]
    if missing:
        raise AnalysisValidationError(f"必須項目が欠けています: {missing}")

    quality = _require_int_range(payload["earnings_quality"], -2, 2, "earnings_quality")
    tone = _require_int_range(payload["guidance_tone"], -2, 2, "guidance_tone")

    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise AnalysisValidationError(f"confidence が範囲外です: {confidence!r}")

    unknown = [f for f in payload["risk_flags"] if f not in RISK_FLAGS]
    if unknown:
        raise AnalysisValidationError(f"未知の risk_flag: {unknown}")

    return DisclosureAnalysis(
        earnings_quality=quality,
        guidance_tone=tone,
        catalysts=tuple(str(c) for c in payload["catalysts"])[:3],
        risk_flags=tuple(payload["risk_flags"])[:5],
        confidence=float(confidence),
        rationale=str(payload["rationale"])[:300],
    )


def _require_int_range(value, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisValidationError(f"{name} が整数ではありません: {value!r}")
    if not low <= value <= high:
        raise AnalysisValidationError(f"{name} が範囲外です: {value}")
    return value
