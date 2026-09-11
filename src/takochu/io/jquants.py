"""J-Quants API クライアント.

認証は 2 段階:
  1. メールアドレス + パスワード -> refreshToken (有効期間 1 週間)
  2. refreshToken -> idToken (有効期間 24 時間)
idToken を Authorization: Bearer ヘッダに載せてデータ API を叩く。

refreshToken は ~/.cache 相当（data_dir/.auth）にキャッシュし、毎回
パスワード認証を走らせないようにしている。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.jquants.com/v1"

# 市場区分コード。プライムは 0111。
MARKET_PRIME = "0111"
MARKET_STANDARD = "0112"
MARKET_GROWTH = "0113"


class JQuantsError(RuntimeError):
    pass


@dataclass
class _Token:
    value: str
    expires_at: datetime

    @property
    def valid(self) -> bool:
        # 期限ぎりぎりでの失効を避けるため 5 分のマージンを取る
        return datetime.now() < self.expires_at - timedelta(minutes=5)


class JQuantsClient:
    """J-Quants API の薄いラッパ.

    - ページネーション (pagination_key) を透過的に処理する
    - 429 / 5xx に対して指数バックオフでリトライする
    - 最低リクエスト間隔を空けてレート制限に当てないようにする
    """

    def __init__(
        self,
        mailaddress: str | None = None,
        password: str | None = None,
        refresh_token: str | None = None,
        cache_dir: Path | None = None,
        min_interval_sec: float = 0.12,
        max_retries: int = 5,
    ) -> None:
        self._mail = mailaddress or os.environ.get("JQUANTS_MAILADDRESS")
        self._password = password or os.environ.get("JQUANTS_PASSWORD")
        self._refresh_token_override = refresh_token or os.environ.get("JQUANTS_REFRESH_TOKEN")
        self._cache_dir = cache_dir
        self._min_interval = min_interval_sec
        self._max_retries = max_retries

        self._session = requests.Session()
        self._id_token: _Token | None = None
        self._refresh: _Token | None = None
        self._last_request_at = 0.0

    # ------------------------------------------------------------------ 認証

    def _auth_cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / ".auth" / "refresh_token.json"

    def _load_cached_refresh(self) -> _Token | None:
        path = self._auth_cache_path()
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            token = _Token(payload["token"], datetime.fromisoformat(payload["expires_at"]))
            return token if token.valid else None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save_cached_refresh(self, token: _Token) -> None:
        path = self._auth_cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token": token.value, "expires_at": token.expires_at.isoformat()})
        )
        path.chmod(0o600)

    def _get_refresh_token(self) -> str:
        if self._refresh_token_override:
            return self._refresh_token_override
        if self._refresh and self._refresh.valid:
            return self._refresh.value

        cached = self._load_cached_refresh()
        if cached:
            self._refresh = cached
            return cached.value

        if not (self._mail and self._password):
            raise JQuantsError(
                "JQUANTS_MAILADDRESS / JQUANTS_PASSWORD が未設定です。"
                ".env を作成するか JQUANTS_REFRESH_TOKEN を設定してください。"
            )

        resp = self._session.post(
            f"{API_BASE}/token/auth_user",
            data=json.dumps({"mailaddress": self._mail, "password": self._password}),
            timeout=30,
        )
        if resp.status_code != 200:
            raise JQuantsError(f"auth_user に失敗しました: {resp.status_code} {resp.text[:200]}")

        token = _Token(resp.json()["refreshToken"], datetime.now() + timedelta(days=6))
        self._refresh = token
        self._save_cached_refresh(token)
        return token.value

    def _get_id_token(self) -> str:
        if self._id_token and self._id_token.valid:
            return self._id_token.value

        refresh = self._get_refresh_token()
        resp = self._session.post(
            f"{API_BASE}/token/auth_refresh",
            params={"refreshtoken": refresh},
            timeout=30,
        )
        if resp.status_code != 200:
            # キャッシュした refreshToken が失効している場合は捨ててやり直す
            path = self._auth_cache_path()
            if path and path.exists() and not self._refresh_token_override:
                path.unlink()
                self._refresh = None
                return self._get_id_token()
            raise JQuantsError(f"auth_refresh に失敗しました: {resp.status_code} {resp.text[:200]}")

        self._id_token = _Token(resp.json()["idToken"], datetime.now() + timedelta(hours=23))
        return self._id_token.value

    # -------------------------------------------------------------- HTTP 基盤

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        for attempt in range(self._max_retries):
            self._throttle()
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            resp = self._session.get(url, params=params, headers=headers, timeout=60)

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                # idToken 失効。作り直して即リトライする。
                self._id_token = None
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2.0**attempt
                log.warning("%s -> %s, %.0fs 待って再試行します", path, resp.status_code, wait)
                time.sleep(wait)
                continue
            raise JQuantsError(f"{path} -> {resp.status_code}: {resp.text[:300]}")

        raise JQuantsError(f"{path}: {self._max_retries} 回リトライしても成功しませんでした")

    def _paged(self, path: str, key: str, **params: Any) -> Iterator[dict[str, Any]]:
        """pagination_key を辿って全件返す."""
        params = {k: v for k, v in params.items() if v is not None}
        while True:
            payload = self._request(path, params)
            yield from payload.get(key, [])
            next_key = payload.get("pagination_key")
            if not next_key:
                return
            params["pagination_key"] = next_key

    def _frame(self, path: str, key: str, **params: Any) -> pd.DataFrame:
        rows = list(self._paged(path, key, **params))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ データ API

    def listed_info(self, on: date | str | None = None) -> pd.DataFrame:
        """銘柄一覧。市場区分・業種コードを含む."""
        return self._frame("/listed/info", "info", date=_ymd(on))

    def daily_quotes(
        self,
        code: str | None = None,
        on: date | str | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """日足四本値。code 指定で期間取得、date 指定で全銘柄の 1 日分."""
        return self._frame(
            "/prices/daily_quotes",
            "daily_quotes",
            code=code,
            date=_ymd(on),
            **{"from": _ymd(start), "to": _ymd(end)},
        )

    def statements(
        self,
        code: str | None = None,
        on: date | str | None = None,
    ) -> pd.DataFrame:
        """決算短信（XBRL 由来）。開示日ベースで取得できるので PIT 構築に使える."""
        return self._frame("/fins/statements", "statements", code=code, date=_ymd(on))

    def announcement(self) -> pd.DataFrame:
        """翌営業日の決算発表予定。決算またぎ回避に使う."""
        return self._frame("/fins/announcement", "announcement")

    def weekly_margin_interest(
        self,
        code: str | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """週次信用取引残高."""
        return self._frame(
            "/markets/weekly_margin_interest",
            "weekly_margin_interest",
            code=code,
            **{"from": _ymd(start), "to": _ymd(end)},
        )

    def trading_calendar(
        self, start: date | str | None = None, end: date | str | None = None
    ) -> pd.DataFrame:
        """営業日カレンダー."""
        return self._frame(
            "/markets/trading_calendar",
            "trading_calendar",
            **{"from": _ymd(start), "to": _ymd(end)},
        )

    def topix(
        self, start: date | str | None = None, end: date | str | None = None
    ) -> pd.DataFrame:
        """TOPIX 指数。レジームフィルタとベンチマークに使う."""
        return self._frame("/indices/topix", "topix", **{"from": _ymd(start), "to": _ymd(end)})


def _ymd(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d")
