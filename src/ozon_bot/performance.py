"""Read-only Ozon Performance API client; credentials stay in process memory.

Contracts: https://docs.ozon.ru/api/performance/ (Performance API 2.0).
An asynchronous report accepts at most ten campaigns and 62 calendar days.
Money in statistics reports is already expressed in rubles, unlike campaign budgets.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
import time
import zipfile
import zlib
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any, Self

import httpx

BASE_URL = "https://api-performance.ozon.ru"
MAX_REPORT_BYTES = 20 * 1024 * 1024
MAX_ROWS = 300_000


class PerformanceAPIError(RuntimeError):
    """Safe, user-facing failure that contains neither response bodies nor credentials."""


def _number(value: Any, field: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or value is None:
        raise PerformanceAPIError(f"В рекламном отчёте некорректное значение поля {field}.")
    normalized = str(value).strip().replace("\u00a0", "").replace("\u202f", "")
    normalized = normalized.replace(" ", "").replace(",", ".")
    normalized = re.sub(r"(?:₽|руб\.?|Р)$", "", normalized, flags=re.IGNORECASE)
    try:
        number = Decimal(normalized)
    except (InvalidOperation, ValueError):
        raise PerformanceAPIError(
            f"В рекламном отчёте некорректное значение поля {field}."
        ) from None
    if not number.is_finite() or (integer and (number < 0 or number != number.to_integral_value())):
        raise PerformanceAPIError(f"В рекламном отчёте некорректное значение поля {field}.")
    if abs(number) > Decimal("1e18"):
        raise PerformanceAPIError(f"В рекламном отчёте слишком большое значение поля {field}.")
    return int(number) if integer else float(number)


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\ufeff", "").strip().lower())


_ALIASES = {
    "date": ("date", "дата", "день"),
    "sku": ("sku", "артикул ozon"),
    "spend": (
        "moneyspent",
        "expense",
        "spent",
        "расход",
        "расход, ₽",
        "расход, р",
        "расход, ₽, с ндс",
        "расход, р, с ндс",
        "расход, руб.",
        "расход, руб., с ндс",
        "расход, ₽ с ндс",
        "расход, р с ндс",
        "расход, руб. с ндс",
    ),
    "revenue": (
        "revenue",
        "sales",
        "ordersmoney",
        "выручка",
        "выручка, ₽",
        "выручка, р",
        "выручка, руб.",
        "заказы, ₽",
        "заказы, р",
        "заказы, руб.",
    ),
    "orders": ("orders", "заказы", "заказы, шт.", "заказы, шт"),
    "clicks": ("clicks", "клики"),
    "impressions": ("views", "impressions", "показы"),
}


def _normalize_row(
    row: dict[str, Any], campaign_id: str, campaign_name: str, start: date, end: date
) -> dict[str, Any] | None:
    keyed = {_key(str(k)): v for k, v in row.items()}
    total_keys = set(_ALIASES["date"]) | set(_ALIASES["sku"])
    if any(
        key in total_keys and _key(str(value)) in {"всего", "итого", "total", "bcero"}
        for key, value in keyed.items()
    ):
        return None
    selected: dict[str, Any] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in keyed:
                selected[field] = keyed[alias]
                break
        if field not in selected and field != "sku":
            raise PerformanceAPIError(
                f"Рекламный отчёт не содержит поле {field}. "
                "Выберите отчёт с группировкой по дням и полными показателями; "
                "этот тип кампании может требовать отдельного отчёта Ozon."
            )
    raw_date = str(selected["date"]).strip()
    parsed: date | None = None
    for pattern in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            parsed = datetime.strptime(raw_date, pattern).date()
            break
        except ValueError:
            pass
    if parsed is None:
        raise PerformanceAPIError("В рекламном отчёте отсутствует корректная дата строки.")
    if not start <= parsed <= end:
        raise PerformanceAPIError(
            "Рекламный отчёт содержит дату за пределами запрошенного периода."
        )
    normalized: dict[str, Any] = {
        "date": parsed.isoformat(),
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "sku": str(selected.get("sku") or "").strip(),
    }
    for field in ("spend", "revenue", "orders", "clicks", "impressions"):
        normalized[field] = _number(
            selected[field], field, integer=field in {"orders", "clicks", "impressions"}
        )
    return normalized


def _inflate_gzip(data: bytes) -> bytes:
    try:
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        result = inflater.decompress(data, MAX_REPORT_BYTES + 1)
        if len(result) > MAX_REPORT_BYTES or inflater.unconsumed_tail:
            raise PerformanceAPIError("Рекламный отчёт слишком большой; выберите меньший период.")
        result += inflater.flush(MAX_REPORT_BYTES + 1 - len(result))
        if len(result) > MAX_REPORT_BYTES or not inflater.eof or inflater.unused_data:
            raise PerformanceAPIError("Недопустимый или слишком большой GZIP рекламного отчёта.")
        return result
    except zlib.error:
        raise PerformanceAPIError("Не удалось прочитать GZIP рекламного отчёта.") from None


def _csv_rows(data: bytes) -> list[dict[str, str]]:
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise PerformanceAPIError("Рекламный CSV должен иметь кодировку UTF-8.") from None
    # Ozon adds a campaign/period preamble before the actual semicolon-separated header.
    lines = decoded.splitlines()
    for index, line in enumerate(lines):
        header = [_key(v) for v in next(csv.reader([line], delimiter=";"), [])]
        if any(alias in header for alias in _ALIASES["spend"]):
            try:
                rows = list(csv.DictReader(io.StringIO("\n".join(lines[index:])), delimiter=";"))
            except csv.Error:
                raise PerformanceAPIError("Некорректный CSV рекламного отчёта.") from None
            if len(rows) > MAX_ROWS:
                raise PerformanceAPIError(
                    "В рекламном отчёте слишком много строк; сократите период."
                )
            return rows
    raise PerformanceAPIError("В рекламном CSV не найдены заголовки показателей.")


def _parse_report(
    data: bytes, campaigns: dict[str, str], start: date, end: date
) -> list[dict[str, Any]]:
    if len(data) > MAX_REPORT_BYTES:
        raise PerformanceAPIError("Рекламный отчёт слишком большой; выберите меньший период.")
    if data.startswith(b"\x1f\x8b"):
        data = _inflate_gzip(data)
    result: list[dict[str, Any]] = []

    def add_rows(rows: Any, campaign_id: str) -> None:
        if campaign_id not in campaigns or not isinstance(rows, list):
            raise PerformanceAPIError("Ozon вернул неизвестную структуру рекламного отчёта.")
        if len(rows) + len(result) > MAX_ROWS:
            raise PerformanceAPIError("В рекламном отчёте слишком много строк; сократите период.")
        for row in rows:
            if not isinstance(row, dict):
                raise PerformanceAPIError("Ozon вернул неизвестную структуру рекламной строки.")
            normalized = _normalize_row(row, campaign_id, campaigns[campaign_id], start, end)
            if normalized is not None:
                result.append(normalized)

    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = [v for v in archive.infolist() if not v.is_dir()]
                if not entries or len(entries) > 10:
                    raise PerformanceAPIError("Недопустимое число файлов в рекламном ZIP.")
                if sum(v.file_size for v in entries) > MAX_REPORT_BYTES:
                    raise PerformanceAPIError("Рекламный ZIP слишком большой; сократите период.")
                seen: set[str] = set()
                for entry in entries:
                    path = PurePosixPath(entry.filename)
                    campaign_id = path.stem
                    if path.suffix.lower() != ".csv" or campaign_id in seen:
                        raise PerformanceAPIError(
                            "Ozon вернул неизвестную структуру рекламного ZIP."
                        )
                    seen.add(campaign_id)
                    add_rows(_csv_rows(archive.read(entry)), campaign_id)
                if seen != set(campaigns):
                    raise PerformanceAPIError("В рекламном ZIP отсутствуют запрошенные кампании.")
        except (zipfile.BadZipFile, RuntimeError, OSError):
            raise PerformanceAPIError("Не удалось прочитать ZIP рекламного отчёта.") from None
        return result
    if data.lstrip().startswith((b"{", b"[")):
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise PerformanceAPIError("Ozon вернул некорректный JSON рекламного отчёта.") from None
        if not isinstance(payload, dict):
            raise PerformanceAPIError("Ozon вернул неизвестную структуру рекламного JSON.")
        if "report" in payload or "rows" in payload:
            if len(campaigns) != 1:
                raise PerformanceAPIError("В рекламном отчёте отсутствуют идентификаторы кампаний.")
            report = payload.get("report", payload)
            if not isinstance(report, dict):
                raise PerformanceAPIError("Ozon вернул неизвестную структуру рекламного отчёта.")
            add_rows(report.get("rows"), next(iter(campaigns)))
        else:
            if set(payload) != set(campaigns):
                raise PerformanceAPIError("Рекламный JSON не содержит все запрошенные кампании.")
            for campaign_id, entry in payload.items():
                if not isinstance(entry, dict):
                    raise PerformanceAPIError(
                        "Ozon вернул неизвестную структуру рекламного отчёта."
                    )
                report = entry.get("report", entry)
                if not isinstance(report, dict):
                    raise PerformanceAPIError(
                        "Ozon вернул неизвестную структуру рекламного отчёта."
                    )
                add_rows(report.get("rows"), campaign_id)
        return result
    if len(campaigns) != 1:
        raise PerformanceAPIError("Для нескольких кампаний Ozon должен вернуть JSON или ZIP.")
    add_rows(_csv_rows(data), next(iter(campaigns)))
    return result


class PerformanceClient:
    """OAuth, bounded retries and sequential polling; no modifications of advertising campaigns."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        poll_interval: float = 2.0,
        report_timeout: float = 120.0,
    ) -> None:
        if not client_id.strip() or not client_secret.strip():
            raise PerformanceAPIError("Укажите Client-ID и Client-Secret рекламного кабинета.")
        if poll_interval < 0 or report_timeout <= 0:
            raise ValueError("Polling interval/timeout must be nonnegative/positive.")
        self._client_id = client_id.strip()
        self._client_secret = client_secret.strip()
        self._access_token = ""
        self._expires_at = 0.0
        self._poll_interval = poll_interval
        self._report_timeout = report_timeout
        self._lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=False,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()
        self._access_token = ""
        self._client_id = ""
        self._client_secret = ""
        self._expires_at = 0.0

    async def _request(
        self, method: str, path: str, *, authenticated: bool = True, **kwargs: Any
    ) -> bytes:
        # Never follow report links to arbitrary hosts or leak the Authorization header there.
        if not path.startswith("/api/client/") or "://" in path:
            raise PerformanceAPIError("Недопустимый адрес рекламного API.")
        if authenticated:
            await self._ensure_token()
        refreshed = False
        for attempt in range(3):
            headers = {"Authorization": f"Bearer {self._access_token}"} if authenticated else {}
            try:
                async with self._http.stream(method, path, headers=headers, **kwargs) as response:
                    if response.status_code == 401 and authenticated and not refreshed:
                        self._access_token = ""
                        await self._ensure_token()
                        refreshed = True
                        continue
                    if response.status_code == 429 and attempt < 2:
                        try:
                            delay = min(
                                10.0, max(0.0, float(response.headers.get("Retry-After", "2")))
                            )
                        except ValueError:
                            delay = 2.0
                        await asyncio.sleep(delay)
                        continue
                    if response.status_code >= 500 and method == "GET" and attempt < 2:
                        await asyncio.sleep(2**attempt)
                        continue
                    if response.status_code >= 300:
                        if response.status_code in {401, 403}:
                            raise PerformanceAPIError(
                                "Ozon Performance отклонил доступ: проверьте Client-ID, "
                                "Client-Secret и права рекламного сервисного аккаунта."
                            )
                        if response.status_code == 429:
                            raise PerformanceAPIError(
                                "Лимит Ozon Performance исчерпан. Дождитесь завершения других "
                                "выгрузок и повторите запрос позже."
                            )
                        raise PerformanceAPIError(
                            f"Ozon Performance вернул HTTP {response.status_code}; "
                            "проверьте доступность этого типа отчёта и повторите позже."
                        )
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_REPORT_BYTES:
                            raise PerformanceAPIError(
                                "Ответ Ozon Performance слишком большой; сократите период."
                            )
                    return bytes(data)
            except httpx.RequestError:
                if method == "GET" and attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise PerformanceAPIError(
                    "Нет соединения с api-performance.ozon.ru. Проверьте сеть и повторите позже."
                ) from None
        raise PerformanceAPIError("Не удалось выполнить запрос Ozon Performance; повторите позже.")

    @staticmethod
    def _json(data: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise PerformanceAPIError("Ozon Performance вернул некорректный JSON.") from None
        if not isinstance(payload, dict):
            raise PerformanceAPIError("Ozon Performance вернул неизвестную структуру ответа.")
        return payload

    async def _ensure_token(self) -> None:
        if self._access_token and time.monotonic() < self._expires_at:
            return
        payload = self._json(
            await self._request(
                "POST",
                "/api/client/token",
                authenticated=False,
                json={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type": "client_credentials",
                },
            )
        )
        token = payload.get("access_token")
        try:
            expires_in = float(payload.get("expires_in", 1800))
        except (ValueError, TypeError):
            raise PerformanceAPIError(
                "Ozon Performance вернул неверный срок действия токена."
            ) from None
        if not isinstance(token, str) or not token or not 0 < expires_in <= 86_400:
            raise PerformanceAPIError("Ozon Performance не вернул действующий токен доступа.")
        token_type = payload.get("token_type", "Bearer")
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise PerformanceAPIError("Ozon Performance вернул неподдерживаемый тип токена.")
        self._access_token = token
        self._expires_at = time.monotonic() + max(1.0, expires_in - min(60.0, expires_in / 2))

    async def validate(self) -> None:
        """Check credentials and read access without generating a statistics report."""
        payload = self._json(
            await self._request("GET", "/api/client/campaign", params={"page": 1, "pageSize": 1})
        )
        if not isinstance(payload.get("list"), list):
            raise PerformanceAPIError("Ozon Performance не вернул список рекламных кампаний.")

    async def _campaigns(self) -> dict[str, str]:
        campaigns: dict[str, str] = {}
        for page in range(1, 102):
            payload = self._json(
                await self._request(
                    "GET", "/api/client/campaign", params={"page": page, "pageSize": 100}
                )
            )
            items = payload.get("list")
            if not isinstance(items, list):
                raise PerformanceAPIError("Ozon Performance не вернул список рекламных кампаний.")
            added = 0
            for item in items:
                if not isinstance(item, dict):
                    raise PerformanceAPIError("Неизвестная структура списка рекламных кампаний.")
                campaign_id = str(item.get("id", ""))
                if not campaign_id.isdigit():
                    raise PerformanceAPIError(
                        "В списке рекламы отсутствует идентификатор кампании."
                    )
                if campaign_id not in campaigns:
                    added += 1
                    campaigns[campaign_id] = str(item.get("title") or campaign_id)
            if len(campaigns) > 1000 or (len(items) >= 100 and not added):
                raise PerformanceAPIError(
                    "Список рекламных кампаний слишком большой или пагинация недоступна. "
                    "Используйте отдельный рекламный аккаунт с меньшим числом кампаний."
                )
            if len(items) < 100:
                return campaigns
        raise PerformanceAPIError("Превышен предел страниц рекламных кампаний.")

    async def _report(
        self, campaigns: dict[str, str], start: date, end: date
    ) -> list[dict[str, Any]]:
        created = self._json(
            await self._request(
                "POST",
                "/api/client/statistics/json",
                json={
                    "campaigns": list(campaigns),
                    "dateFrom": start.isoformat(),
                    "dateTo": end.isoformat(),
                    "groupBy": "DATE",
                },
            )
        )
        report_id = created.get("UUID")
        if not isinstance(report_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,100}", report_id):
            raise PerformanceAPIError("Ozon Performance не вернул идентификатор отчёта.")
        try:
            async with asyncio.timeout(self._report_timeout):
                for _ in range(120):
                    status = self._json(
                        await self._request("GET", f"/api/client/statistics/{report_id}")
                    )
                    state = status.get("state")
                    if state == "OK":
                        data = await self._request(
                            "GET", "/api/client/statistics/report", params={"UUID": report_id}
                        )
                        return _parse_report(data, campaigns, start, end)
                    if state == "ERROR":
                        raise PerformanceAPIError(
                            "Ozon не смог сформировать рекламный отчёт. Проверьте тип кампании, "
                            "права сервисного аккаунта и попробуйте меньший период."
                        )
                    if state not in {"NOT_STARTED", "IN_PROGRESS"}:
                        raise PerformanceAPIError(
                            "Ozon вернул неизвестный статус рекламного отчёта."
                        )
                    await asyncio.sleep(self._poll_interval)
        except TimeoutError:
            raise PerformanceAPIError(
                "Рекламный отчёт Ozon ещё не готов. Дождитесь окончания выгрузки и повторите позже."
            ) from None
        raise PerformanceAPIError("Рекламный отчёт Ozon ещё не готов; повторите позже.")

    async def load(self, start: date, end: date) -> tuple[list[dict[str, Any]], list[str]]:
        if start > end:
            raise PerformanceAPIError("Начало рекламного периода позже окончания.")
        async with self._lock:
            campaigns = await self._campaigns()
            if not campaigns:
                return [], ["В рекламном аккаунте Ozon нет кампаний."]
            windows: list[tuple[date, date]] = []
            cursor = start
            while cursor <= end:
                boundary = min(end, cursor + timedelta(days=61))
                windows.append((cursor, boundary))
                cursor = boundary + timedelta(days=1)
            items = list(campaigns.items())
            batches = [dict(items[i : i + 10]) for i in range(0, len(items), 10)]
            if len(windows) * len(batches) > 80:
                raise PerformanceAPIError(
                    "Для этого периода требуется слишком много рекламных выгрузок. "
                    "Выберите более короткий период; лимит Ozon — 10 кампаний и 62 дня на отчёт."
                )
            result: list[dict[str, Any]] = []
            for window_start, window_end in windows:
                for batch in batches:
                    rows = await self._report(batch, window_start, window_end)
                    if len(result) + len(rows) > MAX_ROWS:
                        raise PerformanceAPIError(
                            "Слишком много рекламных строк; выберите меньший период."
                        )
                    result.extend(rows)
            warnings = [
                (
                    "Рекламная выручка — атрибутированные заказы Ozon, а не фактические выкупы. "
                    "ДРР и ROAS рассчитаны по этой выручке; модельные заказы не добавляются повторно."
                )
            ]
            if not any(row.get("sku") for row in result):
                warnings.append(
                    "Ozon вернул показатели по кампаниям без SKU: "
                    "оценить отдельные невыгодные товары по этому отчёту нельзя."
                )
            return result, warnings
