"""Read-only Wildberries advertising API (Promotion), current v3 statistics.

Contracts: https://dev.wildberries.ru/docs/openapi/promotion . Credentials are
sent only to advert-api.wildberries.ru. Statistics use account currency units,
not kopecks; attributed order amounts are not settled marketplace revenue.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Self

import httpx

BASE_URL = "https://advert-api.wildberries.ru"
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 300_000
COUNT = "/adv/v1/promotion/count"
DETAILS = "/api/advert/v2/adverts"
STATS = "/adv/v3/fullstats"
_STANDARD_INTERVALS = {COUNT: 0.2, DETAILS: 0.2, STATS: 20.0}
_BASIC_INTERVALS = {COUNT: 900.0, DETAILS: 3600.0, STATS: 3600.0}
# Shared between short-lived clients; hashes contain no usable credentials.
_RATE_SLOTS: dict[tuple[str, str], float] = {}
_RATE_LOCK = asyncio.Lock()
_FIELDS = {
    "spend": "sum",
    "revenue": "sum_price",
    "orders": "orders",
    "clicks": "clicks",
    "impressions": "views",
}


class WildberriesAdsError(RuntimeError):
    """Sanitized, actionable error; never contains upstream bodies or keys."""


def _account_type(token: str) -> int:
    """Unverified JWT metadata selects pacing only; WB authenticates every request."""
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        value = claims.get("acc")
        return int(value) if not isinstance(value, bool) else 1
    except (IndexError, ValueError, TypeError, AttributeError):
        return 1  # Unknown tokens get the conservative Basic limits.


def _id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise WildberriesAdsError("WB вернул некорректный идентификатор рекламной кампании/товара.")
    value = str(value)
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise WildberriesAdsError("WB вернул некорректный идентификатор рекламной кампании/товара.")
    return value


def _metrics(item: dict[str, Any]) -> dict[str, Decimal]:
    result = {}
    for field, source in _FIELDS.items():
        value = item.get(source)
        try:
            if value is None or isinstance(value, bool):
                raise ValueError
            number = Decimal(str(value))
            if not number.is_finite() or number < 0 or number > Decimal("1e18"):
                raise ValueError
            if (
                field in {"orders", "clicks", "impressions"}
                and number != number.to_integral_value()
            ):
                raise ValueError
        except (InvalidOperation, ValueError):
            raise WildberriesAdsError(
                f"В рекламном отчёте WB отсутствует или некорректно поле {source}. "
                "Неизвестные показатели не заменяются нулями."
            ) from None
        result[field] = number
    return result


def _matches(parent: dict[str, Any], children: list[dict[str, Any]]) -> bool:
    totals = _metrics(parent)
    for field, value in totals.items():
        subtotal = sum((child[field] for child in children), Decimal(0))
        tolerance = Decimal("0.02") if field in {"spend", "revenue"} else Decimal(0)
        if abs(value - subtotal) > tolerance:
            return False
    return True


def _normalize(
    payload: Any, campaigns: dict[str, str], start: date, end: date
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(payload, list):
        raise WildberriesAdsError("WB вернул неизвестную структуру рекламной статистики.")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_campaigns: set[str] = set()
    missing_breakdown = False
    for campaign in payload:
        if not isinstance(campaign, dict):
            raise WildberriesAdsError("WB вернул некорректные данные рекламной кампании.")
        campaign_id = _id(campaign.get("advertId"))
        if campaign_id not in campaigns or campaign_id in seen_campaigns:
            raise WildberriesAdsError("WB вернул лишнюю или повторяющуюся рекламную кампанию.")
        seen_campaigns.add(campaign_id)
        if campaign.get("currency") != "RUB":
            raise WildberriesAdsError(
                "Рекламный счёт WB должен быть в RUB; смешивать валюты без курса нельзя."
            )
        days = campaign.get("days")
        if not isinstance(days, list):
            raise WildberriesAdsError("В рекламном отчёте WB отсутствует разбивка по дням.")
        seen_dates: set[str] = set()
        day_totals: list[dict[str, Decimal]] = []
        for day in days:
            if not isinstance(day, dict):
                raise WildberriesAdsError("WB вернул некорректный день рекламного отчёта.")
            try:
                day_date = datetime.fromisoformat(str(day.get("date"))).date()
            except ValueError:
                raise WildberriesAdsError(
                    "WB вернул некорректную дату рекламного отчёта."
                ) from None
            if not start <= day_date <= end or day_date.isoformat() in seen_dates:
                raise WildberriesAdsError(
                    "В рекламном отчёте WB повторяются дни или нарушен период."
                )
            seen_dates.add(day_date.isoformat())
            day_totals.append(_metrics(day))
            apps = day.get("apps")
            leaves: list[dict[str, Any]] = []
            app_totals: list[dict[str, Decimal]] = []
            seen_apps: set[int] = set()
            if apps is not None and not isinstance(apps, list):
                raise WildberriesAdsError("WB вернул некорректную разбивку по приложениям.")
            for app in apps or []:
                if not isinstance(app, dict) or type(app.get("appType")) is not int:
                    raise WildberriesAdsError("WB вернул некорректный тип приложения в рекламе.")
                if app["appType"] in seen_apps:
                    raise WildberriesAdsError("В рекламном отчёте WB повторяется приложение.")
                seen_apps.add(app["appType"])
                app_metrics = _metrics(app)
                app_totals.append(app_metrics)
                nms = app.get("nms")
                if nms is not None and not isinstance(nms, list):
                    raise WildberriesAdsError("WB вернул некорректную разбивку рекламы по товарам.")
                nm_rows: list[dict[str, Any]] = []
                seen_nms: set[str] = set()
                for nm in nms or []:
                    if not isinstance(nm, dict):
                        raise WildberriesAdsError(
                            "WB вернул некорректные показатели товара в рекламе."
                        )
                    sku = _id(nm.get("nmId"))
                    if sku in seen_nms:
                        raise WildberriesAdsError(
                            "В одном приложении WB повторяется товар рекламы."
                        )
                    seen_nms.add(sku)
                    nm_rows.append({"sku": sku, **_metrics(nm)})
                if not _matches(app, nm_rows):
                    # Preserve reported app totals, without inventing SKU attribution.
                    nm_rows = [{"sku": "", **app_metrics}]
                    missing_breakdown = True
                leaves.extend(nm_rows)
            if not _matches(day, app_totals):
                leaves = [{"sku": "", **day_totals[-1]}]
                missing_breakdown = True
            combined: dict[str, dict[str, Decimal]] = {}
            for leaf in leaves:
                target = combined.setdefault(leaf["sku"], {field: Decimal(0) for field in _FIELDS})
                for field in _FIELDS:
                    target[field] += leaf[field]
            for sku, totals in combined.items():
                rows.append(
                    {
                        "date": day_date.isoformat(),
                        "campaign_id": campaign_id,
                        "campaign_name": campaigns[campaign_id],
                        "sku": sku,
                        **{
                            field: int(value)
                            if field in {"orders", "clicks", "impressions"}
                            else float(value)
                            for field, value in totals.items()
                        },
                    }
                )
                if len(rows) > MAX_ROWS:
                    raise WildberriesAdsError(
                        "Рекламный отчёт WB слишком большой; сократите период."
                    )
        if not _matches(campaign, day_totals):
            raise WildberriesAdsError(
                "Дневные показатели WB не совпадают с итогами кампании. "
                "Отчёт ещё обновляется или неполон; повторите позже."
            )
    if set(campaigns) != seen_campaigns:
        warnings.append(
            f"WB не вернул статистику для {len(set(campaigns) - seen_campaigns)} кампаний. "
            "Их показатели неизвестны; рекламные итоги могут быть неполными."
        )
    if missing_breakdown:
        warnings.append(
            "Часть рекламных показателей WB доступна только по кампании или приложению. "
            "Эти суммы учтены без SKU и не распределяются между товарами."
        )
    return rows, warnings


class WildberriesAdsClient:
    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise WildberriesAdsError("Укажите токен WB с доступом к категории «Продвижение».")
        self._key = api_key.strip()
        self._token_type = _account_type(self._key)
        if self._token_type == 2:
            raise WildberriesAdsError(
                "Тестовый токен WB работает только в песочнице, нужен рабочий."
            )
        self._fingerprint = hashlib.sha256(self._key.encode()).hexdigest()
        self._intervals = _STANDARD_INTERVALS if self._token_type in {3, 4} else _BASIC_INTERVALS
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            transport=transport,
            timeout=httpx.Timeout(45, connect=10),
            follow_redirects=False,
        )
        self._counts: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()
        self._key = ""
        self._counts = None

    async def _pace(self, path: str, *, defer: float = 0) -> None:
        now = time.monotonic()
        slot = (self._fingerprint, path)
        async with _RATE_LOCK:
            if len(_RATE_SLOTS) > 4096:
                expired = [key for key, due in _RATE_SLOTS.items() if due <= now]
                for key in expired:
                    del _RATE_SLOTS[key]
            due = max(_RATE_SLOTS.get(slot, now), now + defer)
            delay = due - now
            if delay > 60:
                _RATE_SLOTS[slot] = due
                raise WildberriesAdsError(
                    "Лимит рекламного API WB требует длительного ожидания. "
                    "Для базового токена статистика доступна раз в час; "
                    "используйте персональный/сервисный токен или повторите позже."
                )
            _RATE_SLOTS[slot] = due + self._intervals[path]
        if delay > 0:
            await asyncio.sleep(delay)

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        if path not in {COUNT, DETAILS, STATS}:
            raise WildberriesAdsError("Недопустимый адрес рекламного API WB.")
        defer = 0.0
        for attempt in range(3):
            await self._pace(path, defer=defer)
            try:
                async with self._http.stream(
                    "GET", path, params=params, headers={"Authorization": self._key}
                ) as response:
                    if response.status_code == 429:
                        try:
                            defer = float(
                                response.headers.get(
                                    "X-Ratelimit-Retry", response.headers.get("Retry-After", "20")
                                )
                            )
                            if not math.isfinite(defer) or defer < 0:
                                raise ValueError
                        except ValueError:
                            defer = 20.0
                        async with _RATE_LOCK:
                            slot = (self._fingerprint, path)
                            _RATE_SLOTS[slot] = max(
                                _RATE_SLOTS.get(slot, 0), time.monotonic() + defer
                            )
                        if attempt < 2:
                            continue
                        raise WildberriesAdsError(
                            "Лимит рекламного API WB исчерпан; повторите запрос позже."
                        )
                    if response.status_code >= 500 and attempt < 2:
                        defer = 2.0**attempt
                        continue
                    if response.status_code in {401, 403}:
                        raise WildberriesAdsError(
                            "WB отклонил рекламный токен. Проверьте срок действия и доступ "
                            "к категории «Продвижение»; Client-ID не требуется."
                        )
                    if response.status_code >= 300:
                        raise WildberriesAdsError(
                            f"Рекламный API WB вернул HTTP {response.status_code}. "
                            "Проверьте права токена, выбранный период и повторите позже."
                        )
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_BYTES:
                            raise WildberriesAdsError(
                                "Ответ рекламы WB слишком большой; сократите период."
                            )
                    try:
                        return json.loads(data)
                    except (ValueError, UnicodeDecodeError):
                        raise WildberriesAdsError(
                            "Рекламный API WB вернул некорректный JSON."
                        ) from None
            except httpx.RequestError:
                if attempt < 2:
                    defer = 2.0**attempt
                    continue
                raise WildberriesAdsError(
                    "Нет соединения с advert-api.wildberries.ru; проверьте сеть и повторите позже."
                ) from None
        raise WildberriesAdsError("Рекламный API WB временно недоступен; повторите позже.")

    async def validate(self) -> None:
        data = await self._get(COUNT)
        if not isinstance(data, dict) or not isinstance(data.get("adverts"), (list, type(None))):
            raise WildberriesAdsError("WB не вернул список рекламных кампаний.")
        if "adverts" not in data or "all" not in data:
            raise WildberriesAdsError("WB не вернул список рекламных кампаний.")
        if type(data["all"]) is not int or data["all"] < 0 or (data["all"] and not data["adverts"]):
            raise WildberriesAdsError("WB вернул неполный список рекламных кампаний.")
        self._counts = data

    async def load(self, start: date, end: date) -> tuple[list[dict[str, Any]], list[str]]:
        if start > end:
            raise WildberriesAdsError("Начало рекламного периода позже окончания.")
        async with self._lock:
            if self._counts is None:
                await self.validate()
            selected: dict[str, str] = {}
            skipped = 0
            for group in self._counts["adverts"] or []:
                if not isinstance(group, dict) or not isinstance(group.get("advert_list"), list):
                    raise WildberriesAdsError("WB вернул некорректный список рекламных кампаний.")
                if group.get("status") not in {7, 9, 11}:
                    skipped += len(group["advert_list"])
                    continue
                for item in group["advert_list"]:
                    if not isinstance(item, dict):
                        raise WildberriesAdsError("WB вернул некорректную рекламную кампанию.")
                    campaign_id = _id(item.get("advertId"))
                    selected[campaign_id] = f"Кампания {campaign_id}"
            warnings = []
            if skipped:
                warnings.append(
                    f"Не включены {skipped} кампаний со статусами, для которых WB не выдаёт "
                    "fullstats. Доступны завершённые, активные и приостановленные (7, 9, 11)."
                )
            if not selected:
                return [], warnings + ["В WB нет кампаний со статистикой в доступных статусах."]
            ids = list(selected)
            windows = (end - start).days // 31 + 1
            if len(ids) > 1000 or windows * ((len(ids) + 49) // 50) > 24:
                raise WildberriesAdsError(
                    "Для этого периода слишком много рекламных запросов WB. "
                    "Сократите период: лимит одного запроса — 50 кампаний и 31 день."
                )
            missing_names = False
            for offset in range(0, len(ids), 50):
                batch = ids[offset : offset + 50]
                details = await self._get(DETAILS, params={"ids": ",".join(batch)})
                if not isinstance(details, dict) or not isinstance(details.get("adverts"), list):
                    raise WildberriesAdsError("WB не вернул подробности рекламных кампаний.")
                found = set()
                for item in details["adverts"]:
                    if not isinstance(item, dict):
                        raise WildberriesAdsError("WB вернул некорректные сведения о рекламе.")
                    campaign_id = _id(item.get("id"))
                    if campaign_id not in batch or campaign_id in found:
                        raise WildberriesAdsError("WB вернул лишние или повторяющиеся кампании.")
                    found.add(campaign_id)
                    settings = item.get("settings")
                    if isinstance(settings, dict) and isinstance(settings.get("name"), str):
                        selected[campaign_id] = settings["name"] or selected[campaign_id]
                    else:
                        missing_names = True
                missing_names |= found != set(batch)
            if missing_names:
                warnings.append("Для части кампаний WB не вернул название; показаны их ID.")
            rows = []
            cursor = start
            while cursor <= end:
                boundary = min(end, cursor + timedelta(days=30))
                for offset in range(0, len(ids), 50):
                    batch = ids[offset : offset + 50]
                    payload = await self._get(
                        STATS,
                        params={
                            "ids": ",".join(batch),
                            "beginDate": cursor.isoformat(),
                            "endDate": boundary.isoformat(),
                        },
                    )
                    normalized, notes = _normalize(
                        payload, {key: selected[key] for key in batch}, cursor, boundary
                    )
                    rows.extend(normalized)
                    warnings.extend(notes)
                    if len(rows) > MAX_ROWS:
                        raise WildberriesAdsError(
                            "Рекламный отчёт WB слишком большой; сократите период."
                        )
                cursor = boundary + timedelta(days=1)
            warnings.append(
                "Рекламная выручка WB — сумма атрибутированных заказов, а не выкупы и не "
                "чистая прибыль. ДРР/ROAS рассчитаны по ней; суммы указаны в рублях."
            )
            return rows, list(dict.fromkeys(warnings))
