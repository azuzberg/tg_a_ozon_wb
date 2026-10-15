"""Read-only Wildberries sources with explicit source and completeness metadata.

The API contracts follow dev.wildberries.ru OpenAPI (2026-10-03 snapshot). In
particular finance uses finance-api and camelCase; the removed Statistics v5
finance and supplier/stocks methods are deliberately not used. POSTing a finance
report or a content listing is a read-only operation in the WB contract.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Self
from uuid import UUID

import httpx

from .domain import Snapshot


class WildberriesAPIError(Exception):
    """An error safe for Telegram; response bodies and credentials are excluded."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str = "",
        status_code: int | None = None,
        code: str = "upstream",
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.code = code


# The short name mirrors PerformanceError and is used by the bot's error path.
WildberriesError = WildberriesAPIError


_FINANCE_FIELDS = (
    "rrdId",
    "nmId",
    "currency",
    "docTypeName",
    "sellerOperName",
    "rrDate",
    "quantity",
    "retailAmount",
    "forPay",
    "ppvzSalesCommission",
    "acquiringFee",
    "vw",
    "vwNds",
    "deliveryService",
    "paidStorage",
    "deduction",
    "paidAcceptance",
    "penalty",
    "additionalPayment",
    "rebillLogisticCost",
    "cashbackAmount",
    "cashbackDiscount",
    "cashbackCommissionChange",
    "paymentSchedule",
    "installmentCofinancingAmount",
    "vendorCode",
    "subjectName",
    "title",
    "agencyVat",
)
_DEDUCTIONS = {
    "deliveryService": "Логистика WB",
    "paidStorage": "Хранение WB",
    "deduction": "Прочие удержания WB",
    "paidAcceptance": "Приёмка WB",
    "penalty": "Штрафы WB",
    "rebillLogisticCost": "Перевыставленная логистика WB",
}
_AMBIGUOUS_ADJUSTMENTS = (
    "additionalPayment",
    "cashbackAmount",
    "cashbackDiscount",
    "cashbackCommissionChange",
    "paymentSchedule",
    "installmentCofinancingAmount",
    "agencyVat",
)


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _integer(value: Any, *, positive: bool = False) -> int | None:
    number = _number(value)
    if number is None or number != number.to_integral_value():
        return None
    result = int(number)
    return result if result >= int(positive) else None


def _day(value: Any, endpoint: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise WildberriesAPIError(
            "Wildberries вернул некорректную дату операции.", endpoint=endpoint, code="schema"
        ) from None


class WildberriesClient:
    """Use one seller token with Statistics, Finance, Analytics and Content rights.

    Each optional source can be unavailable independently. No guessed zeros are
    substituted for unavailable sources. WB's 60-second pagination limits are
    observed by default; independent sources load concurrently. ``request_interval``
    and ``poll_interval`` exist for deterministic transport tests.
    """

    STATISTICS = "statistics-api.wildberries.ru"
    FINANCE = "finance-api.wildberries.ru"
    CONTENT = "content-api.wildberries.ru"
    ANALYTICS = "seller-analytics-api.wildberries.ru"
    FINANCE_PATH = "/api/finance/v1/sales-reports/detailed"
    ORDERS_PATH = "/api/v1/supplier/orders"
    SALES_PATH = "/api/v1/supplier/sales"
    CARDS_PATH = "/content/v2/get/cards/list"
    STOCKS_PATH = "/api/v1/warehouse_remains"

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        timeout: float = 40.0,
        max_retries: int = 2,
        max_pages: int = 100,
        max_records: int = 100_000,
        page_size: int = 100_000,
        retry_base_delay: float = 0.5,
        request_interval: float = 60.0,
        poll_interval: float = 5.0,
        max_polls: int = 12,
        today: date | None = None,
    ) -> None:
        api_key = api_key.strip()
        if (
            not api_key
            or not api_key.isascii()
            or len(api_key) > 20_000
            or any(ord(char) <= 32 or ord(char) == 127 for char in api_key)
        ):
            raise ValueError("Укажите корректный API-токен Wildberries.")
        if (
            max_pages < 1
            or max_records < 1
            or not 1 <= page_size <= 100_000
            or max_retries < 0
            or max_polls < 1
            or not all(
                math.isfinite(n) and n >= 0
                for n in (
                    retry_base_delay,
                    request_interval,
                    poll_interval,
                )
            )
        ):
            raise ValueError("Недопустимые ограничения запросов Wildberries.")
        self._headers = {"Authorization": api_key}
        self._transport = transport
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_pages = max_pages
        self._max_records = max_records
        self._page_size = page_size
        self._retry_base_delay = retry_base_delay
        self._request_interval = request_interval
        self._poll_interval = poll_interval
        self._max_polls = max_polls
        self._today = today or date.today()
        self._http: httpx.AsyncClient | None = None
        self._last_requests: dict[tuple[str, str], float] = {}
        self._warnings: list[str] = []

    async def __aenter__(self) -> Self:
        if self._http is not None or not self._headers:
            raise RuntimeError("Этот WildberriesClient уже открыт или закрыт.")
        self._http = httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout,
            follow_redirects=False,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._headers.clear()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @staticmethod
    def _schema_error(endpoint: str) -> WildberriesAPIError:
        return WildberriesAPIError(
            "Wildberries вернул неожиданный формат данных; источник не использован.",
            endpoint=endpoint,
            code="schema",
        )

    @staticmethod
    def _pagination_error(endpoint: str) -> WildberriesAPIError:
        return WildberriesAPIError(
            "Загрузка Wildberries неполна: нарушена пагинация или достигнут лимит. "
            "Выберите меньший период; неполные итоги не используются.",
            endpoint=endpoint,
            code="pagination",
        )

    @classmethod
    def _allowed(cls, host: str, method: str, endpoint: str) -> bool:
        fixed = {
            (cls.STATISTICS, "GET", "/ping"),
            (cls.STATISTICS, "GET", cls.ORDERS_PATH),
            (cls.STATISTICS, "GET", cls.SALES_PATH),
            (cls.FINANCE, "POST", cls.FINANCE_PATH),
            (cls.CONTENT, "POST", cls.CARDS_PATH),
            (cls.ANALYTICS, "GET", cls.STOCKS_PATH),
        }
        return (host, method, endpoint) in fixed or (
            host == cls.ANALYTICS
            and method == "GET"
            and bool(
                re.fullmatch(
                    r"/api/v1/warehouse_remains/tasks/[a-fA-F0-9-]{36}/(?:status|download)",
                    endpoint,
                )
            )
        )

    async def _request(
        self,
        host: str,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if not self._allowed(host, method, endpoint):
            raise ValueError("Разрешены только известные методы чтения Wildberries.")
        if self._http is None:
            raise RuntimeError("Используйте WildberriesClient через async with.")
        for attempt in range(self._max_retries + 1):
            key = (host, endpoint)
            interval = (
                self._poll_interval
                if endpoint.endswith("/status")
                else min(self._request_interval, 0.6)
                if host == self.CONTENT
                else self._request_interval
            )
            wait = interval - (time.monotonic() - self._last_requests.get(key, -math.inf))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_requests[key] = time.monotonic()
            try:
                response = await self._http.request(
                    method,
                    f"https://{host}{endpoint}",
                    headers=self._headers,
                    params=params,
                    json=payload,
                )
            except httpx.RequestError:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_base_delay * 2**attempt)
                    continue
                raise WildberriesAPIError(
                    "Не удалось связаться с Wildberries. Повторите позже.",
                    endpoint=endpoint,
                    code="network",
                ) from None
            if response.status_code in {401, 403}:
                raise WildberriesAPIError(
                    "Wildberries отказал в доступе. Проверьте токен и права соответствующей категории.",
                    endpoint=endpoint,
                    status_code=response.status_code,
                    code="authentication",
                )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                delay = self._retry_base_delay * 2**attempt
                try:
                    delay = max(delay, float(response.headers.get("Retry-After", "0")))
                except ValueError:
                    pass
                if attempt < self._max_retries and math.isfinite(delay) and delay <= 60:
                    await asyncio.sleep(max(delay, 0))
                    continue
            if not response.is_success:
                raise WildberriesAPIError(
                    f"Wildberries не выполнил запрос (HTTP {response.status_code}). "
                    + (
                        "Достигнут лимит запросов, повторите позже."
                        if response.status_code == 429
                        else "Повторите позже или проверьте доступ к источнику."
                    ),
                    endpoint=endpoint,
                    status_code=response.status_code,
                    code="rate_limit" if response.status_code == 429 else "upstream",
                )
            if response.status_code == 204:
                return []
            try:
                return response.json()
            except ValueError:
                raise self._schema_error(endpoint) from None
        raise AssertionError("Unreachable retry state")

    def _rows(self, value: Any, endpoint: str) -> list[dict[str, Any]]:
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise self._schema_error(endpoint)
        if len(value) > self._max_records:
            raise self._pagination_error(endpoint)
        return value

    async def validate(self) -> None:
        """A cheap category-aware token validation without loading seller data."""
        result = await self._request(self.STATISTICS, "GET", "/ping")
        if not isinstance(result, dict) or result.get("Status") != "OK":
            raise self._schema_error("/ping")

    async def check_credentials(self) -> None:
        await self.validate()

    async def _finance(self, start: date, end: date) -> list[dict[str, Any]]:
        endpoint = self.FINANCE_PATH
        if start < date(2024, 1, 29):
            raise WildberriesAPIError(
                "Финансовые отчёты WB доступны только с 29.01.2024; выберите более поздний период.",
                endpoint=endpoint,
                code="retention",
            )
        rows: list[dict[str, Any]] = []
        cursor = 0
        for _ in range(self._max_pages):
            page = self._rows(
                await self._request(
                    self.FINANCE,
                    "POST",
                    endpoint,
                    payload={
                        "dateFrom": start.isoformat(),
                        "dateTo": f"{end.isoformat()}T23:59:59",
                        "period": "daily",
                        "limit": self._page_size,
                        "rrdId": cursor,
                        "fields": list(_FINANCE_FIELDS),
                    },
                ),
                endpoint,
            )
            if not page:
                return rows
            ids = [_integer(row.get("rrdId"), positive=True) for row in page]
            if any(value is None for value in ids) or any(
                later <= earlier for earlier, later in zip([cursor, *ids], ids, strict=False)
            ):
                raise self._pagination_error(endpoint)
            rows.extend(page)
            if len(rows) > self._max_records:
                raise self._pagination_error(endpoint)
            cursor = ids[-1]
        raise self._pagination_error(endpoint)

    async def _statistics(self, endpoint: str, start: date) -> list[dict[str, Any]]:
        if start < self._today - timedelta(days=89):
            raise WildberriesAPIError(
                "Оперативные заказы и выкупы WB хранятся не более 90 дней; "
                "для этого периода доступны только финансовые отчёты.",
                endpoint=endpoint,
                code="retention",
            )
        cursor = start.isoformat()
        previous = datetime.fromisoformat(cursor)
        seen: set[str] = set()
        rows: dict[str, dict[str, Any]] = {}
        for _ in range(self._max_pages):
            page = self._rows(
                await self._request(
                    self.STATISTICS,
                    "GET",
                    endpoint,
                    params={
                        "dateFrom": cursor,
                        "flag": 0,
                    },
                ),
                endpoint,
            )
            if not page:
                return list(rows.values())
            for row in page:
                identifier = row.get("saleID" if endpoint == self.SALES_PATH else "srid")
                if not isinstance(identifier, str) or not identifier:
                    raise self._schema_error(endpoint)
                rows[identifier] = row
            if len(rows) > self._max_records:
                raise self._pagination_error(endpoint)
            next_cursor = page[-1].get("lastChangeDate")
            if not isinstance(next_cursor, str):
                raise self._schema_error(endpoint)
            try:
                next_time = datetime.fromisoformat(next_cursor)
                # The documented WB timestamps are Moscow local timestamps.
                next_time = next_time.replace(tzinfo=None)
            except ValueError:
                raise self._schema_error(endpoint) from None
            if next_cursor in seen or next_time <= previous:
                raise self._pagination_error(endpoint)
            seen.add(next_cursor)
            cursor, previous = next_cursor, next_time
        raise self._pagination_error(endpoint)

    async def _cards(self) -> list[dict[str, Any]]:
        endpoint = self.CARDS_PATH
        cursor: dict[str, Any] = {"limit": 100}
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        previous: tuple[float, int] | None = None
        for _ in range(self._max_pages):
            result = await self._request(
                self.CONTENT,
                "POST",
                endpoint,
                payload={
                    "settings": {
                        "sort": {"ascending": True},
                        "cursor": cursor,
                        "filter": {"withPhoto": -1},
                    }
                },
            )
            if not isinstance(result, dict) or not isinstance(result.get("cursor"), dict):
                raise self._schema_error(endpoint)
            page = self._rows(result.get("cards"), endpoint)
            total = _integer(result["cursor"].get("total"))
            if total is None or total != len(page) or total > 100:
                raise self._schema_error(endpoint)
            rows.extend(page)
            if len(rows) > self._max_records:
                raise self._pagination_error(endpoint)
            if total < 100:
                return rows
            updated = result["cursor"].get("updatedAt")
            nm_id = _integer(result["cursor"].get("nmID"), positive=True)
            if not isinstance(updated, str) or not updated or nm_id is None:
                raise self._pagination_error(endpoint)
            key = (updated, nm_id)
            try:
                position = (datetime.fromisoformat(updated).timestamp(), nm_id)
            except (ValueError, OverflowError, OSError):
                raise self._pagination_error(endpoint) from None
            if key in seen or previous is not None and position <= previous:
                raise self._pagination_error(endpoint)
            seen.add(key)
            previous = position
            cursor = {"limit": 100, "updatedAt": updated, "nmID": nm_id}
        raise self._pagination_error(endpoint)

    async def _stocks(self) -> list[dict[str, Any]]:
        endpoint = self.STOCKS_PATH
        result = await self._request(
            self.ANALYTICS,
            "GET",
            endpoint,
            params={
                "locale": "en",
                "groupByNm": "true",
                "groupBySa": "true",
                "groupBySubject": "true",
                "groupByBrand": "true",
            },
        )
        try:
            task_id = str(UUID(result["data"]["taskId"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise self._schema_error(endpoint) from None
        status_path = f"{endpoint}/tasks/{task_id}/status"
        for attempt in range(self._max_polls):
            status = await self._request(self.ANALYTICS, "GET", status_path)
            if not isinstance(status, dict) or not isinstance(status.get("data"), dict):
                raise self._schema_error(status_path)
            state = status["data"].get("status")
            if state == "done":
                return self._rows(
                    await self._request(
                        self.ANALYTICS,
                        "GET",
                        f"{endpoint}/tasks/{task_id}/download",
                    ),
                    endpoint,
                )
            if state not in {"new", "processing"}:
                raise WildberriesAPIError(
                    "WB не сформировал отчёт об остатках; источник недоступен.",
                    endpoint=endpoint,
                    code="report",
                )
            if attempt + 1 < self._max_polls:
                await asyncio.sleep(self._poll_interval)
        raise WildberriesAPIError(
            "Отчёт об остатках WB ещё формируется. Повторите загрузку позже.",
            endpoint=endpoint,
            code="report",
        )

    @staticmethod
    def _product(row: dict[str, Any], *, card: bool = False) -> dict[str, Any]:
        sku = _integer(row.get("nmID" if card else "nmId"), positive=True)
        if sku is None:
            raise WildberriesClient._schema_error("products")
        product = {
            "id": sku,
            "sku": sku,
            "offer_id": str(row.get("vendorCode") or row.get("supplierArticle") or ""),
            "name": str(row.get("title") or ""),
            "category": str(row.get("subjectName") or row.get("subject") or "Без категории"),
        }
        if card:
            product["description"] = str(row.get("description") or "")
            photos = row.get("photos") or []
            if not isinstance(photos, list):
                raise WildberriesClient._schema_error(WildberriesClient.CARDS_PATH)
            product["images"] = [
                str(photo["big"])
                for photo in photos
                if isinstance(photo, dict) and isinstance(photo.get("big"), str)
            ]
        return product

    def _transaction(self, row: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.FINANCE_PATH
        currency = str(row.get("currency") or "").upper()
        if currency not in {"RUB", "RUR"}:
            raise WildberriesAPIError(
                "В финансовом отчёте WB отсутствует валюта RUB или есть другая валюта. "
                "Суммы разных валют не складываются.",
                endpoint=endpoint,
                code="currency",
            )
        document = str(row.get("docTypeName") or "").casefold().strip()
        direction = {"продажа": 1, "sale": 1, "возврат": -1, "return": -1}.get(document)
        gross = _number(row.get("retailAmount"))
        net = _number(row.get("forPay"))
        fee = _number(row.get("ppvzSalesCommission"))
        incomplete = any(value is None for value in (gross, net, fee))
        if direction is None:
            if gross or net or fee:
                incomplete = True
            direction = 1
        gross = direction * abs(gross) if gross is not None else None
        net = direction * abs(net) if net is not None else None
        commission = -direction * fee if fee is not None else None
        services: list[dict[str, Any]] = []
        deduction_sum = Decimal(0)
        for field, name in _DEDUCTIONS.items():
            value = _number(row.get(field))
            if value is None:
                incomplete = True
            else:
                deduction_sum += value
                if value:
                    services.append({"name": name, "price": float(-value)})
        for field in _AMBIGUOUS_ADJUSTMENTS:
            value = _number(row.get(field))
            # agencyVat is optional and only applies to sellers in Kyrgyzstan.
            # Other settlement fields are part of the requested required schema.
            if (value is None and field != "agencyVat") or value:
                incomplete = True
        amount = net - deduction_sum if net is not None else None
        if gross is not None and commission is not None and net is not None:
            # forPay is already net of acquiring/VAT and the WB fee: recording
            # these again would double count costs. Reconcile them as one sum.
            residual = net - gross - commission
            if residual:
                services.append(
                    {
                        "name": "Прочие комиссии и корректировки WB (включая НДС и эквайринг)",
                        "price": float(residual),
                    }
                )
        sku = _integer(row.get("nmId"), positive=True)
        quantity = _integer(row.get("quantity"), positive=True)
        items = []
        goods = document in {"продажа", "sale", "возврат", "return"} and quantity is not None
        if goods and gross == 0:
            # The common finance pipeline allocates COGS on non-zero revenue.
            # Free goods still have a real cost: do not turn omitted COGS into a
            # seemingly exact profit until that allocation can be represented.
            incomplete = True
        if gross or goods:
            if sku is None or quantity is None:
                incomplete = True
            if sku is not None:
                items = [
                    {
                        "sku": sku,
                        "offer_id": str(row.get("vendorCode") or ""),
                        "quantity": quantity if quantity is not None else 0,
                    }
                ]
        return {
            "operation_id": row["rrdId"],
            "operation_date": _day(row.get("rrDate"), endpoint).isoformat(),
            "operation_type_name": str(row.get("sellerOperName") or row.get("docTypeName") or ""),
            "amount": float(amount) if amount is not None else None,
            "accruals_for_sale": float(gross) if gross is not None else None,
            "sale_commission": float(commission) if commission is not None else None,
            "services": services,
            "items": items,
            "incomplete": incomplete,
            "currency": "RUB",
        }

    def _order(self, row: dict[str, Any]) -> dict[str, Any]:
        sku = _integer(row.get("nmId"), positive=True)
        if sku is None:
            raise self._schema_error(self.ORDERS_PATH)
        price = _number(row.get("priceWithDisc"))
        return {
            "posting_number": str(row["srid"]),
            "created_at": _day(row.get("date"), self.ORDERS_PATH).isoformat(),
            "status": "cancelled" if row.get("isCancel") is True else "pending",
            # WB documents one row as exactly one item, not an unknown quantity.
            "products": [
                {
                    "sku": sku,
                    "offer_id": str(row.get("supplierArticle") or ""),
                    "quantity": 1,
                    "price": float(price) if price is not None else None,
                }
            ],
        }

    def _sale(self, row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row["saleID"])
        sku = _integer(row.get("nmId"), positive=True)
        if sku is None or not identifier.startswith(("S", "R")):
            raise self._schema_error(self.SALES_PATH)
        price = _number(row.get("priceWithDisc"))
        return {
            "sale_id": identifier,
            "date": _day(row.get("date"), self.SALES_PATH).isoformat(),
            "sku": sku,
            "offer_id": str(row.get("supplierArticle") or ""),
            "quantity": 1,
            "price": float(abs(price)) if price is not None else None,
            "status": "returned" if identifier.startswith("R") else "delivered",
        }

    def _stock(self, row: dict[str, Any]) -> dict[str, Any]:
        sku = _integer(row.get("nmId"), positive=True)
        warehouses = row.get("warehouses")
        if sku is None or not isinstance(warehouses, list):
            raise self._schema_error(self.STOCKS_PATH)
        physical = total = 0
        total_found = False
        for warehouse in warehouses:
            if not isinstance(warehouse, dict):
                raise self._schema_error(self.STOCKS_PATH)
            quantity = _integer(warehouse.get("quantity"))
            name = str(warehouse.get("warehouseName") or "").casefold().strip()
            if quantity is None or not name:
                raise self._schema_error(self.STOCKS_PATH)
            if name in {"total in warehouses", "всего находится на складах", "всего на складах"}:
                if total_found:
                    raise self._schema_error(self.STOCKS_PATH)
                total, total_found = quantity, True
            elif not any(
                word in name
                for word in (
                    "to the client",
                    "from the client",
                    "in transit",
                    "в пути",
                    "к клиент",
                    "от клиент",
                )
            ):
                physical += quantity
        return {
            "sku": sku,
            "offer_id": str(row.get("vendorCode") or ""),
            "present": total if total_found else physical,
            "reserved": None,
            "reserved_known": False,
            "snapshot_date": self._today.isoformat(),
        }

    async def load(self, start: date, end: date) -> Snapshot:
        if start > end or end > self._today:
            raise ValueError("Проверьте выбранный период Wildberries.")
        self._warnings = []
        snapshot = Snapshot(marketplace="wildberries")
        jobs = (
            ("finance", "Финансы (право «Финансы»)", self._finance(start, end)),
            ("orders", "Заказы (право «Статистика»)", self._statistics(self.ORDERS_PATH, start)),
            (
                "sales",
                "Выкупы и возвраты (право «Статистика»)",
                self._statistics(self.SALES_PATH, start),
            ),
            ("products", "Карточки (право «Контент»)", self._cards()),
            ("stocks", "Остатки (право «Аналитика»)", self._stocks()),
        )
        results = await asyncio.gather(*(job for _, _, job in jobs), return_exceptions=True)
        products: dict[int, dict[str, Any]] = {}
        for (source, label, _), result in zip(jobs, results, strict=True):
            snapshot.sources[source] = False
            try:
                if isinstance(result, WildberriesAPIError):
                    raise result
                if isinstance(result, BaseException):
                    raise result
                # Normalize a whole source before exposing it. A malformed page
                # must not leave usable-looking partial financial/sales totals.
                valid = []
                for row in result:
                    if source == "finance":
                        value = self._transaction(row)
                        if not start <= _day(value["operation_date"], self.FINANCE_PATH) <= end:
                            raise self._schema_error(self.FINANCE_PATH)
                    elif source == "orders":
                        value = self._order(row)
                        if not start <= _day(value["created_at"], self.ORDERS_PATH) <= end:
                            continue
                    elif source == "sales":
                        value = self._sale(row)
                        if not start <= _day(value["date"], self.SALES_PATH) <= end:
                            continue
                    elif source == "stocks":
                        value = self._stock(row)
                    else:
                        value = self._product(row, card=True)
                    valid.append(value)
                if source == "finance":
                    snapshot.transactions = valid
                    if any(row["incomplete"] for row in valid):
                        self._warnings.append(
                            "Финансы WB содержат неполные поля или неоднозначные корректировки; "
                            "точная чистая прибыль не рассчитывается."
                        )
                    if any(row["items"] and row["accruals_for_sale"] == 0 for row in valid):
                        self._warnings.append(
                            "В WB есть продажи/возвраты с нулевой выручкой и ненулевым количеством. "
                            "Себестоимость этих операций требует отдельного расчёта; прибыль недоступна."
                        )
                elif source == "orders":
                    snapshot.postings = valid
                elif source == "sales":
                    snapshot.sales = valid
                    snapshot.sales_available = True
                elif source == "stocks":
                    snapshot.stocks = valid
                else:
                    products.update((row["sku"], row) for row in valid)
                snapshot.sources[source] = True
                if source != "products":
                    for row in result:
                        if _integer(row.get("nmId"), positive=True) is not None:
                            product = self._product(row)
                            products.setdefault(product["sku"], product)
            except WildberriesAPIError as error:
                self._warnings.append(f"{label}: {error}")
        snapshot.products = list(products.values())
        self._warnings.append(
            "WB: заказы и продажи — предварительная статистика с хранением до 90 дней; "
            "не все заказы с отложенной оплатой включены. Выкупы и возвраты считаются по дате "
            "события, не по статусу заказа. Цены статистики могут уточняться в течение суток; "
            "для денег используется отдельный финансовый отчёт. Отмены — текущие статусы "
            "заказов выбранного периода, а не состояние на дату его окончания."
        )
        if snapshot.sources.get("finance"):
            self._warnings.append(
                "Финансы WB — по дате операций ежедневного отчёта; отчётные суммы могут "
                "появляться с задержкой. Комиссия WB в источнике указана без НДС; "
                "прочие комиссии/НДС/эквайринг сверены с суммой «К перечислению»."
            )
        if snapshot.sources.get("stocks"):
            self._warnings.append(
                "Остатки WB — текущие склады WB, без товаров в пути к покупателю/обратно "
                "и без остатков FBS у продавца. Резервы источник не раскрывает."
            )
        snapshot.warnings = list(dict.fromkeys(self._warnings))
        return snapshot
