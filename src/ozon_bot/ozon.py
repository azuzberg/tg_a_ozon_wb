"""Read-only Ozon Seller API client; never includes API responses in errors.

The endpoints used here are versioned explicitly. Ozon Performance API, search
positions and competitors are separate data sources and are not fabricated.
"""

from __future__ import annotations

import asyncio
import calendar
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Self

import httpx

from .domain import Snapshot


class OzonAPIError(Exception):
    """A safe error that can be shown to a Telegram user without exposing keys."""

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


def _month_windows(start: date, end: date):
    current = start
    while current <= end:
        month_end = date(
            current.year, current.month, calendar.monthrange(current.year, current.month)[1]
        )
        last = min(month_end, end)
        yield current, last
        if last == end:
            return
        current = date(last.year + (last.month == 12), last.month % 12 + 1, 1)


def _utc_bounds(start: date, end: date) -> tuple[str, str]:
    return f"{start.isoformat()}T00:00:00Z", f"{end.isoformat()}T23:59:59.999999Z"


class OzonClient:
    """Async read-only client with retry and explicit completeness limits.

    ``transport`` allows offline tests with ``httpx.MockTransport``. A client
    must be used as an async context manager; its credentials are cleared from
    HTTP headers when that context closes.
    """

    BASE_URL = "https://api-seller.ozon.ru"

    def __init__(
        self,
        client_id: str,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
        timeout: float = 40.0,
        max_retries: int = 3,
        max_pages: int = 100,
        page_size: int = 1000,
        max_records: int = 100_000,
        retry_base_delay: float = 0.5,
    ) -> None:
        client_id, api_key = str(client_id).strip(), api_key.strip()
        if (
            len(client_id) > 20
            or not client_id.isascii()
            or not client_id.isdigit()
            or int(client_id) <= 0
        ):
            raise ValueError("Client-Id Ozon должен быть положительным числом.")
        if (
            not api_key
            or not api_key.isascii()
            or any(ord(character) <= 32 or ord(character) == 127 for character in api_key)
        ):
            raise ValueError("Укажите корректный API-ключ Ozon.")
        if not 1 <= page_size <= 1000 or max_pages < 1 or max_records < 1:
            raise ValueError("Недопустимые ограничения пагинации.")
        if max_retries < 0 or retry_base_delay < 0:
            raise ValueError("Недопустимые параметры повторных запросов.")
        self._headers = {"Client-Id": client_id, "Api-Key": api_key}
        self._transport = transport
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_pages = max_pages
        self._page_size = page_size
        self._max_records = max_records
        self._retry_base_delay = retry_base_delay
        self._http: httpx.AsyncClient | None = None
        self._warnings: list[str] = []

    async def __aenter__(self) -> Self:
        if self._http is not None:
            raise RuntimeError("OzonClient уже открыт.")
        if not self._headers:
            raise RuntimeError("Закрытый OzonClient нельзя использовать повторно.")
        self._http = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self._headers,
            timeout=self._timeout,
            transport=self._transport,
            follow_redirects=False,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._headers.clear()
        if self._http is not None:
            self._http.headers.pop("Api-Key", None)
            self._http.headers.pop("Client-Id", None)
            await self._http.aclose()
            self._http = None

    async def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("Используйте OzonClient через async with.")
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(endpoint, json=payload)
            except httpx.RequestError:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_base_delay * 2**attempt)
                    continue
                raise OzonAPIError(
                    "Не удалось связаться с Ozon. Повторите запрос позже.",
                    endpoint=endpoint,
                    code="network",
                ) from None
            if (
                response.status_code == 429 or 500 <= response.status_code < 600
            ) and attempt < self._max_retries:
                delay = self._retry_base_delay * 2**attempt
                try:
                    retry_after = float(response.headers.get("Retry-After", "0"))
                    # Prevent an upstream header from blocking a worker indefinitely.
                    delay = max(delay, min(max(retry_after, 0.0), 10.0))
                except ValueError:
                    pass
                await asyncio.sleep(delay)
                continue
            if response.status_code in {401, 403}:
                raise OzonAPIError(
                    "Ozon отказал в доступе. Проверьте Client-Id, API-ключ и права ключа.",
                    endpoint=endpoint,
                    status_code=response.status_code,
                    code="authentication",
                )
            if not response.is_success:
                raise OzonAPIError(
                    f"Ozon не выполнил запрос {endpoint} (HTTP {response.status_code}).",
                    endpoint=endpoint,
                    status_code=response.status_code,
                    code="rate_limit" if response.status_code == 429 else "upstream",
                )
            try:
                result = response.json()
            except ValueError:
                raise self._schema_error(endpoint) from None
            if not isinstance(result, dict):
                raise self._schema_error(endpoint)
            return result
        raise AssertionError("Unreachable retry state")

    @staticmethod
    def _schema_error(endpoint: str) -> OzonAPIError:
        return OzonAPIError(
            f"Ozon вернул неожиданный формат данных для {endpoint}.",
            endpoint=endpoint,
            code="schema",
        )

    @staticmethod
    def _pagination_error(endpoint: str) -> OzonAPIError:
        return OzonAPIError(
            f"Загрузка {endpoint} неполна: достигнут лимит или нарушена пагинация. "
            "Выберите меньший период; неполные данные не используются.",
            endpoint=endpoint,
            code="pagination",
        )

    def _rows(self, value: Any, endpoint: str) -> list[dict[str, Any]]:
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise self._schema_error(endpoint)
        return value

    def _check_count(self, rows: list[dict[str, Any]], endpoint: str) -> None:
        if len(rows) > self._max_records:
            raise self._pagination_error(endpoint)

    async def check_credentials(self) -> None:
        endpoint = "/v3/product/list"
        response = await self._request(
            endpoint, {"filter": {"visibility": "ALL"}, "last_id": "", "limit": 1}
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise self._schema_error(endpoint)
        self._rows(result.get("items"), endpoint)

    async def validate(self) -> None:
        """Compatibility name used by the Telegram login flow."""
        await self.check_credentials()

    async def product_description(self, product_id: int) -> dict[str, Any]:
        """Fetch real card content on demand instead of requesting every card."""
        endpoint = "/v1/product/info/description"
        response = await self._request(endpoint, {"product_id": product_id})
        result = response.get("result")
        if not isinstance(result, dict):
            raise self._schema_error(endpoint)
        return result

    def _index_categories(
        self, nodes: list[dict[str, Any]]
    ) -> tuple[dict[int, str], dict[tuple[int, int], tuple[str, str]]]:
        """Walk the documented category tree without unbounded recursion."""
        endpoint = "/v1/description-category/tree"
        categories: dict[int, str] = {}
        product_types: dict[tuple[int, int], tuple[str, str]] = {}
        stack = [(node, 0, 0, "") for node in nodes]
        visited: set[int] = set()
        while stack:
            node, depth, inherited_id, inherited_name = stack.pop()
            if depth > 32 or len(visited) >= 100_000 or id(node) in visited:
                raise self._pagination_error(endpoint)
            visited.add(id(node))
            category_id = node.get("description_category_id") or inherited_id
            category_name = node.get("category_name") or inherited_name
            if (
                not isinstance(category_id, int)
                or isinstance(category_id, bool)
                or category_id < 0
                or not isinstance(category_name, str)
                or (category_id > 0 and not category_name.strip())
            ):
                raise self._schema_error(endpoint)
            if category_id:
                category_name = category_name.strip()
                if category_id in categories and categories[category_id] != category_name:
                    raise self._schema_error(endpoint)
                categories[category_id] = category_name
            type_id = node.get("type_id")
            if type_id:
                type_name = node.get("type_name")
                if (
                    not isinstance(type_id, int)
                    or isinstance(type_id, bool)
                    or type_id < 0
                    or not category_id
                    or not isinstance(type_name, str)
                    or not type_name.strip()
                ):
                    raise self._schema_error(endpoint)
                pair = (category_id, type_id)
                value = (category_name, type_name.strip())
                if pair in product_types and product_types[pair] != value:
                    raise self._schema_error(endpoint)
                product_types[pair] = value
            children = self._rows(node.get("children", []), endpoint)
            if len(stack) + len(visited) + len(children) > 100_000:
                raise self._pagination_error(endpoint)
            stack.extend((child, depth + 1, category_id, category_name) for child in children)
        return categories, product_types

    async def _enrich_categories(self, products: list[dict[str, Any]]) -> None:
        if not products:
            return
        endpoint = "/v1/description-category/tree"
        categories: dict[int, str] = {}
        product_types: dict[tuple[int, int], tuple[str, str]] = {}
        dictionary_unavailable = False
        try:
            response = await self._request(endpoint, {"language": "DEFAULT"})
            categories, product_types = self._index_categories(
                self._rows(response.get("result"), endpoint)
            )
        except OzonAPIError:
            dictionary_unavailable = True
            self._warnings.append(
                "Дерево категорий Ozon недоступно: вместо названий показаны "
                "идентификаторы категорий."
            )
        missing = 0
        for product in products:
            category_id = product.get("description_category_id", product.get("category_id"))
            type_id = product.get("type_id")
            typed_name = product_types.get((category_id, type_id))
            name = typed_name[0] if typed_name else categories.get(category_id)
            if name:
                product["category"] = name
                if typed_name:
                    product["category_type"] = typed_name[1]
            else:
                product["category"] = (
                    f"Категория {category_id}" if category_id else "Категория не указана"
                )
                missing += 1
        if missing and not dictionary_unavailable:
            self._warnings.append(
                f"Для {missing} товаров название категории не найдено в справочнике Ozon; "
                "показан её идентификатор либо отметка об отсутствии категории."
            )

    async def _products(self) -> list[dict[str, Any]]:
        endpoint = "/v3/product/list"
        listed: list[dict[str, Any]] = []
        last_id = ""
        seen: set[str] = set()
        seen_product_ids: set[str] = set()
        for _ in range(self._max_pages):
            response = await self._request(
                endpoint,
                {"filter": {"visibility": "ALL"}, "last_id": last_id, "limit": self._page_size},
            )
            result = response.get("result")
            if not isinstance(result, dict):
                raise self._schema_error(endpoint)
            batch = self._rows(result.get("items"), endpoint)
            for item in batch:
                product_id = str(item.get("product_id") or "")
                if not product_id or product_id in seen_product_ids:
                    raise self._pagination_error(endpoint)
                seen_product_ids.add(product_id)
            listed.extend(batch)
            self._check_count(listed, endpoint)
            total = result.get("total_items", result.get("total"))
            if isinstance(total, int) and total >= 0 and len(listed) >= total:
                break
            if not batch:
                if isinstance(total, int) and len(listed) < total:
                    raise self._pagination_error(endpoint)
                break
            cursor = str(result.get("last_id") or "")
            if not cursor or cursor in seen:
                # A short final page is complete even if no total/cursor is provided.
                if len(batch) < self._page_size and not isinstance(total, int):
                    break
                raise self._pagination_error(endpoint)
            seen.add(cursor)
            last_id = cursor
        else:
            raise self._pagination_error(endpoint)

        products: list[dict[str, Any]] = []
        info_endpoint = "/v3/product/info/list"
        # Ozon accepts at most 1000 IDs in an info request.
        for offset in range(0, len(listed), 1000):
            batch = listed[offset : offset + 1000]
            ids = [str(item["product_id"]) for item in batch if item.get("product_id")]
            if len(ids) != len(batch):
                raise self._schema_error(endpoint)
            response = await self._request(info_endpoint, {"product_id": ids})
            items = self._rows(response.get("items"), info_endpoint)
            returned = {str(item.get("id", item.get("product_id", ""))) for item in items}
            if returned != set(ids) or len(returned) != len(items):
                raise OzonAPIError(
                    "Ozon не вернул информацию обо всех товарах. Повторите обновление.",
                    endpoint=info_endpoint,
                    code="pagination",
                )
            for item in items:
                # Keep documented raw fields and a stable identifier for analytics.
                products.append({**item, "product_id": item.get("id", item.get("product_id"))})
        await self._enrich_categories(products)
        if len(products) <= 200:
            semaphore = asyncio.Semaphore(5)

            async def enrich_description(product: dict[str, Any]) -> bool:
                async with semaphore:
                    try:
                        content = await self.product_description(int(product["product_id"]))
                    except OzonAPIError:
                        return False
                    product["description"] = content.get("description", "")
                    return True

            description_results = await asyncio.gather(
                *(enrich_description(product) for product in products)
            )
            unavailable = description_results.count(False)
            if unavailable:
                self._warnings.append(
                    f"Описание карточки недоступно для {unavailable} товаров; "
                    "SEO-анализ этих карточек ограничен названием и изображениями."
                )
        elif products:
            self._warnings.append(
                "В каталоге больше 200 товаров: автоматическая загрузка всех описаний "
                "отключена. Для полного SEO-анализa содержания загрузите CSV с описаниями."
            )
        return products

    def _money(self, value: Any, endpoint: str, *, required: bool = False) -> Decimal:
        if value is None and not required:
            return Decimal(0)
        if not isinstance(value, dict) or "amount" not in value:
            raise self._schema_error(endpoint)
        if value.get("currency") != "RUB":
            raise OzonAPIError(
                "В начислениях Ozon найдена валюта, отличная от RUB. "
                "Финансы не суммируются без правил конвертации валют.",
                endpoint=endpoint,
                code="currency",
            )
        try:
            amount = Decimal(str(value["amount"]))
        except (InvalidOperation, ValueError):
            raise self._schema_error(endpoint) from None
        if not amount.is_finite():
            raise self._schema_error(endpoint)
        return amount

    def _normalize_accrual(
        self, row: dict[str, Any], accrual_types: dict[int, str]
    ) -> dict[str, Any]:
        endpoint = "/v1/finance/accrual/by-day"
        amount = self._money(row.get("total_amount"), endpoint, required=True)
        posting = row.get("posting") or {}
        if not isinstance(posting, dict):
            raise self._schema_error(endpoint)
        products = self._rows(posting.get("products", []), endpoint)
        revenue = Decimal(0)
        commission = Decimal(0)
        services: list[dict[str, Any]] = []
        skus: set[int] = set()

        def append_fee(fee: dict[str, Any]) -> None:
            type_id = fee.get("type_id")
            if not isinstance(type_id, int):
                raise self._schema_error(endpoint)
            price = self._money(fee.get("accrued"), endpoint, required=True)
            services.append(
                {
                    "name": accrual_types.get(type_id, f"Начисление Ozon #{type_id}"),
                    "type_id": type_id,
                    "price": str(price),
                }
            )

        for product in products:
            if isinstance(product.get("sku"), int):
                skus.add(product["sku"])
            commission_data = product.get("commission") or {}
            if not isinstance(commission_data, dict):
                raise self._schema_error(endpoint)
            revenue += self._money(commission_data.get("sale_amount"), endpoint)
            commission += self._money(commission_data.get("commission"), endpoint)
            delivery = product.get("delivery") or {}
            if not isinstance(delivery, dict):
                raise self._schema_error(endpoint)
            delivery_services = self._rows(delivery.get("services", []), endpoint)
            for service in delivery_services:
                append_fee(service)
            if not delivery_services and delivery.get("total_accrued") is not None:
                services.append(
                    {
                        "name": "Доставка (итог)",
                        "price": str(self._money(delivery["total_accrued"], endpoint)),
                    }
                )
        item_fees = row.get("item_fees") or {}
        if not isinstance(item_fees, dict):
            raise self._schema_error(endpoint)
        for item in self._rows(item_fees.get("fees", []), endpoint):
            if isinstance(item.get("sku"), int):
                skus.add(item["sku"])
            for fee in self._rows(item.get("fees", []), endpoint):
                append_fee(fee)
        if row.get("non_item_fee") is not None:
            if not isinstance(row["non_item_fee"], dict):
                raise self._schema_error(endpoint)
            append_fee(row["non_item_fee"])
        container_fees = row.get("container_fees") or {}
        if not isinstance(container_fees, dict):
            raise self._schema_error(endpoint)
        for fee in self._rows(container_fees.get("fees", []), endpoint):
            append_fee(fee)
        operation_name = "; ".join(service["name"] for service in services)
        if revenue < 0:
            operation_name = "Возврат продаж; " + operation_name
        return {
            **row,
            "source": "finance_accrual_by_day",
            "operation_id": row.get("accrual_id"),
            "operation_date": row.get("date"),
            "operation_type": row.get("accrued_category"),
            "operation_type_name": operation_name,
            "amount": str(amount),
            "currency_code": "RUB",
            "accruals_for_sale": str(revenue),
            "sale_commission": str(commission),
            "posting": {**posting, "posting_number": row.get("unit_number", "")},
            "items": [{"sku": sku} for sku in sorted(skus)],
            "services": services,
        }

    async def _finance(self, start: date, end: date) -> list[dict[str, Any]]:
        endpoint = "/v1/finance/accrual/by-day"
        type_endpoint = "/v1/finance/accrual/types"
        type_response = await self._request(type_endpoint, {})
        type_rows = self._rows(type_response.get("accrual_types"), type_endpoint)
        accrual_types: dict[int, str] = {}
        for type_row in type_rows:
            if not isinstance(type_row.get("id"), int) or not isinstance(type_row.get("name"), str):
                raise self._schema_error(type_endpoint)
            accrual_types[type_row["id"]] = type_row["name"]
        operations: list[dict[str, Any]] = []
        current = start
        operation_ids: set[int] = set()
        while current <= end:
            last_id = ""
            seen: set[str] = set()
            for _ in range(self._max_pages):
                response = await self._request(
                    endpoint,
                    {"date": current.isoformat(), "last_id": last_id},
                )
                batch = self._rows(response.get("accruals"), endpoint)
                for operation in batch:
                    operation_id = operation.get("accrual_id")
                    if not isinstance(operation_id, int) or operation_id in operation_ids:
                        raise self._pagination_error(endpoint)
                    if operation.get("date") != current.isoformat():
                        raise self._schema_error(endpoint)
                    operation_ids.add(operation_id)
                    operations.append(self._normalize_accrual(operation, accrual_types))
                self._check_count(operations, endpoint)
                cursor = response.get("last_id")
                if not isinstance(cursor, str):
                    raise self._schema_error(endpoint)
                if not cursor:
                    break
                if not batch or cursor in seen:
                    raise self._pagination_error(endpoint)
                seen.add(cursor)
                last_id = cursor
            else:
                raise self._pagination_error(endpoint)
            current += timedelta(days=1)
        return operations

    async def _postings(self, start: date, end: date, scheme: str) -> list[dict[str, Any]]:
        endpoint = "/v3/posting/fbo/list" if scheme == "fbo" else "/v4/posting/fbs/list"
        postings: list[dict[str, Any]] = []
        seen_postings: set[str] = set()
        for first, last in _month_windows(start, end):
            since, to = _utc_bounds(first, last)
            cursor = ""
            seen_cursors: set[str] = set()
            for _ in range(self._max_pages):
                response = await self._request(
                    endpoint,
                    {
                        "sort_dir": "ASC",
                        "filter": {"since": since, "to": to},
                        "limit": min(self._page_size, 100),
                        "cursor": cursor,
                        "with": {"analytics_data": True, "financial_data": True},
                    },
                )
                if not isinstance(response.get("has_next"), bool):
                    raise self._schema_error(endpoint)
                batch = self._rows(response.get("postings"), endpoint)
                has_next = response["has_next"]
                if not batch and has_next:
                    raise self._pagination_error(endpoint)
                for posting in batch:
                    posting_number = str(posting.get("posting_number") or "")
                    if not posting_number or posting_number in seen_postings:
                        raise self._pagination_error(endpoint)
                    seen_postings.add(posting_number)
                    # Customer addresses and tracking details have no analytical use.
                    cleaned = {
                        key: value
                        for key, value in posting.items()
                        if key
                        not in {
                            "customer",
                            "addressee",
                            "barcodes",
                            "tracking_number",
                            "legal_info",
                        }
                    }
                    normalized_products: list[dict[str, Any]] = []
                    for product in self._rows(cleaned.get("products", []), endpoint):
                        price = product.get("price")
                        self._money(price, endpoint, required=True)
                        normalized_products.append(
                            {
                                **{
                                    key: value
                                    for key, value in product.items()
                                    if key not in {"digital_codes", "imei"}
                                },
                                "price_money": price,
                                "price": price["amount"],
                                "currency_code": price["currency"],
                            }
                        )
                    cleaned["products"] = normalized_products
                    postings.append({**cleaned, "delivery_schema": scheme.upper()})
                self._check_count(postings, endpoint)
                if not has_next:
                    break
                next_cursor = response.get("cursor")
                if (
                    not isinstance(next_cursor, str)
                    or not next_cursor
                    or next_cursor in seen_cursors
                ):
                    raise self._pagination_error(endpoint)
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            else:
                raise self._pagination_error(endpoint)
        return postings

    async def _stocks(self) -> list[dict[str, Any]]:
        endpoint = "/v4/product/info/stocks"
        rows: list[dict[str, Any]] = []
        cursor = ""
        seen: set[str] = set()
        seen_product_ids: set[str] = set()
        for _ in range(self._max_pages):
            response = await self._request(
                endpoint,
                {"cursor": cursor, "filter": {"visibility": "ALL"}, "limit": self._page_size},
            )
            batch = self._rows(response.get("items"), endpoint)
            for item in batch:
                product_id = str(item.get("product_id") or "")
                if not product_id or product_id in seen_product_ids:
                    raise self._pagination_error(endpoint)
                seen_product_ids.add(product_id)
            rows.extend(batch)
            self._check_count(rows, endpoint)
            total = response.get("total_items", response.get("total"))
            if isinstance(total, int) and total >= 0 and len(rows) >= total:
                return rows
            if not batch:
                if isinstance(total, int) and len(rows) < total:
                    raise self._pagination_error(endpoint)
                return rows
            next_cursor = str(response.get("cursor") or "")
            if not next_cursor or next_cursor in seen:
                if len(batch) < self._page_size and not isinstance(total, int):
                    return rows
                raise self._pagination_error(endpoint)
            seen.add(next_cursor)
            cursor = next_cursor
        raise self._pagination_error(endpoint)

    async def load(self, period_start: date, period_end: date) -> Snapshot:
        """Load complete finance and catalog; annotate unavailable optional sections.

        Dates include both boundary days in UTC. Postings are split by calendar
        month; financial accruals are loaded by day. Transaction dates and
        order creation dates are different cohorts, which downstream reports
        must explain when comparing finance to orders.
        """
        if period_start > period_end:
            raise ValueError("Начало периода не может быть позже окончания.")
        if (period_end - period_start).days > 3660:
            raise ValueError("Выберите период не более 10 лет.")
        if period_start < date(2022, 1, 1):
            raise ValueError("API начислений Ozon предоставляет данные с 1 января 2022 года.")
        self._warnings.clear()
        results = await asyncio.gather(
            self._finance(period_start, period_end),
            self._products(),
            self._postings(period_start, period_end, "fbo"),
            self._postings(period_start, period_end, "fbs"),
            self._stocks(),
            return_exceptions=True,
        )
        # These sections are required for the advertised financial calculations.
        for result in results[:2]:
            if isinstance(result, BaseException):
                raise result
        warnings = [
            (
                "Продажи отражают заказы, созданные в выбранный период, и их текущие статусы; "
                "финансы отражают операции, проведённые в период. Это разные выборки."
            ),
            (
                "Ozon Seller API не предоставляет здесь поисковые позиции и данные конкурентов; "
                "для рекламы нужны отчёты или отдельный доступ к Performance API."
            ),
            (
                "Начисления загружены через API accrual/by-day. В этом API нет количества "
                "проданных единиц: себестоимость по отправлениям является оценкой и может "
                "отличаться от себестоимости операций начислений."
            ),
        ]
        warnings.extend(self._warnings)
        optional: list[list[dict[str, Any]]] = []
        for label, result in zip(("Отправления FBO", "Отправления FBS", "Остатки"), results[2:]):
            if isinstance(result, OzonAPIError):
                warnings.append(f"{label}: данные недоступны. {result}")
                optional.append([])
            elif isinstance(result, BaseException):
                # Programming errors are not disguised as missing marketplace data.
                raise result
            else:
                optional.append(result)
        return Snapshot(
            transactions=results[0],
            products=results[1],
            postings=optional[0] + optional[1],
            stocks=optional[2],
            daily=[],
            warnings=warnings,
        )
