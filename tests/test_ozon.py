import json
from datetime import date
from decimal import Decimal

import httpx
import pytest

from ozon_bot.ozon import OzonAPIError, OzonClient, _month_windows


def money(amount, currency="RUB"):
    return {"amount": str(amount), "currency": currency}


def posting(number, *, day="2026-01-31", sku=111):
    return {
        "posting_number": number,
        "created_at": f"{day}T10:00:00Z",
        "in_process_at": f"{day}T10:00:00Z",
        "status": "delivered",
        "products": [
            {
                "sku": sku,
                "offer_id": f"item-{sku}",
                "quantity": 2,
                "price": money("500.25"),
                "digital_codes": ["private-code"],
            }
        ],
        "customer": {"phone": "private-phone", "address": "private-address"},
        "addressee": {"name": "private-name"},
        "tracking_number": "private-tracking",
        "legal_info": {"company_name": "private-company"},
    }


def empty_response(path):
    if path == "/v1/finance/accrual/types":
        return {"accrual_types": []}
    if path == "/v1/finance/accrual/by-day":
        return {"accruals": [], "last_id": ""}
    if path == "/v3/product/list":
        return {"result": {"items": [], "total_items": 0, "last_id": ""}}
    if path in {"/v3/posting/fbo/list", "/v4/posting/fbs/list"}:
        return {"postings": [], "has_next": False, "cursor": ""}
    if path == "/v4/product/info/stocks":
        return {"items": [], "total_items": 0, "cursor": ""}
    if path == "/v1/description-category/tree":
        return {"result": []}
    raise AssertionError(f"Unexpected endpoint: {path}")


@pytest.mark.asyncio
async def test_current_endpoints_full_pagination_money_and_personal_data_minimization():
    requests = []

    async def handler(request):
        path = request.url.path
        payload = json.loads(request.content)
        requests.append((path, payload))
        assert request.headers["Client-Id"] == "123"
        assert request.headers["Api-Key"] == "dummy-private-key"
        if path == "/v1/finance/accrual/types":
            return httpx.Response(
                200,
                json={
                    "accrual_types": [
                        {"id": 1, "name": "Логистика"},
                        {"id": 2, "name": "Реклама"},
                    ]
                },
            )
        if path == "/v1/finance/accrual/by-day":
            assert payload["last_id"] == ""
            if payload["date"] == "2026-01-31":
                accrual = {
                    "accrual_id": 1,
                    "date": payload["date"],
                    "accrued_category": "POSTING",
                    "unit_number": "fbo-1",
                    "total_amount": money("849.75"),
                    "posting": {
                        "delivery_schema": "fbo",
                        "products": [
                            {
                                "sku": 111,
                                "commission": {
                                    "sale_amount": money("1000.50"),
                                    "commission": money("-100.25"),
                                },
                                "delivery": {
                                    "total_accrued": money("-50.50"),
                                    "services": [
                                        {"type_id": 1, "accrued": money("-50.50")},
                                    ],
                                },
                            }
                        ],
                    },
                }
            else:
                accrual = {
                    "accrual_id": 2,
                    "date": payload["date"],
                    "accrued_category": "NON_ITEM",
                    "unit_number": "ad-contract",
                    "total_amount": money("-10.25"),
                    "non_item_fee": {"type_id": 2, "accrued": money("-10.25")},
                }
            return httpx.Response(200, json={"accruals": [accrual], "last_id": ""})
        if path == "/v3/product/list":
            product_id = 1 if not payload["last_id"] else 2
            return httpx.Response(
                200,
                json={
                    "result": {
                        "items": [{"product_id": product_id}],
                        "total_items": 2,
                        "last_id": "product-cursor" if product_id == 1 else "",
                    }
                },
            )
        if path == "/v3/product/info/list":
            assert payload == {"product_id": ["1", "2"]}
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": 1,
                            "name": "Первый товар",
                            "sources": [{"sku": 111}],
                            "description_category_id": 101,
                            "type_id": 1001,
                        },
                        {
                            "id": 2,
                            "name": "Второй товар",
                            "sources": [{"sku": 222}],
                            "description_category_id": 100,
                            "type_id": 999,
                        },
                    ]
                },
            )
        if path == "/v1/description-category/tree":
            assert payload == {"language": "DEFAULT"}
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "description_category_id": 100,
                            "category_name": "Канцелярские товары",
                            "children": [
                                {
                                    "description_category_id": 101,
                                    "category_name": "Печати и штампы",
                                    "children": [
                                        {
                                            "type_id": 1001,
                                            "type_name": "Пистолет-маркиратор",
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            )
        if path == "/v1/product/info/description":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "id": payload["product_id"],
                        "description": "Настоящее описание карточки",
                    }
                },
            )
        if path in {"/v3/posting/fbo/list", "/v4/posting/fbs/list"}:
            assert payload["limit"] <= 100
            assert "offset" not in payload
            since = payload["filter"]["since"]
            to = payload["filter"]["to"]
            assert since[:10] == to[:10]  # The two selected dates are in different months.
            if path == "/v3/posting/fbo/list" and since.startswith("2026-01-31"):
                number = "fbo-1" if not payload["cursor"] else "fbo-2"
                return httpx.Response(
                    200,
                    json={
                        "postings": [posting(number)],
                        "has_next": number == "fbo-1",
                        "cursor": "fbo-cursor" if number == "fbo-1" else "",
                    },
                )
            if path == "/v4/posting/fbs/list" and since.startswith("2026-02-01"):
                return httpx.Response(
                    200,
                    json={
                        "postings": [posting("fbs-1", day="2026-02-01", sku=222)],
                        "has_next": False,
                        "cursor": "",
                    },
                )
            return httpx.Response(200, json=empty_response(path))
        if path == "/v4/product/info/stocks":
            product_id = 1 if not payload["cursor"] else 2
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"product_id": product_id, "stocks": [{"present": 5, "reserved": 1}]}
                    ],
                    "total_items": 2,
                    "cursor": "stock-cursor" if product_id == 1 else "",
                },
            )
        raise AssertionError(path)

    client = OzonClient("123", "dummy-private-key", transport=httpx.MockTransport(handler))
    async with client:
        snapshot = await client.load(date(2026, 1, 31), date(2026, 2, 1))
    assert len(snapshot.transactions) == 2
    assert snapshot.transactions[0]["accruals_for_sale"] == "1000.50"
    assert snapshot.transactions[0]["sale_commission"] == "-100.25"
    assert snapshot.transactions[0]["posting"]["posting_number"] == "fbo-1"
    assert snapshot.transactions[0]["services"] == [
        {"name": "Логистика", "type_id": 1, "price": "-50.50"},
    ]  # Delivery's total is not added a second time.
    assert sum(Decimal(item["amount"]) for item in snapshot.transactions) == Decimal("839.50")
    assert {item["product_id"] for item in snapshot.products} == {1, 2}
    assert snapshot.products[0]["category"] == "Печати и штампы"
    assert snapshot.products[0]["category_type"] == "Пистолет-маркиратор"
    assert snapshot.products[0]["description_category_id"] == 101
    assert snapshot.products[1]["category"] == "Канцелярские товары"
    assert all(item["description"] for item in snapshot.products)
    assert len(snapshot.postings) == 3
    assert len(snapshot.stocks) == 2
    assert snapshot.postings[0]["products"][0]["price"] == "500.25"
    assert "private-" not in json.dumps(snapshot.postings)
    assert snapshot.daily == []
    assert client._headers == {}
    assert client._http is None
    assert not any(
        path in {"/v3/finance/transaction/list", "/v2/posting/fbo/list", "/v3/posting/fbs/list"}
        for path, _ in requests
    )


@pytest.mark.asyncio
async def test_finance_day_cursor_preserves_date_and_never_uses_missing_page_size():
    calls = []

    async def handler(request):
        if request.url.path.endswith("/types"):
            return httpx.Response(200, json={"accrual_types": []})
        payload = json.loads(request.content)
        calls.append(payload)
        number = len(calls)
        return httpx.Response(
            200,
            json={
                "accruals": [
                    {
                        "accrual_id": number,
                        "date": payload["date"],
                        "accrued_category": "NON_ITEM",
                        "total_amount": money(0),
                    }
                ],
                "last_id": "next" if number == 1 else "",
            },
        )

    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        rows = await client._finance(date(2026, 1, 1), date(2026, 1, 1))
    assert len(rows) == 2
    assert calls == [
        {"date": "2026-01-01", "last_id": ""},
        {"date": "2026-01-01", "last_id": "next"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_status", [429, 503])
async def test_transient_status_retries_then_validates_empty_store(failure_status):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(failure_status, headers={"Retry-After": "0"})
        return httpx.Response(200, json=empty_response(request.url.path))

    async with OzonClient(
        "123", "key", transport=httpx.MockTransport(handler), retry_base_delay=0
    ) as client:
        await client.validate()
    assert calls == 3


@pytest.mark.asyncio
async def test_authentication_error_never_discloses_body_or_credentials():
    def handler(request):
        return httpx.Response(403, json={"message": "dummy-private-key server-private-content"})

    async with OzonClient(
        "123", "dummy-private-key", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(OzonAPIError) as captured:
            await client.validate()
    assert captured.value.code == "authentication"
    assert captured.value.status_code == 403
    assert "dummy-private-key" not in str(captured.value)
    assert "server-private-content" not in str(captured.value)


@pytest.mark.asyncio
async def test_transport_exception_is_sanitized_and_retry_is_bounded():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("dummy-private-key private-network-message", request=request)

    async with OzonClient(
        "123",
        "dummy-private-key",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0,
        max_retries=1,
    ) as client:
        with pytest.raises(OzonAPIError) as captured:
            await client.validate()
    assert captured.value.code == "network"
    assert "private" not in str(captured.value)
    assert calls == 2


@pytest.mark.asyncio
async def test_redirect_is_not_followed_with_seller_key():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://untrusted.example/"})

    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OzonAPIError) as captured:
            await client.validate()
    assert captured.value.status_code == 307
    assert calls == ["https://api-seller.ozon.ru/v3/product/list"]


@pytest.mark.asyncio
async def test_bounded_finance_pagination_fails_instead_of_truncating():
    def handler(request):
        if request.url.path.endswith("/types"):
            return httpx.Response(200, json={"accrual_types": []})
        return httpx.Response(
            200,
            json={
                "accruals": [
                    {
                        "accrual_id": 1,
                        "date": "2026-01-01",
                        "total_amount": money(0),
                    }
                ],
                "last_id": "more",
            },
        )

    async with OzonClient(
        "123", "key", transport=httpx.MockTransport(handler), max_pages=1
    ) as client:
        with pytest.raises(OzonAPIError) as captured:
            await client._finance(date(2026, 1, 1), date(2026, 1, 1))
    assert captured.value.code == "pagination"
    assert "неполна" in str(captured.value)


@pytest.mark.asyncio
async def test_unavailable_optional_source_is_explicit_warning():
    def handler(request):
        if request.url.path == "/v4/posting/fbs/list":
            return httpx.Response(500, text="private upstream response")
        return httpx.Response(200, json=empty_response(request.url.path))

    async with OzonClient(
        "123", "key", transport=httpx.MockTransport(handler), max_retries=0
    ) as client:
        snapshot = await client.load(date(2026, 1, 1), date(2026, 1, 1))
    assert snapshot.postings == []
    assert any("Отправления FBS: данные недоступны" in warning for warning in snapshot.warnings)
    assert not any("private upstream response" in warning for warning in snapshot.warnings)


@pytest.mark.asyncio
async def test_financial_source_failure_does_not_produce_zero_profit_snapshot():
    def handler(request):
        if request.url.path == "/v1/finance/accrual/by-day":
            return httpx.Response(403, json={"error": "private-body"})
        return httpx.Response(200, json=empty_response(request.url.path))

    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OzonAPIError, match="Ozon отказал"):
            await client.load(date(2026, 1, 1), date(2026, 1, 1))


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "invalid", None])
def test_invalid_money_is_rejected(amount):
    client = OzonClient("123", "key")
    with pytest.raises(OzonAPIError) as captured:
        client._normalize_accrual({"total_amount": money(amount)}, {})
    assert captured.value.code == "schema"


def test_mixed_currency_never_summed_as_rubles():
    client = OzonClient("123", "key")
    with pytest.raises(OzonAPIError) as captured:
        client._normalize_accrual({"total_amount": money("100", "CNY")}, {})
    assert captured.value.code == "currency"


def test_month_slicing_has_no_overlapping_or_missing_dates():
    assert list(_month_windows(date(2024, 2, 28), date(2024, 3, 2))) == [
        (date(2024, 2, 28), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]
    assert list(_month_windows(date(2025, 12, 31), date(2026, 1, 1))) == [
        (date(2025, 12, 31), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 1)),
    ]


@pytest.mark.asyncio
async def test_period_before_current_api_history_is_rejected_before_requests():
    def handler(request):
        raise AssertionError("No HTTP requests should be made")

    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="с 1 января 2022"):
            await client.load(date(2021, 12, 31), date(2022, 1, 1))


@pytest.mark.parametrize("api_key", ["", "ключ", "key\tsecret", "key\nsecret", "key\x00secret"])
def test_malformed_keys_fail_safely_before_http_headers(api_key):
    with pytest.raises(ValueError, match="корректный API-ключ"):
        OzonClient("123", api_key)


@pytest.mark.asyncio
async def test_duplicate_catalog_rows_cannot_satisfy_total_count():
    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "result": {
                    "items": [{"product_id": 1}],
                    "total_items": 2,
                    "last_id": "different" if payload["last_id"] else "first",
                }
            },
        )

    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OzonAPIError) as captured:
            await client._products()
    assert captured.value.code == "pagination"


@pytest.mark.asyncio
async def test_category_dictionary_failure_preserves_catalog_and_explicit_ids():
    def handler(request):
        assert request.url.path == "/v1/description-category/tree"
        return httpx.Response(403, text="private upstream body")

    products = [{"product_id": 1, "description_category_id": 101, "type_id": 1001}]
    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        await client._enrich_categories(products)
        assert products[0]["category"] == "Категория 101"
        assert products[0]["description_category_id"] == 101
        assert any("Дерево категорий Ozon недоступно" in warning for warning in client._warnings)
        assert not any("private upstream body" in warning for warning in client._warnings)


@pytest.mark.asyncio
async def test_unknown_category_uses_id_and_warns_even_with_valid_dictionary():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "description_category_id": 100,
                        "category_name": "Канцелярия",
                        "children": [],
                    }
                ]
            },
        )

    products = [{"product_id": 1, "description_category_id": 999, "type_id": 1001}]
    async with OzonClient("123", "key", transport=httpx.MockTransport(handler)) as client:
        await client._enrich_categories(products)
        assert products[0]["category"] == "Категория 999"
        assert any(
            "Для 1 товаров название категории не найдено" in warning for warning in client._warnings
        )


def test_category_tree_cycles_and_excessive_depth_are_rejected():
    client = OzonClient("123", "key")
    node = {"description_category_id": 100, "category_name": "Канцелярия", "children": []}
    node["children"].append(node)
    with pytest.raises(OzonAPIError) as captured:
        client._index_categories([node])
    assert captured.value.code == "pagination"
    deep = {"description_category_id": 100, "category_name": "Канцелярия", "children": []}
    for _ in range(34):
        deep = {"children": [deep]}
    with pytest.raises(OzonAPIError) as captured:
        client._index_categories([deep])
    assert captured.value.code == "pagination"
