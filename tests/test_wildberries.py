import json
from datetime import date

import httpx
import pytest

from ozon_bot.wildberries import WildberriesAPIError, WildberriesClient, WildberriesError

TODAY = date(2026, 10, 15)
START = date(2026, 10, 1)
END = date(2026, 10, 14)
TASK = "06e06887-9d9f-491f-b16a-bb1766fcb8d2"


def financial_row(**changes):
    row = {
        "rrdId": 1,
        "nmId": 123,
        "currency": "RUB",
        "docTypeName": "Продажа",
        "sellerOperName": "Продажа",
        "rrDate": "2026-10-10",
        "quantity": 2,
        "retailAmount": "1000.50",
        "forPay": "850.25",
        "ppvzSalesCommission": "100",
        "deliveryService": "50.25",
        "paidStorage": "0",
        "deduction": "0",
        "paidAcceptance": "0",
        "penalty": "0",
        "additionalPayment": "0",
        "rebillLogisticCost": "0",
        "cashbackAmount": "0",
        "cashbackDiscount": "0",
        "cashbackCommissionChange": "0",
        "paymentSchedule": "0",
        "installmentCofinancingAmount": "0",
        "agencyVat": "0",
        "vendorCode": "shirt-1",
        "subjectName": "Рубашки",
        "title": "Рубашка",
        "b2bCustomerTin": "private-tin",
        "ppvzSupplierName": "private-person",
        "ppvzOfficeName": "private-address",
    }
    row.update(changes)
    return row


def order_row(**changes):
    row = {
        "nmId": 123,
        "srid": "order-1",
        "supplierArticle": "shirt-1",
        "subject": "Рубашки",
        "date": "2026-10-10T11:00:00",
        "lastChangeDate": "2026-10-12T12:30:00.12345",
        "priceWithDisc": 500.25,
        "isCancel": False,
        "regionName": "private-region",
        "sticker": "private-sticker",
        "customer": {"name": "private-person"},
    }
    row.update(changes)
    return row


def sale_row(**changes):
    row = order_row(saleID="S1000")
    row.update(changes)
    return row


def stock_row(**changes):
    row = {
        "nmId": 123,
        "vendorCode": "shirt-1",
        "subjectName": "Рубашки",
        "warehouses": [
            {"warehouseName": "To the client", "quantity": 100},
            {"warehouseName": "From the client", "quantity": 30},
            {"warehouseName": "Total in warehouses", "quantity": 12},
            {"warehouseName": "Warehouse A", "quantity": 10},
            {"warehouseName": "Warehouse B", "quantity": 2},
        ],
    }
    row.update(changes)
    return row


def client(handler, **kwargs):
    return WildberriesClient(
        "private-token",
        transport=httpx.MockTransport(handler),
        today=TODAY,
        request_interval=0,
        poll_interval=0,
        retry_base_delay=0,
        **kwargs,
    )


def handler_for(*, finance=None, orders=None, sales=None, stocks=None, cards=None, denied=()):
    calls = []
    seen = set()

    async def handler(request):
        path = request.url.path
        calls.append(request)
        assert request.headers["Authorization"] == "private-token"
        assert request.url.scheme == "https"
        if request.url.host in denied:
            return httpx.Response(403, json={"detail": "private-token private-person"})
        if path == "/ping":
            return httpx.Response(200, json={"Status": "OK", "TS": "2026-10-15"})
        if path == WildberriesClient.FINANCE_PATH:
            payload = json.loads(request.content)
            assert payload["dateTo"].endswith("T23:59:59")
            assert payload["period"] == "daily"
            assert payload["fields"] and "b2bCustomerTin" not in payload["fields"]
            if not payload["rrdId"]:
                return httpx.Response(200, json=finance or [])
            return httpx.Response(204)
        if path in {WildberriesClient.ORDERS_PATH, WildberriesClient.SALES_PATH}:
            assert request.url.params["flag"] == "0"
            if path in seen:
                return httpx.Response(200, json=[])
            seen.add(path)
            rows = orders if path == WildberriesClient.ORDERS_PATH else sales
            return httpx.Response(200, json=rows or [])
        if path == WildberriesClient.CARDS_PATH:
            payload = json.loads(request.content)
            assert payload["settings"]["sort"] == {"ascending": True}
            return httpx.Response(
                200, json={"cards": cards or [], "cursor": {"total": len(cards or [])}}
            )
        if path == WildberriesClient.STOCKS_PATH:
            assert request.url.params["locale"] == "en"
            assert request.url.params["groupByNm"] == "true"
            return httpx.Response(200, json={"data": {"taskId": TASK}})
        if path.endswith("/status"):
            return httpx.Response(200, json={"data": {"status": "done"}})
        if path.endswith("/download"):
            return httpx.Response(200, json=stocks or [])
        raise AssertionError(f"Unexpected endpoint {path}")

    return handler, calls


async def test_real_sources_normalize_sales_finance_and_current_stocks_without_personal_data():
    handler, calls = handler_for(
        finance=[financial_row()],
        orders=[order_row(), order_row(srid="cancel", isCancel=True)],
        sales=[sale_row(), sale_row(saleID="R1000", date="2026-10-11T14:00:00")],
        stocks=[stock_row()],
        cards=[
            {
                "nmID": 123,
                "vendorCode": "shirt-1",
                "subjectName": "Рубашки",
                "title": "Хлопковая рубашка",
                "description": "Настоящее описание",
                "photos": [{"big": "https://images.example/1.webp"}],
                "documents": {"applicant": "private-person"},
            }
        ],
    )
    async with client(handler) as wb:
        await wb.validate()
        snapshot = await wb.load(START, END)
        assert "Authorization" not in wb._http.headers
    assert snapshot.marketplace == "wildberries"
    assert snapshot.sources == dict.fromkeys(
        ("finance", "orders", "sales", "products", "stocks"), True
    )
    assert snapshot.sales_available
    transaction = snapshot.transactions[0]
    assert transaction["accruals_for_sale"] == 1000.50
    assert transaction["amount"] == 800.00
    assert transaction["sale_commission"] == -100
    assert not transaction["incomplete"]
    assert transaction["items"] == [{"sku": 123, "offer_id": "shirt-1", "quantity": 2}]
    assert sum(service["price"] for service in transaction["services"]) == -100.50
    assert [row["status"] for row in snapshot.postings] == ["pending", "cancelled"]
    assert [row["status"] for row in snapshot.sales] == ["delivered", "returned"]
    assert snapshot.sales[1]["date"] == "2026-10-11"
    assert snapshot.sales[0]["quantity"] == 1  # WB guarantees one unit per sales row.
    assert snapshot.stocks[0]["present"] == 12  # No summing total + warehouses + transit.
    assert snapshot.stocks[0]["reserved"] is None
    assert snapshot.stocks[0]["reserved_known"] is False
    assert snapshot.products[0]["name"] == "Хлопковая рубашка"
    assert snapshot.products[0]["description"] == "Настоящее описание"
    assert "private-" not in str(snapshot)
    assert not wb._headers and wb._http is None
    assert {request.url.host for request in calls} == {
        wb.STATISTICS,
        wb.FINANCE,
        wb.CONTENT,
        wb.ANALYTICS,
    }


async def test_return_sign_and_external_fees_are_not_reversed_or_double_counted():
    row = financial_row(docTypeName="Возврат", retailAmount="-1000.50", forPay="-850.25")
    handler, _ = handler_for(finance=[row])
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    transaction = snapshot.transactions[0]
    assert transaction["accruals_for_sale"] == -1000.5
    assert transaction["sale_commission"] == 100
    assert transaction["amount"] == -900.5
    assert transaction["items"][0]["quantity"] == 2
    assert not transaction["incomplete"]


@pytest.mark.parametrize(
    "changes",
    [
        {"quantity": None},
        {"forPay": None},
        {"paidStorage": None},
        {"additionalPayment": "50"},
        {"cashbackAmount": "10"},
        {"agencyVat": 12},
        {"docTypeName": "Неизвестная корректировка"},
    ],
)
async def test_missing_financial_fields_or_uncertain_adjustments_disable_exact_profit(changes):
    handler, _ = handler_for(finance=[financial_row(**changes)])
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert snapshot.sources["finance"]
    assert snapshot.transactions[0]["incomplete"]
    if changes.get("quantity", 2) is None:
        assert snapshot.transactions[0]["items"][0]["quantity"] == 0  # Unknown, never guessed 1.
    assert any("точная чистая прибыль" in warning for warning in snapshot.warnings)


@pytest.mark.parametrize("currency", ["USD", "KZT", "", None])
async def test_non_ruble_or_unknown_currency_makes_whole_finance_source_unavailable(currency):
    handler, _ = handler_for(finance=[financial_row(), financial_row(rrdId=2, currency=currency)])
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert not snapshot.sources["finance"]
    assert snapshot.transactions == []
    assert any("Суммы разных валют" in warning for warning in snapshot.warnings)


async def test_source_permissions_independent_and_missing_data_is_not_zero():
    handler, _ = handler_for(
        orders=[order_row()],
        sales=[sale_row()],
        denied={WildberriesClient.FINANCE, WildberriesClient.CONTENT, WildberriesClient.ANALYTICS},
    )
    async with client(handler) as wb:
        await wb.validate()
        snapshot = await wb.load(START, END)
    assert snapshot.sources["orders"] and snapshot.sources["sales"]
    assert not snapshot.sources["finance"] and not snapshot.sources["stocks"]
    assert not snapshot.sources["products"]
    assert snapshot.products[0]["sku"] == 123  # Identifiers still resolved without Content right.
    assert snapshot.products[0]["category"] == "Рубашки"
    assert snapshot.products[0]["name"] == ""  # No fabricated card content.
    assert len(snapshot.sales) == 1
    assert "private-" not in " ".join(snapshot.warnings)


async def test_old_period_has_finance_but_no_fake_zero_sales_or_orders():
    handler, calls = handler_for(finance=[financial_row(rrDate="2025-10-10")])
    async with client(handler) as wb:
        snapshot = await wb.load(date(2025, 10, 1), date(2025, 10, 14))
    assert snapshot.sources["finance"]
    assert not snapshot.sources["orders"] and not snapshot.sources["sales"]
    assert not snapshot.sales_available
    assert snapshot.sales == []
    assert not any(request.url.host == WildberriesClient.STATISTICS for request in calls)
    assert any("90 дней" in warning for warning in snapshot.warnings)


async def test_successfully_empty_sales_source_is_distinct_from_unavailable():
    handler, _ = handler_for()
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert snapshot.sales == []
    assert snapshot.sales_available and snapshot.sources["sales"]


async def test_operational_rows_filter_actual_event_date_not_last_change_or_order_date():
    handler, _ = handler_for(
        orders=[order_row(date="2026-09-15T12:00:00")],
        sales=[
            sale_row(date="2026-09-20T12:00:00"),
            sale_row(saleID="R1000", date="2026-10-12T15:00:00"),
        ],
    )
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert snapshot.postings == []
    assert len(snapshot.sales) == 1 and snapshot.sales[0]["status"] == "returned"
    assert snapshot.sales[0]["date"] == "2026-10-12"


async def test_explicit_zero_sales_price_remains_zero_and_null_is_unknown():
    handler, _ = handler_for(
        sales=[sale_row(priceWithDisc=0), sale_row(saleID="S2000", priceWithDisc=None)]
    )
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert snapshot.sales[0]["price"] == 0
    assert snapshot.sales[0]["quantity"] == 1
    assert snapshot.sales[1]["price"] is None
    assert any("в течение суток" in warning for warning in snapshot.warnings)


@pytest.mark.parametrize("document", ["Продажа", "Возврат"])
async def test_zero_revenue_goods_quantity_cannot_hide_costs_in_exact_profit(document):
    handler, _ = handler_for(
        finance=[
            financial_row(
                docTypeName=document,
                retailAmount="0",
                forPay="0",
                ppvzSalesCommission="0",
                quantity=2,
            )
        ]
    )
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    transaction = snapshot.transactions[0]
    assert transaction["incomplete"]
    assert transaction["items"] == [{"sku": 123, "offer_id": "shirt-1", "quantity": 2}]
    assert transaction["accruals_for_sale"] == 0
    assert any("Себестоимость этих операций" in warning for warning in snapshot.warnings)


async def test_finance_paginates_until_no_data_even_for_short_page():
    cursors = []

    async def handler(request):
        payload = json.loads(request.content)
        cursors.append(payload["rrdId"])
        if payload["rrdId"] < 2:
            return httpx.Response(200, json=[financial_row(rrdId=payload["rrdId"] + 1)])
        return httpx.Response(204)

    async with client(handler) as wb:
        rows = await wb._finance(START, END)
    assert len(rows) == 2 and cursors == [0, 1, 2]


async def test_repeated_finance_cursor_or_limit_rejects_partial_source():
    async def handler(request):
        return httpx.Response(200, json=[financial_row()])

    async with client(handler) as wb:
        with pytest.raises(WildberriesError) as error:
            await wb._finance(START, END)
    assert error.value.code == "pagination"


async def test_statistics_uses_full_last_change_timestamp_and_deduplicates_boundary_rows():
    cursors = []

    async def handler(request):
        cursor = request.url.params["dateFrom"]
        cursors.append(cursor)
        if len(cursors) == 1:
            return httpx.Response(200, json=[sale_row()])
        if len(cursors) == 2:
            return httpx.Response(
                200,
                json=[
                    sale_row(),
                    sale_row(
                        saleID="S2000",
                        lastChangeDate="2026-10-13T13:00:00.54321",
                    ),
                ],
            )
        return httpx.Response(200, json=[])

    async with client(handler) as wb:
        rows = await wb._statistics(wb.SALES_PATH, START)
    assert len(rows) == 2
    assert cursors == ["2026-10-01", "2026-10-12T12:30:00.12345", "2026-10-13T13:00:00.54321"]


async def test_repeated_statistics_cursor_rejects_partial_totals():
    async def handler(request):
        return httpx.Response(200, json=[sale_row()])

    async with client(handler) as wb:
        with pytest.raises(WildberriesAPIError) as error:
            await wb._statistics(wb.SALES_PATH, START)
    assert error.value.code == "pagination"


async def test_stocks_task_generation_polling_is_read_only_and_transit_not_reserved():
    states = iter(["new", "processing", "done"])
    paths = []

    async def handler(request):
        paths.append(request.url.path)
        assert request.method == "GET"
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"data": {"status": next(states)}})
        if request.url.path.endswith("/download"):
            return httpx.Response(
                200,
                json=[
                    stock_row(
                        warehouses=[
                            {"warehouseName": "To the client", "quantity": 10},
                            {"warehouseName": "From the client", "quantity": 20},
                            {"warehouseName": "Warehouse A", "quantity": 8},
                        ]
                    )
                ],
            )
        return httpx.Response(200, json={"data": {"taskId": TASK}})

    async with client(handler) as wb:
        rows = await wb._stocks()
        normalized = wb._stock(rows[0])
    assert len(paths) == 5
    assert normalized["present"] == 8
    assert normalized["reserved"] is None


async def test_stock_report_timeout_is_explicit():
    async def handler(request):
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"data": {"status": "processing"}})
        return httpx.Response(200, json={"data": {"taskId": TASK}})

    async with client(handler, max_polls=2) as wb:
        with pytest.raises(WildberriesAPIError, match="ещё формируется"):
            await wb._stocks()


async def test_token_never_sent_to_arbitrary_hosts_or_followed_redirects():
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})

    async with client(handler, max_retries=0) as wb:
        with pytest.raises(ValueError):
            await wb._request("evil.example", "GET", "/ping")
        with pytest.raises(ValueError):
            await wb._request(wb.CONTENT, "POST", "/content/v2/cards/delete/trash")
        with pytest.raises(WildberriesAPIError):
            await wb.validate()
    assert len(calls) == 1 and calls[0].url.host == WildberriesClient.STATISTICS
    assert not wb._headers


@pytest.mark.parametrize(
    "status,expected_code",
    [(401, "authentication"), (403, "authentication"), (429, "rate_limit"), (500, "upstream")],
)
async def test_errors_sanitized_and_classified(status, expected_code):
    async def handler(request):
        return httpx.Response(status, text="private-token private-person")

    async with client(handler, max_retries=0) as wb:
        with pytest.raises(WildberriesAPIError) as error:
            await wb.validate()
    assert error.value.code == expected_code
    assert error.value.status_code == status
    assert "private-" not in str(error.value)


async def test_retry_transient_http_failure_with_no_secret_body_exposure():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(503, text="private-token")
        return httpx.Response(200, json={"Status": "OK"})

    async with client(handler) as wb:
        await wb.validate()
    assert attempts == 2


async def test_cards_cursor_pagination_and_card_whitelist():
    calls = []

    async def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "cards": [{"nmID": i + 1} for i in range(100)],
                    "cursor": {"total": 100, "nmID": 100, "updatedAt": "2026-10-01T00:00:00Z"},
                },
            )
        return httpx.Response(200, json={"cards": [{"nmID": 101}], "cursor": {"total": 1}})

    async with client(handler) as wb:
        rows = await wb._cards()
    assert len(rows) == 101
    assert calls[1]["settings"]["cursor"] == {
        "limit": 100,
        "nmID": 100,
        "updatedAt": "2026-10-01T00:00:00Z",
    }


async def test_backwards_card_cursor_rejects_incomplete_catalog():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "cards": [{"nmID": i + 1} for i in range(100)],
                "cursor": {
                    "total": 100,
                    "nmID": 100,
                    "updatedAt": "2026-10-02T00:00:00Z" if calls == 1 else "2026-10-01T00:00:00Z",
                },
            },
        )

    async with client(handler) as wb:
        with pytest.raises(WildberriesAPIError) as error:
            await wb._cards()
    assert error.value.code == "pagination"
    assert calls == 2


async def test_optional_kyrgyz_tax_field_can_be_absent_for_russian_seller():
    row = financial_row()
    del row["agencyVat"]
    handler, _ = handler_for(finance=[row])
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert not snapshot.transactions[0]["incomplete"]


async def test_malformed_sales_row_discards_whole_source_instead_of_partial_totals():
    handler, _ = handler_for(sales=[sale_row(), sale_row(saleID="unknown-id")])
    async with client(handler) as wb:
        snapshot = await wb.load(START, END)
    assert not snapshot.sources["sales"] and not snapshot.sales_available
    assert snapshot.sales == []


async def test_statistics_rejects_malformed_non_string_cursor():
    async def handler(request):
        return httpx.Response(200, json=[sale_row(lastChangeDate=20261012)])

    async with client(handler) as wb:
        with pytest.raises(WildberriesAPIError) as error:
            await wb._statistics(wb.SALES_PATH, START)
    assert error.value.code == "schema"
