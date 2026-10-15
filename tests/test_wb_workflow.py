"""Current WB HTTP contracts through normalization and actual Polars reports."""

import json
from datetime import date

import httpx
import pytest

from ozon_bot.analytics import build_report
from ozon_bot.domain import AnalysisContext, Period
from ozon_bot.wildberries import WildberriesClient


@pytest.fixture
def wb_transport():
    def finance(rrd_id, *, returned=False):
        zero_fields = (
            "paidStorage deduction paidAcceptance penalty additionalPayment rebillLogisticCost "
            "cashbackAmount cashbackDiscount cashbackCommissionChange paymentSchedule "
            "installmentCofinancingAmount"
        ).split()
        return {
            **dict.fromkeys(zero_fields, "0"),
            "rrdId": rrd_id,
            "nmId": 123,
            "currency": "RUB",
            "rrDate": "2026-10-13" if returned else "2026-10-11",
            "docTypeName": "Возврат" if returned else "Продажа",
            "quantity": 1 if returned else 2,
            "retailAmount": "200" if returned else "1000",
            "forPay": "170" if returned else "850",
            "ppvzSalesCommission": "20" if returned else "100",
            "deliveryService": "10" if returned else "50",
            "vendorCode": "shirt-1",
            "subjectName": "Рубашки",
            "title": "Хлопковая рубашка",
        }

    def event(day, **extra):
        return {
            "nmId": 123,
            "supplierArticle": "shirt-1",
            "subject": "Рубашки",
            "date": f"2026-10-{day}T12:00:00",
            "lastChangeDate": "2026-10-14T14:00:00.12345",
            "priceWithDisc": 500,
            **extra,
        }

    async def handler(request):
        path = request.url.path
        if path == WildberriesClient.FINANCE_PATH:
            return (
                httpx.Response(204)
                if json.loads(request.content)["rrdId"]
                else httpx.Response(
                    200,
                    json=[finance(1), finance(2, returned=True)],
                )
            )
        if path in {WildberriesClient.ORDERS_PATH, WildberriesClient.SALES_PATH}:
            if request.url.params["dateFrom"] != "2026-10-10":
                return httpx.Response(200, json=[])
            rows = (
                [event("09", srid="older-order"), event("10", srid="new-order")]
                if path == WildberriesClient.ORDERS_PATH
                else [
                    event("11", saleID="S1"),
                    event("12", saleID="S2"),
                    event("13", saleID="R1", priceWithDisc=200),
                ]
            )
            return httpx.Response(200, json=rows)
        if path == WildberriesClient.CARDS_PATH:
            return httpx.Response(
                200,
                json={
                    "cards": [
                        {
                            "nmID": 123,
                            "vendorCode": "shirt-1",
                            "subjectName": "Рубашки",
                            "title": "Хлопковая рубашка",
                            "description": "Тестовое описание",
                        }
                    ],
                    "cursor": {"total": 1},
                },
            )
        if path == WildberriesClient.STOCKS_PATH:
            return httpx.Response(
                200, json={"data": {"taskId": "06e06887-9d9f-491f-b16a-bb1766fcb8d2"}}
            )
        if path.endswith("/status"):
            return httpx.Response(200, json={"data": {"status": "done"}})
        if path.endswith("/download"):
            return httpx.Response(
                200,
                json=[
                    {
                        "nmId": 123,
                        "vendorCode": "shirt-1",
                        "warehouses": [
                            {"warehouseName": "Total in warehouses", "quantity": 12},
                            {"warehouseName": "Warehouse A", "quantity": 12},
                            {"warehouseName": "To the client", "quantity": 40},
                        ],
                    }
                ],
            )
        raise AssertionError(f"Unexpected endpoint: {path}")

    return httpx.MockTransport(handler)


async def test_current_wb_http_to_finance_sales_stocks_and_category_reports(wb_transport):
    period = Period(date(2026, 10, 10), date(2026, 10, 14), "Тестовый период")
    async with WildberriesClient(
        "synthetic-token",
        transport=wb_transport,
        today=date(2026, 10, 15),
        request_interval=0,
        poll_interval=0,
    ) as client:
        snapshot = await client.load(period.start, period.end)
    context = AnalysisContext(snapshot, period, costs={"shirt-1": 100}, tax_rate=6)
    finance = build_report("finance", context).facts
    assert finance["source_complete"] and finance["revenue"] == 800
    assert (finance["commission"], finance["logistics"], finance["other"]) == (80, 60, 40)
    assert finance["ledger_net"] == 620 == finance["revenue"] - 80 - 60 - 40
    assert finance["cogs"] == 100 and finance["taxes"] == 48 and finance["profit"] == 472
    sales = build_report("sales", context).facts
    assert sales["orders"] == 1  # Earlier order excluded, its later sale still counted.
    assert (sales["delivered_units"], sales["returned_units"]) == (2, 1)
    assert sales["sales_basis"] == "event_date" and sales["average_unit_price"] == 500
    daily = {row["date"]: row for row in sales["daily"]}
    assert daily["2026-10-10"]["delivered_units"] == 0
    assert daily["2026-10-11"]["delivered_units"] == 1
    assert daily["2026-10-13"]["returned_units"] == 1
    stock = build_report("stocks", context).facts
    assert stock["available_units"] == 12 and stock["reserved_units"] is None
    assert stock["products"][0]["daily_demand"] == pytest.approx(0.2)
    category = build_report("categories", context).facts["categories"][0]
    assert category["category"] == "Рубашки"
    assert (category["orders"], category["buyouts"], category["returns"]) == (1, 2, 1)
    assert category["net_realized_turnover"] == 800
