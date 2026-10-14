"""A representative 2026 Seller API response -> Polars analytics integration."""

import json
from datetime import date

import httpx
import pytest

from ozon_bot.analytics import build_report
from ozon_bot.domain import AnalysisContext, Period
from ozon_bot.ozon import OzonClient


def money(amount):
    return {"amount": str(amount), "currency": "RUB"}


@pytest.mark.asyncio
@pytest.mark.parametrize("with_refund", [False, True])
async def test_seller_responses_feed_consistent_finance_sales_stock_and_categories(with_refund):
    day = date(2026, 1, 15)
    accruals = [
        {
            "accrual_id": 1,
            "date": day.isoformat(),
            "accrued_category": "POSTING",
            "unit_number": "fbo-delivered",
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
                            "services": [{"type_id": 1, "accrued": money("-50.50")}],
                        },
                    }
                ],
            },
        },
        {
            "accrual_id": 2,
            "date": day.isoformat(),
            "accrued_category": "NON_ITEM",
            "total_amount": money("-10.25"),
            "non_item_fee": {"type_id": 2, "accrued": money("-10.25")},
        },
    ]
    if with_refund:
        accruals.append(
            {
                "accrual_id": 3,
                "date": day.isoformat(),
                "accrued_category": "POSTING",
                "unit_number": "fbo-delivered",
                "total_amount": money("-430.00"),
                "posting": {
                    "delivery_schema": "fbo",
                    "products": [
                        {
                            "sku": 111,
                            "commission": {
                                "sale_amount": money("-500.25"),
                                "commission": money("50.00"),
                            },
                            "delivery": {"services": [{"type_id": 1, "accrued": money("20.25")}]},
                        }
                    ],
                },
            }
        )

    def handler(request):
        path = request.url.path
        payload = json.loads(request.content)
        if path == "/v1/finance/accrual/types":
            result = {
                "accrual_types": [{"id": 1, "name": "Логистика"}, {"id": 2, "name": "Реклама"}]
            }
        elif path == "/v1/finance/accrual/by-day":
            assert payload["date"] == day.isoformat()
            result = {"accruals": accruals, "last_id": ""}
        elif path == "/v3/product/list":
            result = {
                "result": {
                    "items": [{"product_id": 1}, {"product_id": 2}],
                    "total_items": 2,
                    "last_id": "",
                }
            }
        elif path == "/v3/product/info/list":
            result = {
                "items": [
                    {
                        "id": 1,
                        "name": "Кружка",
                        "offer_id": "cup",
                        "description_category_id": 901,
                        "sources": [{"sku": 111}],
                        "price": "500.25",
                    },
                    {
                        "id": 2,
                        "name": "Ложка",
                        "offer_id": "spoon",
                        "description_category_id": 902,
                        "sources": [{"sku": 222}],
                        "price": "300.00",
                    },
                ]
            }
        elif path == "/v1/description-category/tree":
            result = {
                "result": [
                    {
                        "description_category_id": 901,
                        "category_name": "Категория 901",
                        "children": [],
                    },
                    {
                        "description_category_id": 902,
                        "category_name": "Категория 902",
                        "children": [],
                    },
                ]
            }
        elif path == "/v1/product/info/description":
            result = {"result": {"id": payload["product_id"], "description": "Содержание карточки"}}
        elif path in {"/v3/posting/fbo/list", "/v4/posting/fbs/list"}:
            fbo = path == "/v3/posting/fbo/list"
            result = {
                "postings": [
                    {
                        "posting_number": "fbo-delivered" if fbo else "fbs-cancelled",
                        "status": "delivered" if fbo else "cancelled",
                        "in_process_at": f"{day.isoformat()}T10:00:00Z",
                        "products": [
                            {
                                "sku": 111 if fbo else 222,
                                "quantity": 2 if fbo else 1,
                                "price": money("500.25" if fbo else "300.00"),
                            }
                        ],
                    }
                ],
                "has_next": False,
                "cursor": "",
            }
        elif path == "/v4/product/info/stocks":
            result = {
                "items": [{"product_id": 1, "stocks": [{"present": 10, "reserved": 3}]}],
                "total_items": 1,
                "cursor": "",
            }
        else:
            raise AssertionError(path)
        return httpx.Response(200, json=result)

    async with OzonClient(
        "123", "test-store-key", transport=httpx.MockTransport(handler)
    ) as client:
        snapshot = await client.load(day, day)
    context = AnalysisContext(
        snapshot=snapshot, period=Period(day, day, "15 января"), costs={"111": 200}, tax_rate=6
    )

    finance = build_report("finance", context).facts
    assert finance["ledger_net"] == pytest.approx(409.50 if with_refund else 839.50)
    assert finance["ledger_net"] == pytest.approx(
        finance["revenue"] - finance["commission"] - finance["logistics"] - finance["other"]
    )
    assert finance["other"] == pytest.approx(10.25)  # Advertising is included once in the ledger.
    assert finance["known_cogs"] == 400  # Finance quantities matched to the exact two-unit posting.
    if with_refund:
        assert (
            finance["profit"] is None
        )  # Original order quantity cannot prove a partial return's units.
        assert finance["profit_estimated"]
    else:
        assert finance["cogs"] == 400
        assert finance["profit"] == pytest.approx(379.47)

    sales = build_report("sales", context).facts
    assert sales["orders"] == 2
    assert sales["delivered_units"] == 2
    assert sales["cancelled_orders"] == 1
    assert sales["returned_units"] == (1 if with_refund else 0)
    assert sales["average_basket"] == pytest.approx(1000.50)

    stock = build_report("stocks", context).facts["products"][0]
    assert stock["sku"] == "111"  # Stocks identify product_id; postings identify SKU.
    assert stock["available"] == 7
    assert stock["sold_units"] == 2

    categories = {
        row["category"]: row for row in build_report("categories", context).facts["categories"]
    }
    assert categories["Категория 901"]["buyouts"] == 2
    assert categories["Категория 901"]["buyout_revenue"] == pytest.approx(1000.50)
    assert categories["Категория 902"]["orders"] == 1
    assert categories["Категория 902"]["buyouts"] == 0
