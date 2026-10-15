from datetime import date

import pytest

from ozon_bot.analytics import build_report
from ozon_bot.domain import AnalysisContext, Period, Snapshot


def context(**kwargs):
    snapshot_fields = {
        key: kwargs.pop(key, [])
        for key in ("transactions", "postings", "products", "stocks", "daily", "warnings")
    }
    snapshot_fields.update(
        marketplace=kwargs.pop("marketplace", "ozon"),
        sales=kwargs.pop("sales", []),
        sources=kwargs.pop("sources", {}),
        sales_available=kwargs.pop("sales_available", False),
    )
    return AnalysisContext(
        snapshot=Snapshot(**snapshot_fields),
        period=Period(date(2026, 1, 1), date(2026, 1, 30), "январь"),
        costs=kwargs.pop("costs", {}),
        tax_rate=kwargs.pop("tax_rate", 6),
        ads=kwargs.pop("ads", []),
        seo=kwargs.pop("seo", []),
        competitors=kwargs.pop("competitors", []),
        **kwargs,
    )


def test_finance_reconciles_signed_ledger_and_returns_without_duplicate_ads():
    report = build_report(
        "finance",
        context(
            transactions=[
                {
                    "amount": 720,
                    "accruals_for_sale": 1000,
                    "sale_commission": -100,
                    "services": [
                        {"name": "MarketplaceServiceItemDeliveryToCustomer", "price": -80},
                        {"name": "MarketplaceServiceItemStorage", "price": -100},
                    ],
                    "items": [{"sku": 1, "quantity": 2}],
                },
                {
                    "amount": -390,
                    "accruals_for_sale": -500,
                    "sale_commission": 50,
                    "services": [{"name": "MarketplaceServiceItemDeliveryToCustomer", "price": 60}],
                    "items": [{"sku": 1, "quantity": 1}],
                },
            ],
            costs={"1": 200},
            ads=[{"spend": 100}],
        ),
    )
    facts = report.facts
    assert facts["revenue"] == 500
    assert facts["commission"] == 50
    assert facts["logistics"] == 20
    assert facts["other"] == 100
    assert (
        facts["ledger_net"]
        == facts["revenue"] - facts["commission"] - facts["logistics"] - facts["other"]
    )
    assert facts["cogs"] == 200
    assert facts["taxes"] == 30
    assert facts["profit"] == 100


def test_finance_never_fakes_profit_with_missing_cost_or_missing_ledger():
    transaction = {
        "amount": 80,
        "accruals_for_sale": 100,
        "sale_commission": -20,
        "items": [{"sku": 1, "quantity": 1}],
    }
    report = build_report("finance", context(transactions=[transaction]))
    assert report.facts["profit"] is None
    assert report.facts["missing_cost_skus"] == ["1"]
    without_amount = {key: value for key, value in transaction.items() if key != "amount"}
    report = build_report("finance", context(transactions=[without_amount], costs={"1": 30}))
    assert report.facts["profit"] is None
    assert not report.facts["ledger_complete"]


def test_finance_reports_estimated_quantities_and_rejects_nonfinite_costs():
    transaction = {
        "amount": 80,
        "accruals_for_sale": 100,
        "sale_commission": -20,
        "items": [{"sku": 1}],
    }
    report = build_report("finance", context(transactions=[transaction], costs={"1": 30}))
    assert report.facts["profit_estimated"]
    assert report.facts["profit"] is None
    assert report.facts["estimated_profit"] == 44
    assert "принято 1 шт." in report.text
    assert (
        build_report(
            "finance", context(transactions=[transaction], costs={"1": float("nan")})
        ).facts["profit"]
        is None
    )


def test_finance_matches_sale_quantity_but_never_infers_partial_refund_quantity():
    sale = {
        "amount": 80,
        "accruals_for_sale": 100,
        "sale_commission": -20,
        "posting": {"posting_number": "a"},
        "items": [{"sku": 1}],
    }
    ctx = context(
        transactions=[sale], costs={"1": 30}, postings=[posting("a", "delivered", quantity=2)]
    )
    report = build_report("finance", ctx)
    assert report.facts["cogs"] == 60
    assert report.facts["profit"] == 14
    refund = {**sale, "amount": -80, "accruals_for_sale": -100, "sale_commission": 20}
    ctx.snapshot.transactions.append(refund)
    assert build_report("finance", ctx).facts["profit"] is None


def posting(number, status, sku=1, quantity=1, price=100):
    return {
        "posting_number": number,
        "status": status,
        "in_process_at": "2026-01-10T00:00:00Z",
        "products": [{"sku": sku, "quantity": quantity, "price": price}],
    }


def test_sales_distinguishes_order_delivery_cancellation_and_return():
    report = build_report(
        "sales",
        context(
            postings=[
                posting("a", "delivered", quantity=2),
                posting("b", "awaiting_packaging"),
                posting("c", "cancelled"),
                posting("d", "returned"),
                posting("a", "delivered", quantity=2),
            ]
        ),
    )
    assert report.facts["orders"] == 4
    assert report.facts["delivered_orders"] == 1
    assert report.facts["delivered_units"] == 2
    assert report.facts["cancelled_orders"] == 1
    assert report.facts["pending_orders"] == 1
    assert report.facts["returned_units"] == 1
    assert report.facts["average_basket"] == 200


def test_categories_paginate_aggregate_and_keep_unknown_coverage():
    ctx = context(
        postings=[
            posting(str(i), "delivered", sku=i, quantity=2, price=i * 10) for i in range(1, 11)
        ],
        products=[{"sku": i, "description_category_id": i} for i in range(1, 10)],
    )
    first = build_report("categories", ctx)
    second = build_report("categories", ctx, page=1)
    assert first.facts["pages"] == 2
    assert len(first.facts["page_categories"]) == 8
    assert len(second.facts["page_categories"]) == 2
    assert first.facts["ordered_turnover"] == 1100
    assert all(row["buyouts"] == 2 for row in first.facts["categories"])
    assert any(row["category"] == "Без категории" for row in first.facts["categories"])
    assert "Не для всех SKU" in second.text


def test_stocks_subtract_reservations_and_derive_supply_from_deliveries():
    report = build_report(
        "stocks",
        context(
            products=[{"id": 100, "sku": 1, "name": "Товар"}],
            stocks=[{"product_id": 100, "stocks": [{"present": 20, "reserved": 8}]}],
            postings=[
                posting("a", "delivered", quantity=30),
                posting("b", "cancelled", quantity=100),
            ],
        ),
    )
    product = report.facts["products"][0]
    assert product["available"] == 12
    assert product["daily_demand"] == 1
    assert product["days_cover"] == 12
    assert product["recommended_supply"] == 18
    assert report.facts["shortage_count"] == 1


def test_ad_zero_revenue_is_undefined_drr_and_inefficient():
    report = build_report(
        "ads",
        context(
            ads=[
                {
                    "date": "2026-01-02",
                    "campaign_id": "a",
                    "sku": "1",
                    "spend": 150,
                    "revenue": 0,
                    "orders": 0,
                }
            ]
        ),
    )
    assert report.facts["drr"] is None
    assert report.facts["roas"] == 0
    assert len(report.facts["inefficient_products"]) == 1
    assert "ДРР не определён" in report.text


def test_reports_do_not_invent_search_ranks_or_competitor_data():
    seo = build_report("seo", context(products=[{"sku": 1, "name": "Товар"}]))
    assert seo.facts["ranks_available"] == 0
    assert "описание: не передано" in seo.text
    competitors = build_report("competitors", context())
    assert not competitors.facts["available"]


def test_seo_import_matches_api_source_sku_and_checks_supplied_keywords():
    report = build_report(
        "seo",
        context(
            products=[{"id": 100, "name": "Термокружка стальная", "sources": [{"sku": 1}]}],
            seo=[
                {
                    "sku": "1",
                    "title": "Термокружка стальная",
                    "keywords": "термокружка, кружка 500 мл",
                }
            ],
        ),
    )
    assert report.facts["card_count"] == 1
    assert report.facts["products"][0]["keywords_not_in_title"] == ["кружка 500 мл"]


def test_competitor_latest_observation_and_price_comparison():
    report = build_report(
        "competitors",
        context(
            products=[{"sku": 1, "price": 120}],
            competitors=[
                {"sku": 1, "competitor_sku": 2, "date": "2026-01-01", "price": 100, "position": 10},
                {"sku": 1, "competitor_sku": 2, "date": "2026-01-15", "price": 100, "position": 7},
            ],
        ),
    )
    product = report.facts["products"][0]
    assert product["price_gap_pct"] == 20
    assert product["position_change"] == 3
    assert product["rating"] is None


@pytest.mark.parametrize(
    "kind", ["finance", "sales", "ads", "seo", "competitors", "stocks", "categories"]
)
def test_empty_sources_are_explicitly_unavailable(kind):
    assert not build_report(kind, context()).facts["available"]


@pytest.mark.parametrize("marketplace", ["wb", "wildberries"])
def test_wb_sales_use_actual_event_dates_and_do_not_invent_basket_checks(marketplace):
    report = build_report(
        "sales",
        context(
            marketplace=marketplace,
            sources={"sales": True, "orders": True},
            postings=[posting("a", "unknown"), posting("b", "cancelled", sku=2)],
            products=[{"sku": 1, "category": "Посуда"}],
            sales=[
                {
                    "sale_id": "S1",
                    "date": "2026-01-15",
                    "sku": 1,
                    "quantity": 2,
                    "price": 100,
                    "status": "delivered",
                },
                {
                    "sale_id": "R1",
                    "date": "2026-01-18",
                    "sku": 1,
                    "quantity": 1,
                    "price": 100,
                    "status": "returned",
                },
                {
                    "sale_id": "old",
                    "date": "2025-12-31",
                    "sku": 1,
                    "quantity": 10,
                    "price": 100,
                    "status": "delivered",
                },
            ],
        ),
    )
    assert report.facts["orders"] == 2
    assert report.facts["cancelled_orders"] == 1
    assert report.facts["delivered_units"] == 2
    assert report.facts["returned_units"] == 1
    assert report.facts["delivered_value"] == 200
    assert report.facts["average_basket"] is None
    assert report.facts["average_unit_price"] == 100
    daily = {row["date"]: row for row in report.facts["daily"]}
    assert daily["2026-01-15"]["delivered_units"] == 2
    assert daily["2026-01-15"]["orders"] == 0
    assert daily["2026-01-18"]["returned_units"] == 1
    assert "Ozon" not in report.text


def test_wb_unavailable_sales_do_not_become_zero_buyouts_or_returns():
    report = build_report(
        "sales",
        context(
            marketplace="wb",
            sources={"sales": False, "orders": True},
            postings=[posting("a", "unknown")],
        ),
    )
    assert report.facts["orders"] == 1
    assert report.facts["delivered_units"] is None
    assert report.facts["returned_units"] is None
    assert report.facts["delivered_value"] is None
    assert not report.facts["sales_complete"]


def test_wb_successfully_loaded_empty_sales_are_true_zeroes():
    report = build_report(
        "sales", context(marketplace="wb", sources={"sales": True, "orders": True})
    )
    assert report.facts["available"]
    assert report.facts["orders"] == 0
    assert report.facts["delivered_units"] == 0
    assert report.facts["returned_units"] == 0
    assert report.facts["average_unit_price"] is None


def test_wb_missing_sale_price_does_not_become_zero_revenue():
    ctx = context(
        marketplace="wildberries",
        sources={"sales": True, "orders": False},
        sales=[
            {
                "sale_id": "S1",
                "date": "2026-01-15",
                "sku": 1,
                "quantity": 1,
                "price": None,
                "status": "delivered",
            }
        ],
    )
    sales = build_report("sales", ctx).facts
    assert sales["delivered_units"] == 1
    assert sales["delivered_value"] is None
    assert sales["average_unit_price"] is None
    assert sales["orders"] is None
    categories = build_report("categories", ctx).facts
    assert categories["categories"][0]["buyouts"] == 1
    assert categories["categories"][0]["buyout_revenue"] is None
    assert categories["buyout_revenue"] is None


def test_wb_finance_reconciles_signed_payout_and_never_subtracts_fees_twice():
    ctx = context(
        marketplace="wb",
        sources={"finance": True},
        costs={"1": 100},
        transactions=[
            {
                "amount": 730,
                "accruals_for_sale": 1000,
                "sale_commission": -150,
                "items": [{"sku": 1, "quantity": 2}],
                "services": [
                    {"name": "Логистика WB", "price": -70},
                    {"name": "Хранение", "price": -20},
                ],
            },
            {
                "amount": -360,
                "accruals_for_sale": -500,
                "sale_commission": 75,
                "items": [{"sku": 1, "quantity": 1}],
                "services": [{"name": "Логистика WB", "price": 35}],
            },
        ],
    )
    report = build_report("finance", ctx)
    assert report.facts["ledger_net"] == 370
    assert report.facts["revenue"] == 500
    assert report.facts["commission"] == 75
    assert report.facts["logistics"] == 35
    assert report.facts["other"] == 20
    assert report.facts["cogs"] == 100
    assert report.facts["profit"] == 240
    assert "Итог операций Wildberries" in report.text
    assert "Ozon" not in report.text
    ctx.snapshot.transactions[0]["incomplete"] = True
    incomplete = build_report("finance", ctx).facts
    assert incomplete["profit"] is None
    assert incomplete["estimated_profit"] is None
    assert incomplete["cogs"] is None


def test_wb_stocks_keep_available_quantity_and_do_not_treat_transit_as_reserves():
    report = build_report(
        "stocks",
        context(
            marketplace="wb",
            sources={"sales": True},
            stocks=[
                {
                    "sku": 1,
                    "present": 10,
                    "reserved": 4,
                    "reserved_known": False,
                    "inWayToClient": 9,
                    "inWayFromClient": 5,
                }
            ],
            sales=[
                {
                    "sale_id": "S1",
                    "date": "2026-01-15",
                    "sku": 1,
                    "quantity": 6,
                    "price": 100,
                    "status": "delivered",
                },
                {
                    "sale_id": "R1",
                    "date": "2026-01-16",
                    "sku": 1,
                    "quantity": 2,
                    "price": 100,
                    "status": "returned",
                },
            ],
        ),
    )
    product = report.facts["products"][0]
    assert product["available"] == 10
    assert product["sold_units"] == 4
    assert product["days_cover"] == 75
    assert product["reserved"] is None
    assert report.facts["reserved_units"] is None
    assert report.facts["excess_count"] == 1


def test_wb_stock_forecast_is_unavailable_without_complete_sales_source():
    report = build_report(
        "stocks",
        context(
            marketplace="wb",
            sources={"sales": False},
            stocks=[{"sku": 1, "present": 10, "reserved_known": False}],
        ),
    )
    product = report.facts["products"][0]
    assert product["available"] == 10
    assert product["daily_demand"] is None
    assert product["recommended_supply"] is None
    assert report.facts["shortage_count"] is None
    assert report.facts["excess_count"] is None


def test_wb_category_pages_separate_orders_from_realized_sale_events():
    products = [{"sku": i, "category": f"Категория {i}"} for i in range(1, 11)]
    sales = [
        {
            "sale_id": f"S{i}",
            "date": "2026-01-20",
            "sku": i,
            "quantity": 2,
            "price": 100,
            "status": "delivered",
        }
        for i in range(1, 11)
    ]
    sales.append(
        {
            "sale_id": "R1",
            "date": "2026-01-21",
            "sku": 1,
            "quantity": 1,
            "price": 100,
            "status": "returned",
        }
    )
    ctx = context(
        marketplace="wb",
        sources={"sales": True, "orders": True},
        products=products,
        postings=[posting(str(i), "unknown", sku=i) for i in range(1, 11)],
        sales=sales,
    )
    first = build_report("categories", ctx)
    second = build_report("categories", ctx, page=1)
    assert first.facts["pages"] == 2
    assert len(first.facts["page_categories"]) == 8
    assert len(second.facts["page_categories"]) == 2
    rows = {row["category"]: row for row in first.facts["categories"]}
    assert rows["Категория 1"]["orders"] == 1
    assert rows["Категория 1"]["buyouts"] == 2
    assert rows["Категория 1"]["returns"] == 1
    assert rows["Категория 1"]["net_realized_turnover"] == 100
    assert first.facts["buyout_revenue"] == 2000
    assert first.facts["net_realized_turnover"] == 1900


def test_wb_category_buyouts_remain_unknown_when_sales_source_is_unavailable():
    report = build_report(
        "categories",
        context(
            marketplace="wb",
            sources={"orders": True, "sales": False},
            products=[{"sku": 1, "category": "Посуда"}],
            postings=[posting("a", "unknown")],
        ),
    )
    row = report.facts["categories"][0]
    assert row["orders"] == 1
    assert row["buyouts"] is None
    assert row["returns"] is None
    assert row["buyout_revenue"] is None
    assert report.facts["buyout_revenue"] is None


def test_wb_seo_does_not_score_unavailable_content_as_empty_cards():
    ctx = context(
        marketplace="wildberries",
        sources={"products": False},
        products=[
            {"sku": 1, "name": "", "category": "Посуда"},
            {"sku": 2, "name": "", "category": "Посуда"},
        ],
    )
    missing = build_report("seo", ctx)
    assert not missing.facts["available"]
    assert "«Контент»" in missing.text
    assert "нет названия" not in missing.text
    ctx.seo = [{"sku": "1", "title": "Кружка керамическая с ручкой 350 мл"}]
    imported = build_report("seo", ctx)
    assert imported.facts["available"]
    assert imported.facts["card_count"] == 1
    assert imported.facts["audit_source"] == "csv_only"
    assert imported.facts["products"][0]["sku"] == "1"
    assert imported.facts["products"][0]["description_known"] is False
    assert "нет названия" not in imported.text
