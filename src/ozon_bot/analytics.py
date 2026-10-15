"""Explainable seller analytics. Monetary amounts are RUB and tax_rate is a percent.

API ledgers and observed snapshots stay separate: a missing source is never a zero
profit, a cancellation is never a return, and a missing search rank is never a rank.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import polars as pl

from ozon_bot.domain import AnalysisContext, Report

PAGE_SIZE = 8
DELIVERED = {"delivered", "delivered_to_customer", "received"}
CANCELLED = {"cancelled", "canceled"}
RETURNED = {"returned", "returned_to_seller", "return"}
LOGISTICS_NAMES = (
    "delivery",
    "logistic",
    "fulfillment",
    "dropoff",
    "sort",
    "redispatch",
    "returnflow",
    "pickup",
    "выдач",
    "достав",
    "логист",
    "перевоз",
    "обработка возврат",
    "возврат товаров",
    "возврат товара",
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _valid(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    return None


def _during(row: dict, context: AnalysisContext, keys: tuple[str, ...]) -> bool:
    observed = next((_date(row.get(key)) for key in keys if _date(row.get(key))), None)
    return observed is None or context.period.start <= observed <= context.period.end


def _money(value: float | None) -> str:
    return "нет данных" if value is None else f"{value:,.2f}".replace(",", " ") + " ₽"


def _pct(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.1f}%"


def _ratio(numerator: float, denominator: float, multiplier: float = 1) -> float | None:
    return numerator / denominator * multiplier if denominator > 0 else None


def _frame(rows: list[dict], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, strict=False) if rows else pl.DataFrame(schema=schema)


def _sum(frame: pl.DataFrame, column: str) -> float:
    return float(frame.select(pl.col(column).sum()).item() or 0)


def _marketplace(context: AnalysisContext) -> str:
    return "Wildberries" if _is_wb(context) else "Ozon"


def _is_wb(context: AnalysisContext) -> bool:
    return getattr(context.snapshot, "marketplace", "ozon") in {"wb", "wildberries"}


def _source_complete(context: AnalysisContext, kind: str, observed: bool) -> bool:
    sources = getattr(context.snapshot, "sources", {})
    if kind in sources:
        return sources[kind] is True
    if kind == "sales" and getattr(context.snapshot, "sales_available", False):
        return True
    return observed


def _report(
    context: AnalysisContext,
    title: str,
    lines: list[str],
    facts: dict,
    warnings: list[str] | None = None,
) -> Report:
    extra = list(dict.fromkeys([*(warnings or []), *context.snapshot.warnings]))
    facts["warnings"] = extra
    facts["marketplace"] = getattr(context.snapshot, "marketplace", "ozon")
    body = [f"{title} · {context.period.label}", *lines]
    if extra:
        body += ["Ограничения: " + " ".join(extra)]
    text = "\n".join(body)
    if len(text) > 3500:
        text = (
            text[:3440].rsplit("\n", 1)[0]
            + "\nДля подробного отчёта выберите меньший период или следующую страницу."
        )
    return Report(title=title, text=text, facts=facts)


def _catalog(context: AnalysisContext) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in context.snapshot.products:
        for key in ("sku", "id", "product_id", "offer_id"):
            if row.get(key) is not None:
                index[str(row[key])] = row
        for source in row.get("sources") or []:
            if source.get("sku") is not None:
                index[str(source["sku"])] = row
    return index


def _key(row: dict) -> str:
    return str(
        next(
            (row[k] for k in ("sku", "offer_id", "product_id", "id") if row.get(k) is not None),
            "неизвестный",
        )
    )


def _canonical_key(item: dict, catalog: dict[str, dict]) -> str:
    product = catalog.get(_key(item), {})
    if product.get("sku") is not None:
        return str(product["sku"])
    sources = product.get("sources") or []
    if sources and sources[0].get("sku") is not None:
        return str(sources[0]["sku"])
    return _key(item)


def _sale_quantity(transaction: dict, item: dict, context: AnalysisContext) -> float | None:
    """Resolve a sale's missing quantity from the exact posting, not a cohort total.

    Never infer refund units from the original posting: partial returns exist.
    """
    if item.get("quantity") is not None:
        value = _num(item.get("quantity"))
        return value if value > 0 else None
    if _num(transaction.get("accruals_for_sale")) <= 0:
        return None
    posting_data = transaction.get("posting") or {}
    number = posting_data.get("posting_number") or transaction.get("unit_number")
    if not number:
        return None
    matches = [
        post for post in context.snapshot.postings if str(post.get("posting_number")) == str(number)
    ]
    if not matches:
        return None
    matches = [
        product for product in matches[-1].get("products") or [] if _key(product) == _key(item)
    ]
    if len(matches) != 1:
        return None
    value = _num(matches[0].get("quantity"))
    return value if value > 0 else None


def _cost(item: dict, catalog: dict[str, dict], costs: dict[str, float]) -> float | None:
    product = catalog.get(_key(item), {})
    for row in (item, product):
        for key in ("sku", "offer_id", "id", "product_id"):
            identifier = str(row.get(key))
            if identifier in costs and _valid(costs[identifier]) and float(costs[identifier]) >= 0:
                return float(costs[identifier])
    return None


def _transactions(context: AnalysisContext) -> list[dict]:
    return [
        row
        for row in context.snapshot.transactions
        if _during(row, context, ("operation_date", "date"))
    ]


def _finance(context: AnalysisContext) -> Report:
    transactions = _transactions(context)
    if not transactions:
        return _report(
            context,
            "Финансы",
            ["Финансовые операции за период не получены. Прибыль рассчитать нельзя."],
            {"available": False, "profit": None, "revenue": None},
        )
    catalog = _catalog(context)
    rows: list[dict] = []
    missing: set[str] = set()
    quantity_assumed = False
    ledger_complete = True
    revenue_complete = True
    unknown_units = False
    invalid_inputs = False
    source_incomplete = not _source_complete(context, "finance", bool(transactions))
    for transaction in transactions:
        source_incomplete |= bool(transaction.get("incomplete"))
        amount = _num(transaction.get("amount"))
        revenue = _num(transaction.get("accruals_for_sale"))
        commission = _num(transaction.get("sale_commission"))
        ledger_complete &= _valid(transaction.get("amount"))
        revenue_complete &= _valid(transaction.get("accruals_for_sale"))
        invalid_inputs |= transaction.get("sale_commission") is not None and not _valid(
            transaction.get("sale_commission")
        )
        logistics = 0.0
        for service in transaction.get("services") or []:
            name = str(service.get("name", "")).lower()
            if any(word in name for word in LOGISTICS_NAMES):
                if not _valid(service.get("price")):
                    invalid_inputs = True
                logistics += _num(service.get("price"))
        cogs = confirmed_cogs = 0.0
        sold_units = returned_units = covered_units = 0.0
        if revenue:
            items = transaction.get("items") or []
            unknown_units |= not bool(items)
            sign = 1 if revenue > 0 else -1
            for item in items:
                quantity = _sale_quantity(transaction, item, context)
                estimated_quantity = quantity is None
                if item.get("quantity") is not None and quantity is None:
                    unknown_units = True
                    continue
                if estimated_quantity:
                    quantity_assumed = True
                    quantity = 1.0
                if sign > 0:
                    sold_units += quantity
                else:
                    returned_units += quantity
                cost = _cost(item, catalog, context.costs)
                if cost is None:
                    missing.add(_key(item))
                else:
                    cogs += sign * quantity * cost
                    if not estimated_quantity:
                        confirmed_cogs += sign * quantity * cost
                    covered_units += quantity
        rows.append(
            {
                "amount": amount,
                "revenue": revenue,
                "commission": commission,
                "logistics": logistics,
                "other": amount - revenue - commission - logistics,
                "cogs": cogs,
                "confirmed_cogs": confirmed_cogs,
                "sold_units": sold_units,
                "returned_units": returned_units,
                "covered_units": covered_units,
            }
        )
    frame = _frame(rows, {key: pl.Float64 for key in rows[0]})
    totals = {key: _sum(frame, key) for key in frame.columns}
    gross = _sum(frame.filter(pl.col("revenue") > 0), "revenue")
    refunds = -_sum(frame.filter(pl.col("revenue") < 0), "revenue")
    tax_valid = _valid(context.tax_rate) and 0 <= float(context.tax_rate) <= 100
    taxes = (
        max(totals["revenue"], 0) * float(context.tax_rate) / 100
        if tax_valid and revenue_complete
        else None
    )
    can_estimate = (
        ledger_complete
        and revenue_complete
        and not missing
        and not unknown_units
        and taxes is not None
        and not invalid_inputs
        and not source_incomplete
    )
    complete = can_estimate and not quantity_assumed
    estimated_profit = totals["amount"] - totals["cogs"] - taxes if can_estimate else None
    profit = estimated_profit if complete else None
    warnings = []
    if missing:
        warnings.append(f"Нет себестоимости {len(missing)} SKU; чистая прибыль не рассчитана.")
    if quantity_assumed:
        warnings.append(
            f"API {_marketplace(context)} не передал количество части позиций и сопоставить отправления нельзя: принято 1 шт. только для оценки; точная себестоимость и прибыль недоступны."
        )
    if source_incomplete:
        warnings.append(
            "Финансовый отчёт содержит неполные или неоднозначные поля; точная прибыль не рассчитана."
        )
    if unknown_units:
        warnings.append(
            "В части продаж нет состава или корректного количества; себестоимость неполная."
        )
    if not ledger_complete or not revenue_complete or invalid_inputs:
        warnings.append("В операциях отсутствуют или некорректны денежные поля; итоги неполные.")
    if not tax_valid:
        warnings.append("Налоговая ставка должна быть от 0 до 100%.")
    warnings += [
        "Налог — упрощённая оценка от положительной выручки за вычетом возвратов; налоговый режим и льготы не учтены.",
        f"Прибыль основана на операциях {_marketplace(context)}; внешние расходы и выплаты поставщикам не учтены. Реклама отдельно повторно не вычитается.",
    ]
    if any(row.get("source") == "finance_accrual_by_day" for row in transactions):
        warnings.append(
            "Выручка/комиссия взяты из структуры commission начислений API; итог total_amount не уменьшается ими повторно. До проверки живых данных трактовка разбивки предварительная."
        )
    if refunds:
        warnings.append(
            "Себестоимость возвращённых товаров восстанавливается; брак, утилизация и изменение закупочной цены отдельно не учтены."
        )
    facts = {
        "available": True,
        "revenue": totals["revenue"],
        "gross_revenue": gross,
        "refund_revenue": refunds,
        "commission": -totals["commission"],
        "logistics": -totals["logistics"],
        "other": -totals["other"],
        "ledger_net": totals["amount"],
        "cogs": totals["cogs"]
        if not missing
        and not unknown_units
        and not quantity_assumed
        and not (_is_wb(context) and source_incomplete)
        else None,
        "known_cogs": totals["confirmed_cogs"],
        "estimated_cogs": totals["cogs"],
        "taxes": taxes,
        "profit": profit,
        "estimated_profit": estimated_profit if quantity_assumed else None,
        "profit_estimated": quantity_assumed,
        "missing_cost_skus": sorted(missing),
        "sold_units": totals["sold_units"],
        "returned_units": totals["returned_units"],
        "cost_coverage": _ratio(
            totals["covered_units"], totals["sold_units"] + totals["returned_units"], 100
        ),
        "operation_count": len(rows),
        "ledger_complete": ledger_complete,
        "source_complete": not source_incomplete,
    }
    lines = [
        f"Продажи: {_money(gross)}; возвраты: {_money(refunds)}",
        f"Выручка после возвратов: {_money(facts['revenue'])}",
        f"Комиссия: {_money(facts['commission'])}",
        f"Логистика: {_money(facts['logistics'])}",
        f"Прочие списания/начисления: {_money(facts['other'])}",
        f"Итог операций {_marketplace(context)}: {_money(facts['ledger_net'])}",
        f"Себестоимость: {_money(facts['cogs'])}",
        f"Налог ({context.tax_rate}%): {_money(taxes)}",
        f"Чистая прибыль: {_money(profit)}",
    ]
    if quantity_assumed and estimated_profit is not None:
        lines.append(f"Оценка прибыли при 1 шт. на неизвестную позицию: {_money(estimated_profit)}")
    return _report(context, "Финансы", lines, facts, warnings)


POST_SCHEMA = {
    "posting": pl.String,
    "status": pl.String,
    "date": pl.String,
    "sku": pl.String,
    "name": pl.String,
    "category": pl.String,
    "quantity": pl.Float64,
    "price": pl.Float64,
    "value": pl.Float64,
    "price_known": pl.Boolean,
}


def _postings(context: AnalysisContext) -> tuple[pl.DataFrame, list[dict], list[str]]:
    catalog = _catalog(context)
    rows: list[dict] = []
    posts: dict[str, dict] = {}
    warnings = []
    for i, posting in enumerate(context.snapshot.postings):
        if not _during(posting, context, ("in_process_at", "created_at", "date")):
            continue
        identifier = str(posting.get("posting_number") or posting.get("order_id") or f"unknown-{i}")
        posts[identifier] = posting
    for identifier, posting in posts.items():
        observed = next(
            (
                _date(posting.get(key))
                for key in ("in_process_at", "created_at", "date")
                if _date(posting.get(key))
            ),
            None,
        )
        status = str(posting.get("status", "unknown")).lower()
        if not posting.get("products"):
            warnings.append(
                "У части отправлений нет состава товаров; количество товаров и суммы неполные."
            )
        for item in posting.get("products") or []:
            product = catalog.get(_key(item), {})
            category = str(
                product.get("category")
                or product.get("category_name")
                or item.get("category")
                or (
                    f"Категория {product['description_category_id']}"
                    if product.get("description_category_id")
                    else "Без категории"
                )
            )
            quantity = _num(item.get("quantity"), 1)
            if quantity <= 0:
                warnings.append("Позиции с некорректным количеством не учтены.")
                continue
            known = _valid(item.get("price")) and _num(item.get("price")) >= 0
            price = _num(item.get("price")) if known else 0.0
            rows.append(
                {
                    "posting": identifier,
                    "status": status,
                    "date": str(observed or "Неизвестно"),
                    "sku": _canonical_key(item, catalog),
                    "name": str(item.get("name") or product.get("name") or _key(item)),
                    "category": category,
                    "quantity": quantity,
                    "price": price,
                    "value": quantity * price,
                    "price_known": known,
                }
            )
    if any(not row["price_known"] for row in rows):
        warnings.append("Часть цен не передана; денежные показатели по отправлениям неполные.")
    if any(row["date"] == "Неизвестно" for row in rows):
        warnings.append(
            "У части отправлений нет даты; они включены в текущую выборку без проверки периода."
        )
    return _frame(rows, POST_SCHEMA), list(posts.values()), list(dict.fromkeys(warnings))


def _refund_units(context: AnalysisContext) -> tuple[float, bool]:
    quantities = []
    estimated = False
    for transaction in _transactions(context):
        if _num(transaction.get("accruals_for_sale")) < 0:
            for item in transaction.get("items") or []:
                estimated |= item.get("quantity") is None
                quantities.append({"quantity": max(_num(item.get("quantity"), 1), 0)})
    frame = _frame(quantities, {"quantity": pl.Float64})
    return _sum(frame, "quantity"), estimated


EVENT_SCHEMA = {
    "sale_id": pl.String,
    "date": pl.String,
    "sku": pl.String,
    "name": pl.String,
    "category": pl.String,
    "status": pl.String,
    "quantity": pl.Float64,
    "value": pl.Float64,
    "price_known": pl.Boolean,
}


def _sale_events(context: AnalysisContext) -> tuple[pl.DataFrame, bool, list[str]]:
    """WB sales are observed sale/return events, independent of order creation dates."""
    source = getattr(context.snapshot, "sales", [])
    complete = _source_complete(context, "sales", bool(source))
    catalog = _catalog(context)
    rows = []
    warnings = []
    seen = set()
    for i, event in enumerate(source):
        observed = _date(event.get("date"))
        status = str(event.get("status", "")).lower()
        quantity = _num(event.get("quantity"))
        if observed is None or quantity <= 0 or status not in (DELIVERED | RETURNED):
            warnings.append(
                "События продаж без корректной даты, количества или статуса пропущены; итоги неполные."
            )
            complete = False
            continue
        if not context.period.start <= observed <= context.period.end:
            continue
        sale_id = str(event.get("sale_id") or f"event-{i}")
        if sale_id in seen:
            warnings.append("Повторяющиеся идентификаторы событий продаж исключены.")
            continue
        seen.add(sale_id)
        product = catalog.get(_key(event), {})
        category = str(
            product.get("category")
            or product.get("category_name")
            or event.get("category")
            or "Без категории"
        )
        raw_price = event.get("sale_price", event.get("price"))
        known = _valid(raw_price) and _num(raw_price) >= 0
        if not known:
            warnings.append(
                "Часть цен выкупов/возвратов отсутствует; денежные итоги событий неполные."
            )
        rows.append(
            {
                "sale_id": sale_id,
                "date": observed.isoformat(),
                "sku": _canonical_key(event, catalog),
                "name": str(event.get("name") or product.get("name") or _key(event)),
                "category": category,
                "status": "delivered" if status in DELIVERED else "returned",
                "quantity": quantity,
                "value": _num(raw_price) * quantity if known else 0,
                "price_known": known,
            }
        )
    if not complete:
        warnings.append(
            "Источник продаж Wildberries недоступен или неполон; отсутствие событий не означает нулевые выкупы и возвраты."
        )
    return _frame(rows, EVENT_SCHEMA), complete, list(dict.fromkeys(warnings))


def _wb_sales(context: AnalysisContext) -> Report:
    orders, posts, warnings = _postings(context)
    events, complete, event_warnings = _sale_events(context)
    warnings += event_warnings
    orders_complete = _source_complete(context, "orders", bool(posts))
    orders_observed = orders_complete or bool(posts)
    sales_observed = complete or not events.is_empty()
    if not orders_observed and not sales_observed:
        return _report(
            context,
            "Продажи",
            ["Источники заказов и продаж Wildberries недоступны за выбранный период."],
            {
                "available": False,
                "orders": None,
                "delivered_units": None,
                "returned_units": None,
                "average_basket": None,
            },
            warnings,
        )
    delivered = events.filter(pl.col("status") == "delivered")
    returned = events.filter(pl.col("status") == "returned")
    units = _sum(delivered, "quantity") if complete or not delivered.is_empty() else None
    returned_units = _sum(returned, "quantity") if complete or not returned.is_empty() else None
    delivered_value = (
        _sum(delivered, "value") if units is not None and delivered["price_known"].all() else None
    )
    returned_value = (
        _sum(returned, "value")
        if returned_units is not None and returned["price_known"].all()
        else None
    )
    average_price = (
        _ratio(delivered_value, units)
        if delivered_value is not None and units is not None
        else None
    )
    statuses = [str(post.get("status", "unknown")).lower() for post in posts]
    cancelled = sum(status in CANCELLED for status in statuses) if orders_observed else None
    order_daily = orders.group_by("date").agg(pl.col("posting").n_unique().alias("orders"))
    sale_daily = events.group_by("date").agg(
        pl.col("quantity").filter(pl.col("status") == "delivered").sum().alias("delivered_units"),
        pl.col("quantity").filter(pl.col("status") == "returned").sum().alias("returned_units"),
    )
    daily_frame = order_daily.join(sale_daily, on="date", how="full", coalesce=True)
    for column, known in (
        ("orders", orders_complete),
        ("delivered_units", complete),
        ("returned_units", complete),
    ):
        if known:
            daily_frame = daily_frame.with_columns(pl.col(column).fill_null(0))
    daily = daily_frame.sort("date").to_dicts()
    order_count = len(posts) if orders_observed else None
    lines = [
        f"Заказов: {order_count if order_count is not None else 'нет данных'}; отменённых: {cancelled if cancelled is not None else 'нет данных'}",
        f"Выкуплено товаров по дате продажи: {f'{units:g}' if units is not None else 'нет данных'}",
        f"Возвраты по дате возврата: {f'{returned_units:g}' if returned_units is not None else 'нет данных'}",
        f"Выручка выкупов: {_money(delivered_value)}; возвраты: {_money(returned_value)}",
        f"Средняя цена выкупленного товара: {_money(average_price)}",
        "Динамика (заказы — дата заказа; выкупы/возвраты — дата события):",
    ]
    for row in daily[-10:]:
        lines.append(
            f"{row['date']}: заказов {row['orders'] if row['orders'] is not None else 'н/д'}; выкуплено {row['delivered_units'] if row['delivered_units'] is not None else 'н/д'}; возвратов {row['returned_units'] if row['returned_units'] is not None else 'н/д'}"
        )
    warnings.append(
        "Заказы и продажи — разные события и выборки. По строкам Statistics API нельзя восстановить общую корзину покупателя: средний чек корзины не рассчитывается; показана средняя цена товара."
    )
    if not orders_complete:
        warnings.append(
            "История заказов недоступна или неполна; число заказов и отмен не является итогом полного периода."
        )
    return _report(
        context,
        "Продажи",
        lines,
        {
            "available": True,
            "orders": order_count,
            "cancelled_orders": cancelled,
            "delivered_orders": None,
            "delivered_events": delivered.height if sales_observed else None,
            "delivered_units": units,
            "returned_units": returned_units,
            "delivered_value": delivered_value,
            "returned_value": returned_value,
            "average_basket": None,
            "average_unit_price": average_price,
            "daily": daily,
            "sales_complete": complete,
            "orders_complete": orders_complete,
            "sales_basis": "event_date",
        },
        warnings,
    )


def _sales(context: AnalysisContext) -> Report:
    if _is_wb(context):
        return _wb_sales(context)
    frame, posts, warnings = _postings(context)
    if not posts:
        return _report(
            context, "Продажи", ["Отправления за период не получены."], {"available": False}
        )
    status_rows = [{"status": str(post.get("status", "unknown")).lower()} for post in posts]
    status_counts = _frame(status_rows, {"status": pl.String}).group_by("status").len()
    counts = {row["status"]: row["len"] for row in status_counts.to_dicts()}
    delivered = frame.filter(pl.col("status").is_in(DELIVERED))
    delivered_count = sum(counts.get(status, 0) for status in DELIVERED)
    returned, estimated = _refund_units(context)
    return_source = "финансовые операции"
    if not returned:
        returned = _sum(frame.filter(pl.col("status").is_in(RETURNED)), "quantity")
        return_source = "статусы отправлений"
        warnings.append(
            "Возвраты известны только по загруженным статусам/операциям; отмены не считаются возвратами."
        )
    if estimated:
        warnings.append("Количество возвратов оценочное: 1 шт. для элемента операции без quantity.")
    cancelled = sum(counts.get(status, 0) for status in CANCELLED)
    pending = (
        len(posts) - delivered_count - cancelled - sum(counts.get(status, 0) for status in RETURNED)
    )
    delivery_revenue = _sum(delivered, "value")
    delivered_compositions = delivered["posting"].n_unique()
    basket = (
        _ratio(delivery_revenue, delivered_count)
        if delivered["price_known"].all() and delivered_compositions == delivered_count
        else None
    )
    daily = (
        frame.group_by("date")
        .agg(
            pl.col("posting").n_unique().alias("orders"),
            pl.col("quantity")
            .filter(pl.col("status").is_in(DELIVERED))
            .sum()
            .alias("delivered_units"),
        )
        .sort("date")
        .to_dicts()
    )
    lines = [
        f"Заказы/отправления: {len(posts)}",
        f"Выкупленные отправления: {delivered_count}; товаров: {_sum(delivered, 'quantity'):g}",
        f"Отменённые: {cancelled}; в работе: {pending}",
        f"Возвраты товаров: {returned:g} ({return_source})",
        f"Средний чек выкупленного отправления: {_money(basket)}",
        "Динамика по дате создания отправления:",
    ]
    lines += [
        f"{row['date']}: заказов {row['orders']}, выкуплено товаров {row['delivered_units']:g}"
        for row in daily[-10:]
    ]
    warnings.append(
        "Выкупы — текущий статус delivered у созданных в выбранный период отправлений; это не число доставок по дате фактической доставки."
    )
    facts = {
        "available": True,
        "orders": len(posts),
        "delivered_orders": delivered_count,
        "delivered_units": _sum(delivered, "quantity"),
        "cancelled_orders": cancelled,
        "pending_orders": pending,
        "returned_units": returned,
        "average_basket": basket,
        "delivered_value": delivery_revenue,
        "daily": daily,
        "status_counts": counts,
    }
    return _report(context, "Продажи", lines, facts, warnings)


def _ads(context: AnalysisContext) -> Report:
    rows = []
    invalid = False
    for row in context.ads:
        if not _during(row, context, ("date",)):
            continue
        invalid |= any(
            row.get(key) is not None and not _valid(row[key])
            for key in ("spend", "revenue", "orders", "clicks", "impressions")
        )
        rows.append(
            {
                "campaign": str(
                    row.get("campaign_id") or row.get("campaign_name") or "Без кампании"
                ),
                "name": str(row.get("campaign_name") or row.get("campaign_id") or "Без кампании"),
                "sku": str(row.get("sku") or "Без SKU"),
                **{
                    key: max(_num(row.get(key)), 0)
                    for key in ("spend", "revenue", "orders", "clicks", "impressions")
                },
            }
        )
    if not rows:
        return _report(
            context,
            "Реклама",
            [
                "Подключите "
                + ("API рекламы Wildberries" if _is_wb(context) else "Ozon Performance API")
                + " или импортируйте CSV со статистикой рекламы."
            ],
            {"available": False, "drr": None},
        )
    frame = _frame(
        rows,
        {
            "campaign": pl.String,
            "name": pl.String,
            "sku": pl.String,
            **{key: pl.Float64 for key in ("spend", "revenue", "orders", "clicks", "impressions")},
        },
    )
    totals = {
        key: _sum(frame, key) for key in ("spend", "revenue", "orders", "clicks", "impressions")
    }
    aggregates = [pl.col(key).sum() for key in totals]
    campaigns = (
        frame.group_by("campaign", "name").agg(aggregates).sort("spend", descending=True).to_dicts()
    )
    products = (
        frame.filter(pl.col("sku") != "Без SKU")
        .group_by("sku")
        .agg(aggregates)
        .sort("spend", descending=True)
        .to_dicts()
    )
    drr = _ratio(totals["spend"], totals["revenue"], 100)
    roas = _ratio(totals["revenue"], totals["spend"])
    inefficient = []
    for row in products:
        row["drr"] = _ratio(row["spend"], row["revenue"], 100)
        if row["spend"] > 0 and (row["revenue"] == 0 or row["spend"] >= row["revenue"]):
            inefficient.append(row)
    lines = [
        f"Расходы: {_money(totals['spend'])}; выручка по атрибуции: {_money(totals['revenue'])}",
        f"ДРР: {_pct(drr)}; ROAS: {f'{roas:.2f}' if roas is not None else 'н/д'}",
        f"Заказов: {totals['orders']:g}; CTR: {_pct(_ratio(totals['clicks'], totals['impressions'], 100))}",
        "Кампании:",
    ]
    for row in campaigns[:8]:
        lines.append(
            f"{row['name'][:70]}: {_money(row['spend'])}, ДРР {_pct(_ratio(row['spend'], row['revenue'], 100))}"
        )
    if inefficient:
        lines += [
            "Товары для проверки (расходы ≥ выручки или нет продаж):",
            *[
                f"SKU {row['sku']}: расходы {_money(row['spend'])}, выручка {_money(row['revenue'])}"
                for row in inefficient[:5]
            ],
        ]
    else:
        lines.append(
            "Для оценки убыточности товаров нужны маржа и себестоимость; ДРР сам по себе не определяет прибыль."
        )
    warnings = [
        "Рекламная выручка атрибутирована платформой; не прибавляйте её к финансовой выручке. Кампании могут пересекаться по атрибуции.",
        "Порог расходы ≥ выручки — сигнал для проверки, а не расчёт рентабельности с комиссией, логистикой и налогом.",
    ]
    if totals["spend"] > 0 and totals["revenue"] == 0:
        warnings.append(
            "При нулевой выручке ДРР не определён; расходы не окупились в загруженной атрибуции."
        )
    if invalid:
        warnings.append("Некорректные числовые поля пропущены; итоги неполные.")
    facts = {
        "available": True,
        **totals,
        "drr": drr,
        "roas": roas,
        "campaigns": campaigns,
        "products": products,
        "inefficient_products": inefficient,
    }
    return _report(context, "Реклама", lines, facts, warnings)


def _seo(context: AnalysisContext, page: int) -> Report:
    catalog = _catalog(context)
    content_unavailable = (
        _is_wb(context) and getattr(context.snapshot, "sources", {}).get("products") is False
    )
    rows: dict[str, dict] = {}
    if not content_unavailable:
        for product in context.snapshot.products:
            rows[_canonical_key(product, catalog)] = dict(product)
    for imported in sorted(context.seo, key=lambda row: str(row.get("date", ""))):
        observed = _date(imported.get("date"))
        if observed and observed > context.period.end:
            continue
        target_key = _canonical_key(imported, catalog)
        rows[target_key] = {**rows.get(target_key, {}), **imported}
    if not rows:
        return _report(
            context,
            "SEO",
            [
                "Карточки Wildberries не загружены. Разрешите токену доступ «Контент» или импортируйте SEO CSV с названиями и описаниями."
                if content_unavailable
                else "Карточки товаров не загружены. Для позиций и ключевых слов импортируйте SEO CSV."
            ],
            {"available": False},
        )
    assessed = []
    for key, row in rows.items():
        title = str(row.get("title") or row.get("name") or "")
        issues = []
        if not title:
            issues.append("нет названия")
        elif len(title) < 30:
            issues.append("короткое название: проверьте тип, бренд и характеристики")
        elif len(title) > 150:
            issues.append("длинное название: проверьте читаемость")
        if "description" in row and not str(row.get("description") or "").strip():
            issues.append("описание пустое")
        if "images" in row and not row.get("images") and not row.get("primary_image"):
            issues.append("нет изображений")
        description_known = "description" in row
        keywords = str(row.get("keywords") or "").strip()
        keyword_phrases = [
            phrase.strip() for phrase in re.split(r"[,;]", keywords) if phrase.strip()
        ]
        title_words = set(re.findall(r"\w+", title.casefold()))
        uncovered_keywords = [
            phrase
            for phrase in keyword_phrases
            if not set(re.findall(r"\w+", phrase.casefold())).issubset(title_words)
        ]
        position = (
            _num(row.get("position"))
            if _valid(row.get("position")) and _num(row.get("position")) > 0
            else None
        )
        previous = (
            _num(row.get("previous_position"))
            if _valid(row.get("previous_position")) and _num(row.get("previous_position")) > 0
            else None
        )
        assessed.append(
            {
                "sku": key,
                "title": title,
                "title_length": len(title),
                "issues": issues,
                "description_known": description_known,
                "keywords": keywords,
                "keywords_not_in_title": uncovered_keywords,
                "position": position,
                "position_change": previous - position if previous and position else None,
            }
        )
    frame = _frame(
        [
            {
                "issues": len(row["issues"]),
                "description_known": row["description_known"],
                "rank_known": row["position"] is not None,
            }
            for row in assessed
        ],
        {"issues": pl.Int64, "description_known": pl.Boolean, "rank_known": pl.Boolean},
    )
    pages = max(math.ceil(len(assessed) / PAGE_SIZE), 1)
    page = max(0, min(page, pages - 1))
    lines = [
        f"Карточек: {len(assessed)}; с рекомендациями: {frame.filter(pl.col('issues') > 0).height}",
        f"Страница {page + 1}/{pages}",
    ]
    for row in assessed[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        text = f"SKU {row['sku']} · {row['title'][:65] or 'Без названия'}\n"
        text += "; ".join(row["issues"]) or "Название: формальных замечаний нет"
        text += f"; описание: {'доступно' if row['description_known'] else 'не передано'}"
        text += f"; ключи: {row['keywords'][:65] if row['keywords'] else 'не загружены'}"
        if row["keywords_not_in_title"]:
            text += f"; не встречаются в названии: {', '.join(row['keywords_not_in_title'])[:75]}"
        text += f"; позиция: {row['position']:g}" if row["position"] else "; позиция: не загружена"
        if row["position_change"] is not None:
            text += f" (изменение {row['position_change']:+g}; плюс = выше)"
        lines.append(text)
    return _report(
        context,
        "SEO",
        lines,
        {
            "available": True,
            "page": page,
            "pages": pages,
            "products": assessed,
            "card_count": len(assessed),
            "audit_source": "csv_only" if content_unavailable else "catalog_and_csv",
            "ranks_available": frame.filter(pl.col("rank_known")).height,
        },
        [
            f"Проверки названия — эвристики, не гарантии ранжирования. Подключённые API {_marketplace(context)} не сообщают поисковые запросы и позиции; они доступны только из загруженных данных.",
            "Ключи проверяются по буквальным словам без морфологии; включайте только релевантные запросы, избегайте повторов и переспама.",
            "Не переданное описание/фото не означает пустую карточку.",
        ]
        + (
            ["Доступ к «Контенту» Wildberries отсутствует; проверены только карточки из SEO CSV."]
            if content_unavailable
            else []
        ),
    )


def _competitors(context: AnalysisContext, page: int) -> Report:
    source = [
        row
        for row in context.competitors
        if not _date(row.get("date")) or _date(row.get("date")) <= context.period.end
    ]
    if not source:
        return _report(
            context,
            "Конкуренты",
            [
                f"Импортируйте CSV наблюдений: sku, competitor_sku, price, rating, reviews, position, date. Подключённые API {_marketplace(context)} не предоставляют данные конкурентов."
            ],
            {"available": False},
        )
    history: dict[tuple[str, str], list[dict]] = {}
    for i, row in enumerate(source):
        key = (
            str(row.get("sku") or ""),
            str(row.get("competitor_sku") or row.get("name") or f"row-{i}"),
        )
        history.setdefault(key, []).append(row)
    catalog = _catalog(context)
    rows = []
    for (sku, competitor), observations in history.items():
        observations.sort(key=lambda row: str(row.get("date", "")))
        latest = observations[-1]
        price = (
            _num(latest.get("price"))
            if _valid(latest.get("price")) and _num(latest.get("price")) >= 0
            else None
        )
        own = catalog.get(sku, {})
        own_price = _num(own.get("price")) if _valid(own.get("price")) else None
        positions = [
            (_date(row.get("date")), _num(row["position"]))
            for row in observations
            if _valid(row.get("position")) and _num(row.get("position")) > 0
        ]
        change = (
            positions[-2][1] - positions[-1][1]
            if len(positions) > 1 and positions[-2][0] != positions[-1][0]
            else None
        )
        rows.append(
            {
                "sku": sku,
                "competitor_sku": competitor,
                "name": str(latest.get("name") or competitor),
                "category": str(latest.get("category") or "Не указана"),
                "date": str(latest.get("date") or "Не указана"),
                "price": price,
                "own_price": own_price,
                "price_gap_pct": _ratio(own_price - price, price, 100)
                if own_price is not None and price is not None
                else None,
                "rating": _num(latest.get("rating")) if _valid(latest.get("rating")) else None,
                "reviews": _num(latest.get("reviews")) if _valid(latest.get("reviews")) else None,
                "position": positions[-1][1] if positions else None,
                "position_change": change,
            }
        )
    frame = _frame(
        [{key: row[key] for key in ("category", "competitor_sku", "price")} for row in rows],
        {"category": pl.String, "competitor_sku": pl.String, "price": pl.Float64},
    )
    assortment = (
        frame.group_by("category")
        .agg(
            pl.col("competitor_sku").n_unique().alias("products"),
            pl.col("price").median().alias("median_price"),
        )
        .sort("products", descending=True)
        .to_dicts()
    )
    pages = max(math.ceil(len(rows) / PAGE_SIZE), 1)
    page = max(0, min(page, pages - 1))
    lines = [
        f"Наблюдаемых конкурентных товаров: {frame['competitor_sku'].n_unique()}; страница {page + 1}/{pages}",
        "Ассортимент в загруженной выборке: "
        + "; ".join(f"{r['category']}: {r['products']}" for r in assortment[:6]),
    ]
    for row in rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        line = f"{row['name'][:55]}: {_money(row['price'])}; рейтинг {row['rating'] if row['rating'] is not None else 'н/д'}; отзывов {row['reviews'] if row['reviews'] is not None else 'н/д'}"
        if row["price_gap_pct"] is not None:
            line += f"; ваша цена {row['price_gap_pct']:+.1f}% к конкуренту"
        if row["position"]:
            line += f"; позиция {row['position']:g}"
        if row["position_change"] is not None:
            line += f" ({row['position_change']:+g}, плюс = выше)"
        lines.append(line + f"; дата {row['date']}")
    return _report(
        context,
        "Конкуренты",
        lines,
        {
            "available": True,
            "page": page,
            "pages": pages,
            "products": rows,
            "assortment": assortment,
        },
        [
            "Только импортированные наблюдения, не весь рынок. Сравнивайте позиции по одинаковому запросу, региону и условиям поиска."
        ],
    )


def _stock_rows(context: AnalysisContext) -> tuple[pl.DataFrame, list[str]]:
    source = context.snapshot.stocks or [p for p in context.snapshot.products if p.get("stocks")]
    catalog = _catalog(context)
    rows = []
    warnings = []
    for product in source:
        details = catalog.get(_key(product), {})
        sku = _canonical_key(product, catalog)
        stocks = product.get("stocks")
        if isinstance(stocks, list):
            entries = stocks
        elif isinstance(stocks, dict):
            entries = [stocks]
        elif any(key in product for key in ("present", "quantity", "fbo", "fbs")):
            entries = [product]
        else:
            warnings.append("У части товаров нет количественных остатков.")
            continue
        for stock in entries:
            raw_present = stock.get(
                "present", stock.get("quantity", _num(stock.get("fbo")) + _num(stock.get("fbs")))
            )
            present = max(_num(raw_present), 0)
            reserved = max(_num(stock.get("reserved")), 0)
            reserved_known = bool(
                stock.get("reserved_known", product.get("reserved_known", not _is_wb(context)))
            )
            if not _valid(raw_present) or (
                stock.get("reserved") is not None and not _valid(stock["reserved"])
            ):
                warnings.append("Некорректные остатки не учтены; покрытие неполное.")
                continue
            if reserved_known and reserved > present:
                warnings.append(
                    "Резерв превышает остаток у части товаров; доступный остаток принят равным 0."
                )
            rows.append(
                {
                    "sku": sku,
                    "name": str(details.get("name") or product.get("name") or sku),
                    "present": present,
                    "reserved": reserved if reserved_known else None,
                    "reserved_known": reserved_known,
                    "available": max(present - reserved, 0) if reserved_known else present,
                }
            )
    frame = _frame(
        rows,
        {
            "sku": pl.String,
            "name": pl.String,
            "present": pl.Float64,
            "reserved": pl.Float64,
            "reserved_known": pl.Boolean,
            "available": pl.Float64,
        },
    )
    return frame.group_by("sku").agg(
        pl.col("name").first(),
        pl.col("present").sum(),
        pl.when(pl.col("reserved_known").all())
        .then(pl.col("reserved").sum())
        .otherwise(None)
        .alias("reserved"),
        pl.col("reserved_known").all(),
        pl.col("available").sum(),
    ), list(dict.fromkeys(warnings))


def _stocks(context: AnalysisContext, page: int) -> Report:
    stock, warnings = _stock_rows(context)
    if stock.is_empty():
        return _report(
            context, "Остатки", ["Остатки не загружены."], {"available": False}, warnings
        )
    demand_complete = True
    if _is_wb(context):
        events, demand_complete, event_warnings = _sale_events(context)
        warnings += event_warnings
        demand = events.group_by("sku").agg(
            pl.when(pl.col("status") == "returned")
            .then(-pl.col("quantity"))
            .otherwise(pl.col("quantity"))
            .sum()
            .clip(lower_bound=0)
            .alias("sold_units")
        )
    else:
        postings, _, post_warnings = _postings(context)
        warnings += post_warnings
        demand = (
            postings.filter(pl.col("status").is_in(DELIVERED))
            .group_by("sku")
            .agg(pl.col("quantity").sum().alias("sold_units"))
        )
    joined = stock.join(demand, on="sku", how="left").with_columns(
        pl.col("sold_units").fill_null(0)
    )
    if not demand_complete:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("sold_units"))
    days = max((context.period.end - context.period.start).days + 1, 1)
    joined = joined.with_columns((pl.col("sold_units") / days).alias("daily_demand"))
    joined = joined.with_columns(
        pl.when(pl.col("daily_demand") > 0)
        .then(pl.col("available") / pl.col("daily_demand"))
        .otherwise(None)
        .alias("days_cover"),
        (pl.col("daily_demand") * 30 - pl.col("available"))
        .clip(lower_bound=0)
        .ceil()
        .alias("recommended_supply"),
    )
    joined = joined.with_columns(
        pl.when(pl.col("daily_demand").is_null())
        .then(pl.lit("Недостаточно данных о продажах"))
        .when((pl.col("daily_demand") > 0) & (pl.col("days_cover") < 14))
        .then(pl.lit("Дефицит"))
        .when(
            (pl.col("days_cover") > 60) | ((pl.col("sold_units") == 0) & (pl.col("available") > 0))
        )
        .then(pl.lit("Излишек/нет наблюдаемых продаж"))
        .otherwise(pl.lit("Достаточно/нет спроса в выборке"))
        .alias("status")
    )
    rows = joined.sort(["recommended_supply", "available"], descending=[True, True]).to_dicts()
    pages = max(math.ceil(len(rows) / PAGE_SIZE), 1)
    page = max(0, min(page, pages - 1))
    reserved_known = stock["reserved_known"].all()
    reserve_text = f"{_sum(stock, 'reserved'):g} шт." if reserved_known else "не передан API"
    lines = [
        f"Товаров: {len(rows)}; доступно {_sum(stock, 'available'):g} шт.; резерв {reserve_text}",
        f"Страница {page + 1}/{pages}; ориентир поставки — запас на 30 дней",
    ]
    for row in rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        cover = (
            f"{row['days_cover']:.1f} дней"
            if row["days_cover"] is not None
            else "нет данных о спросе"
            if row["daily_demand"] is None
            else "нет наблюдаемого спроса"
        )
        supply = (
            f"{row['recommended_supply']:g} шт."
            if row["recommended_supply"] is not None
            else "нет данных"
        )
        lines.append(
            f"{row['name'][:55]} (SKU {row['sku']}): {row['available']:g} шт.; {cover}; {row['status']}; поставить {supply}"
        )
    warnings += [
        (
            "WB quantity уже является доступным остатком; товары в пути к клиенту или от клиента не вычитаются как резерв. Резерв отдельно не передан. Спрос — выкупы минус возвраты по датам событий, не менее 0."
            if _is_wb(context)
            else "Доступно = present − reserved (не менее 0). Оборачиваемость оценена по текущему остатку и выкупам созданных в период отправлений."
        ),
        "Дефицит: менее 14 дней, излишек: более 60 дней или нет выкупов. Прогноз не учитывает срок поставки, сезонность, ожидаемые поступления и неполную историю.",
    ]
    return _report(
        context,
        "Остатки",
        lines,
        {
            "available": True,
            "page": page,
            "pages": pages,
            "products": rows,
            "available_units": _sum(stock, "available"),
            "reserved_units": _sum(stock, "reserved") if reserved_known else None,
            "sales_complete": demand_complete,
            "shortage_count": joined.filter(pl.col("status") == "Дефицит").height
            if demand_complete
            else None,
            "excess_count": joined.filter(
                pl.col("status") == "Излишек/нет наблюдаемых продаж"
            ).height
            if demand_complete
            else None,
        },
        warnings,
    )


def _wb_categories(context: AnalysisContext, page: int) -> Report:
    orders, posts, warnings = _postings(context)
    events, sales_complete, sale_warnings = _sale_events(context)
    warnings += sale_warnings
    orders_complete = _source_complete(context, "orders", bool(posts))
    if orders.is_empty() and events.is_empty() and not orders_complete and not sales_complete:
        return _report(
            context,
            "Категории",
            ["Данные заказов и продаж за период не получены."],
            {"available": False},
            warnings,
        )
    ordered = orders.group_by("category").agg(
        pl.col("posting").n_unique().alias("orders"),
        pl.col("quantity").sum().alias("ordered_units"),
        pl.col("value").sum().alias("ordered_turnover"),
        pl.col("price_known").all().alias("order_prices_complete"),
    )
    sold = events.group_by("category").agg(
        pl.col("quantity").filter(pl.col("status") == "delivered").sum().alias("buyouts"),
        pl.col("quantity").filter(pl.col("status") == "returned").sum().alias("returns"),
        pl.col("value").filter(pl.col("status") == "delivered").sum().alias("buyout_revenue"),
        pl.col("value").filter(pl.col("status") == "returned").sum().alias("return_revenue"),
        pl.col("price_known").all().alias("sale_prices_complete"),
    )
    aggregate = ordered.join(sold, on="category", how="full", coalesce=True)
    order_columns = ("orders", "ordered_units", "ordered_turnover")
    sale_columns = ("buyouts", "returns", "buyout_revenue", "return_revenue")
    for column in (*order_columns, *sale_columns):
        if orders_complete if column in order_columns else sales_complete:
            aggregate = aggregate.with_columns(pl.col(column).fill_null(0))
    aggregate = aggregate.with_columns(
        pl.when(pl.col("order_prices_complete").not_())
        .then(None)
        .otherwise(pl.col("ordered_turnover"))
        .alias("ordered_turnover"),
        *[
            pl.when(pl.col("sale_prices_complete").not_())
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("buyout_revenue", "return_revenue")
        ],
    )
    aggregate = aggregate.with_columns(
        (pl.col("buyout_revenue") - pl.col("return_revenue")).alias("net_realized_turnover")
    )
    rows = aggregate.sort(
        ["buyout_revenue", "category"], descending=[True, False], nulls_last=True
    ).to_dicts()
    pages = max(math.ceil(len(rows) / PAGE_SIZE), 1)
    page = max(0, min(page, pages - 1))
    lines = [f"Категорий: {len(rows)} · страница {page + 1}/{pages}"]
    for row in rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        orders_text = str(row["orders"]) if row["orders"] is not None else "нет данных"
        buyouts = f"{row['buyouts']:g}" if row["buyouts"] is not None else "нет данных"
        returns = f"{row['returns']:g}" if row["returns"] is not None else "нет данных"
        lines.append(
            f"{row['category']}: заказов {orders_text}; выкуплено {buyouts} шт.; возвратов {returns} шт.; оборот заказов {_money(row['ordered_turnover'])}; выкупы {_money(row['buyout_revenue'])}; после возвратов {_money(row['net_realized_turnover'])}"
        )
    warnings += [
        "Заказы сгруппированы по дате заказа, выкупы и возвраты — по дате фактического события. Отменённые заказы входят в оборот заказов и не считаются возвратами.",
        "Денежный оборот — цена события × количество; комиссия, логистика, налоги и прочие удержания сюда не входят. Начисления и выплаты WB смотрите в финансах.",
    ]
    if not orders_complete:
        warnings.append(
            "Источник заказов недоступен или неполон; отсутствующие количества не заменяются нулями."
        )
    if any(row["category"] == "Без категории" for row in rows):
        warnings.append(
            "Не для всех SKU известна категория: такие товары объединены в «Без категории»."
        )
    return _report(
        context,
        "Категории",
        lines,
        {
            "available": True,
            "page": page,
            "pages": pages,
            "categories": rows,
            "page_categories": rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE],
            "ordered_turnover": _sum(aggregate, "ordered_turnover")
            if orders_complete and aggregate["ordered_turnover"].null_count() == 0
            else None,
            "buyout_revenue": _sum(aggregate, "buyout_revenue")
            if sales_complete and aggregate["buyout_revenue"].null_count() == 0
            else None,
            "net_realized_turnover": _sum(aggregate, "net_realized_turnover")
            if sales_complete and aggregate["net_realized_turnover"].null_count() == 0
            else None,
            "orders_complete": orders_complete,
            "sales_complete": sales_complete,
            "sales_basis": "event_date",
        },
        warnings,
    )


def _categories(context: AnalysisContext, page: int) -> Report:
    if _is_wb(context):
        return _wb_categories(context, page)
    frame, _, warnings = _postings(context)
    if frame.is_empty():
        return _report(
            context,
            "Категории",
            ["Для анализа категорий нужны отправления с составом и каталог товаров."],
            {"available": False},
        )
    aggregate = (
        frame.group_by("category")
        .agg(
            pl.col("posting").n_unique().alias("orders"),
            pl.col("quantity").sum().alias("ordered_units"),
            pl.col("quantity").filter(pl.col("status").is_in(DELIVERED)).sum().alias("buyouts"),
            pl.col("posting")
            .filter(pl.col("status").is_in(DELIVERED))
            .n_unique()
            .alias("buyout_orders"),
            pl.col("value").sum().alias("ordered_turnover"),
            pl.col("value").filter(pl.col("status").is_in(DELIVERED)).sum().alias("buyout_revenue"),
            pl.col("price_known").all().alias("prices_complete"),
        )
        .sort(["ordered_turnover", "category"], descending=[True, False])
    )
    rows = aggregate.to_dicts()
    pages = max(math.ceil(len(rows) / PAGE_SIZE), 1)
    page = max(0, min(page, pages - 1))
    lines = [f"Категорий: {len(rows)} · страница {page + 1}/{pages}"]
    for row in rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        lines.append(
            f"{row['category']}: заказов {row['orders']}; выкуплено {row['buyouts']:g} шт. ({row['buyout_orders']} отправлений); оборот заказов {_money(row['ordered_turnover'])}; выкупы {_money(row['buyout_revenue'])}"
        )
    warnings += [
        "Оборот — цена × количество в созданных отправлениях, включая отменённые; выкупы — текущий статус delivered. Это не финансовые начисления Ozon.",
        "Заказ с товарами разных категорий учитывается в каждой категории. Возвраты и комиссии не вычтены из оборота.",
    ]
    if frame.filter(pl.col("category") == "Без категории").height:
        warnings.append(
            "Не для всех SKU найден каталог/категория; они объединены в «Без категории»."
        )
    return _report(
        context,
        "Категории",
        lines,
        {
            "available": True,
            "page": page,
            "pages": pages,
            "categories": rows,
            "page_categories": rows[page * PAGE_SIZE : (page + 1) * PAGE_SIZE],
            "ordered_turnover": _sum(aggregate, "ordered_turnover"),
            "buyout_revenue": _sum(aggregate, "buyout_revenue"),
        },
        warnings,
    )


def build_report(kind: str, context: AnalysisContext, page: int = 0) -> Report:
    """Build a plain-text report and structured, JSON-serializable evidence for AI/export."""
    if context.period.end < context.period.start:
        raise ValueError("Конец периода должен быть не раньше начала")
    builders = {"finance": _finance, "sales": _sales, "ads": _ads}
    paginated = {
        "seo": _seo,
        "competitors": _competitors,
        "stocks": _stocks,
        "categories": _categories,
    }
    if kind in builders:
        return builders[kind](context)
    if kind in paginated:
        return paginated[kind](context, page)
    raise ValueError(f"Неизвестный вид отчёта: {kind}")
