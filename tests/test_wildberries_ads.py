import base64
import json
from copy import deepcopy
from datetime import date

import httpx
import pytest

from ozon_bot import wildberries_ads as wb
from ozon_bot.wildberries_ads import WildberriesAdsClient, WildberriesAdsError, _normalize

START = date(2026, 9, 1)
END = date(2026, 9, 30)


def token(acc=3):
    payload = base64.urlsafe_b64encode(json.dumps({"acc": acc}).encode()).decode().rstrip("=")
    return f"header.{payload}.private-signature"


def metrics(spend=100, revenue=500, orders=1, clicks=10, views=100):
    return {"sum": spend, "sum_price": revenue, "orders": orders, "clicks": clicks, "views": views}


def stats(campaign_id=42, day="2026-09-01"):
    # The WB schema repeats totals at campaign/day/app levels. Only nms leaves count.
    return [
        {
            "advertId": campaign_id,
            "currency": "RUB",
            **metrics(150, 750, 3, 15, 150),
            "days": [
                {
                    "date": f"{day}T00:00:00Z",
                    **metrics(150, 750, 3, 15, 150),
                    "apps": [
                        {
                            "appType": 1,
                            **metrics(),
                            "nms": [{"nmId": 123, "name": "Товар", **metrics()}],
                        },
                        {
                            "appType": 32,
                            **metrics(50, 250, 2, 5, 50),
                            "nms": [
                                {"nmId": 123, "name": "Товар", **metrics(50, 250, 2, 5, 50)},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


@pytest.fixture(autouse=True)
def no_network_waits(monkeypatch):
    wb._RATE_SLOTS.clear()
    monkeypatch.setattr(wb, "_STANDARD_INTERVALS", {wb.COUNT: 0, wb.DETAILS: 0, wb.STATS: 0})
    monkeypatch.setattr(wb, "_BASIC_INTERVALS", {wb.COUNT: 0, wb.DETAILS: 0, wb.STATS: 0})


@pytest.mark.asyncio
async def test_current_readonly_contract_host_token_statuses_names_and_nested_totals():
    paths = []

    async def handler(request):
        assert request.method == "GET"
        assert request.url.host == "advert-api.wildberries.ru"
        assert request.headers["Authorization"] == token()
        assert "Client-Id" not in request.headers
        paths.append(request.url.path)
        if request.url.path == wb.COUNT:
            return httpx.Response(
                200,
                json={
                    "all": 3,
                    "adverts": [
                        {"type": 9, "status": 7, "advert_list": [{"advertId": 42}]},
                        {"type": 9, "status": 4, "advert_list": [{"advertId": 43}]},
                        {"type": 9, "status": -1, "advert_list": [{"advertId": 44}]},
                    ],
                },
            )
        assert request.url.params["ids"] == "42"
        if request.url.path == wb.DETAILS:
            return httpx.Response(
                200, json={"adverts": [{"id": 42, "settings": {"name": "Обувь"}}]}
            )
        assert request.url.path == wb.STATS
        assert request.url.params["beginDate"] == "2026-09-01"
        assert request.url.params["endDate"] == "2026-09-30"
        return httpx.Response(200, json=stats())

    client = WildberriesAdsClient(token(), transport=httpx.MockTransport(handler))
    async with client:
        await client.validate()
        rows, warnings = await client.load(START, END)
        assert rows == [
            {
                "date": "2026-09-01",
                "campaign_id": "42",
                "campaign_name": "Обувь",
                "sku": "123",
                "spend": 150.0,
                "revenue": 750.0,
                "orders": 3,
                "clicks": 15,
                "impressions": 150,
            }
        ]
        assert "Не включены 2 кампаний" in warnings[0]
        assert "атрибутированных" in warnings[-1]
        assert paths == [wb.COUNT, wb.DETAILS, wb.STATS]
    assert client._key == ""
    assert client._counts is None


def test_multiple_products_and_platforms_aggregate_without_adding_parent_totals():
    payload = stats()
    payload[0]["days"][0]["apps"][1]["nms"][0]["nmId"] = 456
    rows, warnings = _normalize(payload, {"42": "Обувь"}, START, END)
    assert len(rows) == 2
    assert {row["sku"] for row in rows} == {"123", "456"}
    assert sum(row["spend"] for row in rows) == 150
    assert sum(row["revenue"] for row in rows) == 750
    assert warnings == []


def test_incomplete_sku_breakdown_preserves_reported_totals_without_inventing_attribution():
    payload = stats()
    payload[0]["days"][0]["apps"][1]["nms"] = []
    rows, warnings = _normalize(payload, {"42": "Обувь"}, START, END)
    assert sum(row["spend"] for row in rows) == 150
    assert rows[1]["sku"] == ""
    assert "не распределяются" in warnings[0]


def test_incomplete_app_breakdown_uses_day_only():
    payload = stats()
    payload[0]["days"][0]["apps"] = []
    rows, warnings = _normalize(payload, {"42": "Обувь"}, START, END)
    assert len(rows) == 1
    assert rows[0]["sku"] == ""
    assert rows[0]["spend"] == 150
    assert warnings


@pytest.mark.parametrize(
    "field,value",
    [("sum_price", None), ("sum", "NaN"), ("clicks", 1.5), ("orders", True), ("views", -1)],
)
def test_unknown_or_invalid_metric_is_not_fabricated_as_zero(field, value):
    payload = stats()
    payload[0]["days"][0]["apps"][0]["nms"][0][field] = value
    with pytest.raises(WildberriesAdsError, match="отсутствует или некорректно"):
        _normalize(payload, {"42": "Обувь"}, START, END)


def test_non_ruble_accounts_never_look_like_rubles():
    payload = stats()
    payload[0]["currency"] = "BYN"
    with pytest.raises(WildberriesAdsError, match="RUB"):
        _normalize(payload, {"42": "Обувь"}, START, END)


@pytest.mark.parametrize("duplicate", ["day", "campaign", "app", "sku"])
def test_duplicate_entities_never_inflate_totals(duplicate):
    payload = stats()
    day = payload[0]["days"][0]
    if duplicate == "campaign":
        payload.append(deepcopy(payload[0]))
    elif duplicate == "day":
        payload[0]["days"].append(deepcopy(day))
    elif duplicate == "app":
        day["apps"].append(deepcopy(day["apps"][0]))
    else:
        day["apps"][0]["nms"].append(deepcopy(day["apps"][0]["nms"][0]))
    with pytest.raises(WildberriesAdsError, match="повторя|повторяющуюся"):
        _normalize(payload, {"42": "Обувь"}, START, END)


def test_missing_campaigns_are_explicitly_unknown_and_mismatched_days_fail():
    rows, warnings = _normalize([], {"42": "Обувь"}, START, END)
    assert rows == []
    assert "показатели неизвестны" in warnings[0]
    payload = stats()
    payload[0]["sum_price"] = 1000
    with pytest.raises(WildberriesAdsError, match="не совпадают"):
        _normalize(payload, {"42": "Обувь"}, START, END)


@pytest.mark.asyncio
async def test_maximum_50_campaigns_and_31_days_are_chunked_without_date_overlap():
    calls = []

    async def handler(request):
        if request.url.path == wb.COUNT:
            return httpx.Response(
                200,
                json={
                    "all": 51,
                    "adverts": [
                        {
                            "type": 9,
                            "status": 11,
                            "advert_list": [{"advertId": i} for i in range(1, 52)],
                        },
                    ],
                },
            )
        ids = [int(value) for value in request.url.params["ids"].split(",")]
        assert len(ids) <= 50
        if request.url.path == wb.DETAILS:
            return httpx.Response(
                200,
                json={
                    "adverts": [{"id": value, "settings": {"name": str(value)}} for value in ids],
                },
            )
        begin, end = request.url.params["beginDate"], request.url.params["endDate"]
        assert (date.fromisoformat(end) - date.fromisoformat(begin)).days <= 30
        calls.append((begin, end, ids))
        return httpx.Response(200, json=[stats(value, begin)[0] for value in ids])

    async with WildberriesAdsClient(token(), transport=httpx.MockTransport(handler)) as client:
        rows, _ = await client.load(START, date(2026, 10, 2))
    assert len(calls) == 4
    assert {(call[0], call[1]) for call in calls} == {
        ("2026-09-01", "2026-10-01"),
        ("2026-10-02", "2026-10-02"),
    }
    assert len(rows) == 102


@pytest.mark.asyncio
async def test_authentication_error_never_exposes_token_or_response_body():
    async def handler(request):
        return httpx.Response(403, json={"detail": f"secret={token()}"})

    async with WildberriesAdsClient(token(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WildberriesAdsError, match="Продвижение") as captured:
            await client.validate()
        assert "private-signature" not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [None, True, -1, "1", 42])
async def test_incomplete_campaign_inventory_is_not_misreported_as_empty(count):
    async def handler(request):
        return httpx.Response(200, json={"all": count, "adverts": None})

    async with WildberriesAdsClient(token(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WildberriesAdsError, match="неполный список"):
            await client.validate()


@pytest.mark.asyncio
async def test_redirect_does_not_send_authorization_to_another_host():
    calls = []

    async def handler(request):
        calls.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://elsewhere.example/secret"})

    async with WildberriesAdsClient(token(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WildberriesAdsError, match="302"):
            await client.validate()
    assert calls == ["advert-api.wildberries.ru"]


@pytest.mark.asyncio
async def test_fullstats_pacing_is_shared_between_clients_and_basic_tokens_are_conservative(
    monkeypatch,
):
    now = [100.0]
    slept = []

    async def sleep(delay):
        slept.append(delay)
        now[0] += delay

    monkeypatch.setattr(wb.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(wb.asyncio, "sleep", sleep)
    monkeypatch.setattr(wb, "_STANDARD_INTERVALS", {wb.COUNT: 0.2, wb.DETAILS: 0.2, wb.STATS: 20})
    monkeypatch.setattr(wb, "_BASIC_INTERVALS", {wb.COUNT: 900, wb.DETAILS: 3600, wb.STATS: 3600})
    async with WildberriesAdsClient(token()) as first, WildberriesAdsClient(token()) as second:
        await first._pace(wb.STATS)
        await second._pace(wb.STATS)
        assert slept == [20]
    async with WildberriesAdsClient(token(1)) as basic:
        await basic._pace(wb.STATS)
        with pytest.raises(WildberriesAdsError, match="раз в час"):
            await basic._pace(wb.STATS)


@pytest.mark.asyncio
async def test_long_wb_retry_header_stops_without_retrying_early():
    requests = []

    async def handler(request):
        requests.append(request.url.path)
        return httpx.Response(429, headers={"X-Ratelimit-Retry": "3600"})

    async with WildberriesAdsClient(token(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WildberriesAdsError, match="длительного ожидания"):
            await client.validate()
    assert len(requests) == 1


def test_test_token_is_rejected_before_any_production_request():
    with pytest.raises(WildberriesAdsError, match="песочнице"):
        WildberriesAdsClient(token(2))
