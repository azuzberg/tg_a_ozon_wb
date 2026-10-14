import gzip
import io
import json
import zipfile
from datetime import date

import httpx
import pytest

from ozon_bot.performance import PerformanceAPIError, PerformanceClient, _parse_report

START = date(2026, 1, 1)
END = date(2026, 1, 31)


def report_row(day="01.01.2026", **overrides):
    return {
        "date": day,
        "sku": "123456789",
        "views": "1200",
        "clicks": "30",
        "moneySpent": "1 234,56",
        "orders": "2",
        "revenue": "5 000,00",
        "modelOrders": "1",
        "modelRevenue": "1 000,00",
        **overrides,
    }


@pytest.mark.asyncio
async def test_real_async_oauth_campaign_and_json_report_contract():
    calls = []
    polls = 0

    async def handler(request):
        nonlocal polls
        calls.append(request.url.path)
        if request.url.path == "/api/client/token":
            assert json.loads(request.content) == {
                "client_id": "ad-client",
                "client_secret": "private-secret",
                "grant_type": "client_credentials",
            }
            assert "Authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "access_token": "private-bearer",
                    "expires_in": 1800,
                    "token_type": "Bearer",
                },
            )
        assert request.headers["Authorization"] == "Bearer private-bearer"
        if request.url.path == "/api/client/campaign":
            assert request.url.params["page"] == "1"
            return httpx.Response(200, json={"list": [{"id": "48949", "title": "Товары"}]})
        if request.url.path == "/api/client/statistics/json":
            assert json.loads(request.content) == {
                "campaigns": ["48949"],
                "dateFrom": "2026-01-01",
                "dateTo": "2026-01-31",
                "groupBy": "DATE",
            }
            return httpx.Response(200, json={"UUID": "abc-123", "vendor": False})
        if request.url.path == "/api/client/statistics/abc-123":
            polls += 1
            return httpx.Response(
                200,
                json={
                    "UUID": "abc-123",
                    "state": "IN_PROGRESS" if polls == 1 else "OK",
                    "link": "https://untrusted.example/report",  # This is never followed.
                },
            )
        assert request.url.path == "/api/client/statistics/report"
        assert request.url.params["UUID"] == "abc-123"
        return httpx.Response(
            200,
            json={
                "48949": {
                    "report": {"rows": [report_row()], "totals": {"moneySpent": "1234,56"}},
                    "campaign": {"id": "48949", "title": "Товары"},
                }
            },
        )

    client = PerformanceClient(
        "ad-client", "private-secret", transport=httpx.MockTransport(handler), poll_interval=0
    )
    async with client:
        rows, warnings = await client.load(START, END)
        assert rows == [
            {
                "date": "2026-01-01",
                "campaign_id": "48949",
                "campaign_name": "Товары",
                "sku": "123456789",
                "spend": 1234.56,
                "revenue": 5000.0,
                "orders": 2,
                "clicks": 30,
                "impressions": 1200,
            }
        ]
        assert "атрибутированные" in warnings[0]
        assert polls == 2
        assert calls.count("/api/client/token") == 1
    assert client._access_token == ""
    assert client._client_secret == ""


CSV = (
    "; Кампания по продвижению товаров, период 01.01.2026-31.01.2026\n"
    "Дата;sku;Название товара;Показы;Клики;Расход, Р, с НДС;Заказы;Выручка, Р\n"
    "01.01.2026;123456789;Товар;1200;30;1234,56;2;5000,00\n"
    ";Всего;;1200;30;1234,56;2;5000,00\n"
).encode("utf-8-sig")


@pytest.mark.parametrize("compressed", [False, True])
def test_csv_preamble_russian_headers_totals_and_gzip(compressed):
    rows = _parse_report(gzip.compress(CSV) if compressed else CSV, {"48949": "Товары"}, START, END)
    assert len(rows) == 1
    assert rows[0]["spend"] == 1234.56
    assert rows[0]["impressions"] == 1200


def test_documented_zip_with_one_csv_per_campaign():
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("48949.csv", CSV)
        archive.writestr("48852.csv", CSV)
    rows = _parse_report(target.getvalue(), {"48949": "Товары", "48852": "Другая"}, START, END)
    assert {row["campaign_id"] for row in rows} == {"48949", "48852"}
    assert sum(row["spend"] for row in rows) == 2469.12


def test_missing_metric_never_becomes_fake_zero():
    row = report_row()
    del row["revenue"]
    data = json.dumps({"48949": {"report": {"rows": [row]}}}).encode()
    with pytest.raises(PerformanceAPIError, match="не содержит поле revenue"):
        _parse_report(data, {"48949": "Товары"}, START, END)


def test_product_called_total_is_not_silently_discarded():
    data = json.dumps({"48949": {"report": {"rows": [report_row(title="Total")]}}}).encode()
    rows = _parse_report(data, {"48949": "Товары"}, START, END)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_campaign_pagination_does_not_drop_archived_campaigns():
    pages = []

    async def handler(request):
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "token", "expires_in": 1800})
        page = int(request.url.params["page"])
        pages.append(page)
        assert "state" not in request.url.params
        indices = range(1, 101) if page == 1 else range(101, 102)
        return httpx.Response(
            200,
            json={
                "list": [{"id": str(index), "title": str(index)} for index in indices],
            },
        )

    async with PerformanceClient(
        "client", "secret", transport=httpx.MockTransport(handler)
    ) as client:
        campaigns = await client._campaigns()
    assert len(campaigns) == 101
    assert pages == [1, 2]


@pytest.mark.asyncio
async def test_expired_bearer_refreshes_once_without_repeating_stats_creation():
    tokens = 0
    reads = 0

    async def handler(request):
        nonlocal tokens, reads
        if request.url.path.endswith("/token"):
            tokens += 1
            return httpx.Response(200, json={"access_token": f"token-{tokens}", "expires_in": 1800})
        reads += 1
        if reads == 1:
            return httpx.Response(401)
        assert request.headers["Authorization"] == "Bearer token-2"
        return httpx.Response(200, json={"list": []})

    async with PerformanceClient(
        "client", "secret", transport=httpx.MockTransport(handler)
    ) as client:
        await client.validate()
    assert tokens == 2
    assert reads == 2


@pytest.mark.parametrize("value", ["NaN", "Infinity", "not-a-number", None, True])
def test_invalid_ad_money_is_rejected(value):
    data = json.dumps({"48949": {"report": {"rows": [report_row(moneySpent=value)]}}}).encode()
    with pytest.raises(PerformanceAPIError, match="некорректное значение поля spend"):
        _parse_report(data, {"48949": "Товары"}, START, END)


def test_missing_campaign_or_unrequested_dates_do_not_inflate_totals():
    data = json.dumps({"48949": {"report": {"rows": [report_row()]}}}).encode()
    with pytest.raises(PerformanceAPIError, match="все запрошенные кампании"):
        _parse_report(data, {"48949": "А", "48852": "Б"}, START, END)
    outside = json.dumps({"48949": {"report": {"rows": [report_row(day="2025-12-31")]}}}).encode()
    with pytest.raises(PerformanceAPIError, match="за пределами"):
        _parse_report(outside, {"48949": "А"}, START, END)


@pytest.mark.asyncio
async def test_campaign_and_date_limits_split_reports_without_overlapping_days():
    creations = []

    async def handler(request):
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "token", "expires_in": 1800})
        if request.url.path == "/api/client/campaign":
            return httpx.Response(
                200,
                json={
                    "list": [{"id": str(index), "title": str(index)} for index in range(1, 12)],
                },
            )
        if request.url.path == "/api/client/statistics/json":
            payload = json.loads(request.content)
            creations.append(payload)
            assert len(payload["campaigns"]) <= 10
            assert (
                date.fromisoformat(payload["dateTo"]) - date.fromisoformat(payload["dateFrom"])
            ).days <= 61
            return httpx.Response(200, json={"UUID": f"report-{len(creations)}"})
        if request.url.path.startswith("/api/client/statistics/report-"):
            return httpx.Response(200, json={"state": "OK"})
        current = creations[-1]
        return httpx.Response(
            200,
            json={
                campaign: {"report": {"rows": [report_row(day=current["dateFrom"])]}}
                for campaign in current["campaigns"]
            },
        )

    async with PerformanceClient(
        "client", "secret", transport=httpx.MockTransport(handler), poll_interval=0
    ) as client:
        rows, _ = await client.load(START, date(2026, 3, 4))  # 63 days.
    assert len(creations) == 4
    assert {item["dateFrom"] for item in creations} == {"2026-01-01", "2026-03-04"}
    assert len(rows) == 22


@pytest.mark.asyncio
async def test_service_error_does_not_expose_credentials_or_server_body():
    async def handler(request):
        return httpx.Response(401, json={"error": "private-secret private-client private-token"})

    async with PerformanceClient(
        "private-client", "private-secret", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(PerformanceAPIError) as captured:
            await client.validate()
        assert "private" not in str(captured.value)


@pytest.mark.asyncio
async def test_report_polling_error_does_not_return_partial_or_fabricated_results():
    async def handler(request):
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "token", "expires_in": 1800})
        if request.url.path.endswith("/campaign"):
            return httpx.Response(200, json={"list": [{"id": "48949", "title": "А"}]})
        if request.url.path.endswith("/json"):
            return httpx.Response(200, json={"UUID": "failed-report"})
        return httpx.Response(200, json={"state": "ERROR", "error": "do not expose raw error"})

    async with PerformanceClient(
        "client", "secret", transport=httpx.MockTransport(handler), poll_interval=0
    ) as client:
        with pytest.raises(PerformanceAPIError, match="не смог сформировать"):
            await client.load(START, END)


@pytest.mark.asyncio
async def test_bounded_polling_does_not_hang_forever():
    async def handler(request):
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "token", "expires_in": 1800})
        if request.url.path.endswith("/campaign"):
            return httpx.Response(200, json={"list": [{"id": "48949", "title": "А"}]})
        if request.url.path.endswith("/json"):
            return httpx.Response(200, json={"UUID": "slow-report"})
        return httpx.Response(200, json={"state": "IN_PROGRESS"})

    async with PerformanceClient(
        "client",
        "secret",
        transport=httpx.MockTransport(handler),
        poll_interval=0,
        report_timeout=0.01,
    ) as client:
        with pytest.raises(PerformanceAPIError, match="ещё не готов"):
            await client.load(START, END)


def test_gzip_expansion_size_is_bounded(monkeypatch):
    monkeypatch.setattr("ozon_bot.performance.MAX_REPORT_BYTES", 500)
    data = gzip.compress(b"x" * 1000)
    with pytest.raises(PerformanceAPIError, match="слишком большой"):
        _parse_report(data, {"48949": "А"}, START, END)
