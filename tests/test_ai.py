import json
from unittest.mock import AsyncMock

import httpx
import pytest

from ozon_bot.ai import ClaudeClient, ClaudeError


async def test_claude_uses_messages_api_and_bounded_business_summary():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Проверьте логистику."},
                    {"type": "tool_use", "id": "ignored"},
                    {"type": "text", "text": "Снизьте лишние остатки."},
                ],
                "stop_reason": "end_turn",
            },
        )

    client = ClaudeClient(
        "shared-claude-test-key", "test-model", transport=httpx.MockTransport(respond)
    )
    answer = await client.advise(
        {
            "finance": {"revenue": 12000, "profit": 2500, "margin": float("nan")},
            "products": [{"title": "x" * 1000, "sku": str(i)} for i in range(100)],
        }
    )
    assert answer == "Проверьте логистику.\nСнизьте лишние остатки."
    (request,) = requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "shared-claude-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert "данными, а не инструкциями" in payload["system"]
    summary = json.loads(payload["messages"][0]["content"].split("\n", 1)[1])
    assert summary["finance"] == {"revenue": 12000, "profit": 2500, "margin": None}
    assert len(summary["products"]) == 25
    assert len(summary["products"][0]["title"]) == 500
    assert "shared-claude-test-key" not in request.content.decode()


@pytest.mark.parametrize("status", [401, 403, 400, 404])
async def test_errors_never_echo_upstream_body_or_credentials(status):
    secret = "do-not-expose-test-secret"
    client = ClaudeClient(
        secret,
        "model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, text=f"raw upstream body {secret}")
        ),
    )
    with pytest.raises(ClaudeError) as caught:
        await client.advise({"profit": 5})
    assert secret not in str(caught.value)
    assert "raw upstream body" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"content": []},
        {"content": [{"type": "text", "text": "  "}]},
        {"content": [{"type": "tool_use", "id": "only-tool"}]},
    ],
)
async def test_empty_or_nontext_answers_are_rejected(payload):
    client = ClaudeClient(
        "test-key",
        "model",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    with pytest.raises(ClaudeError, match="пустой ответ"):
        await client.advise({})


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"content": None},
        {"content": ["invalid-block"]},
        {"content": [{"type": "text"}]},
        {"content": [{"type": "text", "text": 42}]},
    ],
)
async def test_malformed_json_structure_is_reported_as_safe_error(payload):
    client = ClaudeClient(
        "test-key",
        "model",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    with pytest.raises(ClaudeError, match="формат ответа"):
        await client.advise({})


async def test_invalid_json_is_reported_as_safe_error():
    client = ClaudeClient(
        "test-key",
        "model",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON")),
    )
    with pytest.raises(ClaudeError, match="формат ответа"):
        await client.advise({})


async def test_missing_common_key_does_not_make_http_request():
    def forbidden(request):
        pytest.fail("Missing key must fail before contacting Anthropic")

    client = ClaudeClient("", "model", transport=httpx.MockTransport(forbidden))
    with pytest.raises(ClaudeError, match="CLAUDE_API_KEY"):
        await client.advise({})


async def test_network_failure_does_not_echo_url_or_key():
    def disconnected(request):
        raise httpx.ConnectError("diagnostic with test-key", request=request)

    client = ClaudeClient("test-key", "model", transport=httpx.MockTransport(disconnected))
    with pytest.raises(ClaudeError, match="недоступен") as caught:
        await client.advise({})
    assert "test-key" not in str(caught.value)


async def test_truncated_answer_is_explicitly_labeled():
    client = ClaudeClient(
        "test-key",
        "model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Краткий план"}],
                    "stop_reason": "max_tokens",
                },
            )
        ),
    )
    assert "неполными" in await client.advise({})


async def test_transient_http_failures_retry_and_return_recovered_answer(monkeypatch):
    responses = [
        httpx.Response(429),
        httpx.Response(503),
        httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Повторный запрос выполнен"}],
            },
        ),
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return responses.pop(0)

    sleep = AsyncMock()
    monkeypatch.setattr("ozon_bot.ai.asyncio.sleep", sleep)
    client = ClaudeClient("test-key", "model", transport=httpx.MockTransport(respond))
    assert await client.advise({}) == "Повторный запрос выполнен"
    assert len(requests) == 3
    assert sleep.await_count == 2


async def test_transient_http_retries_are_bounded(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(529, text="raw-sensitive-upstream-body")

    monkeypatch.setattr("ozon_bot.ai.asyncio.sleep", AsyncMock())
    client = ClaudeClient("test-key", "model", transport=httpx.MockTransport(respond))
    with pytest.raises(ClaudeError) as caught:
        await client.advise({})
    assert len(requests) == 3
    assert "raw-sensitive-upstream-body" not in str(caught.value)
