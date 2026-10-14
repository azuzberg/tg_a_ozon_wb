import ssl

import pytest
from aiohttp import web

from ozon_bot.network import EnvironmentSession


@pytest.mark.asyncio
async def test_telegram_transport_uses_environment_proxy_without_resolving_target(monkeypatch):
    requested = []

    async def proxy(request):
        requested.append(request.raw_path)
        return web.Response(text="proxy reached")

    application = web.Application()
    application.router.add_route("*", "/{path:.*}", proxy)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{port}")
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.delenv("no_proxy", raising=False)
    session = EnvironmentSession()
    try:
        client = await session.create_session()
        async with client.get("http://no-such-ozon-bot-host.invalid/ping") as response:
            assert response.status == 200
            assert await response.text() == "proxy reached"
        assert requested == ["http://no-such-ozon-bot-host.invalid/ping"]
        context = client.connector._ssl
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_invalid_supplied_ca_fails_instead_of_disabling_verification(monkeypatch, tmp_path):
    bundle = tmp_path / "bad-ca.pem"
    bundle.write_text("not a certificate")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    session = EnvironmentSession()
    try:
        with pytest.raises(ssl.SSLError):
            await session.create_session()
    finally:
        await session.close()
