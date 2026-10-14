"""Telegram transport that respects the cloud environment's HTTPS proxy and CA."""

import os

from aiogram.client.session.aiohttp import SERVER_SOFTWARE, AiohttpSession
from aiohttp import ClientSession
from aiohttp.hdrs import USER_AGENT

from . import __version__


class EnvironmentSession(AiohttpSession):
    async def create_session(self) -> ClientSession:
        if self._should_reset_connector:
            await self.close()
        if self._session is None or self._session.closed:
            context = self._connector_init["ssl"]
            if cafile := os.getenv("SSL_CERT_FILE"):
                context.load_verify_locations(cafile=cafile)
            if capath := os.getenv("SSL_CERT_DIR"):
                context.load_verify_locations(capath=capath)
            self._session = ClientSession(
                connector=self._connector_type(**self._connector_init),
                trust_env=True,
                headers={USER_AGENT: f"{SERVER_SOFTWARE} ozon-bot/{__version__}"},
            )
            self._should_reset_connector = False
        return self._session
