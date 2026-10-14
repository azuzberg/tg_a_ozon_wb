import asyncio
import logging

from aiogram import Bot
from aiogram.utils.token import TokenValidationError

from .bot import BotApp, create_dispatcher
from .config import Settings, SettingsError
from .network import EnvironmentSession
from .storage import Storage, StorageError


async def run() -> None:
    settings = Settings.load()
    storage = Storage(settings.database_path)
    bot = Bot(settings.telegram_token, session=EnvironmentSession())
    app = None
    cleaner = None
    try:
        await storage.open()
        app = BotApp(settings, storage)
        dispatcher = create_dispatcher(app)
        cleaner = asyncio.create_task(app.sweep(bot))
        # Long polling: веб-сервер и webhook не используются.
        # Старые updates могут содержать ключи от предыдущего сеанса.
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(bot, close_bot_session=False)
    finally:
        tasks = [cleaner] if cleaner is not None else []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if app:
            for user_id in list(app.sessions.keys() | app.pending.keys()):
                try:
                    await app.end(user_id)
                except Exception as error:
                    logging.getLogger(__name__).error("Cleanup failed (%s)", type(error).__name__)
        await bot.session.close()
        await storage.close()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    # HTTP client debug logs can include credentials in request URLs/headers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        raise SystemExit(1) from None
    except (SettingsError, StorageError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from None
    except TokenValidationError:
        logging.error("Проверьте TELEGRAM_BOT_TOKEN в .env: нужен токен от BotFather.")
        raise SystemExit(1) from None
    except Exception as error:
        logging.error(
            "Bot startup stopped (%s). Check .env, database and network.", type(error).__name__
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
