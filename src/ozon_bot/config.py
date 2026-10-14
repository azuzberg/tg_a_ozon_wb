import os
from dataclasses import dataclass

from dotenv import load_dotenv

# При желании вставьте один общий ключ Claude здесь. Предпочтительнее .env.
# Никогда не добавляйте настоящий ключ в Git.
CLAUDE_API_KEY = ""


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    database_url: str
    claude_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    session_ttl_minutes: int = 60

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        database = os.getenv("DATABASE_URL", "").strip()
        if not token or not database:
            raise ValueError("Заполните TELEGRAM_BOT_TOKEN и DATABASE_URL в .env.")
        ttl = int(os.getenv("SESSION_TTL_MINUTES", "60"))
        if not 5 <= ttl <= 1440:
            raise ValueError("SESSION_TTL_MINUTES должен быть от 5 до 1440.")
        return cls(
            token,
            database,
            os.getenv("CLAUDE_API_KEY", "").strip() or CLAUDE_API_KEY,
            os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            ttl,
        )
