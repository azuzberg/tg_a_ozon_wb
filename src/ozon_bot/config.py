import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# При желании вставьте один общий ключ Claude здесь. Предпочтительнее .env.
# Никогда не добавляйте настоящий ключ в Git.
CLAUDE_API_KEY = ""
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(ValueError):
    """Configuration messages safe to display without revealing secret values."""


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    database_path: Path
    claude_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    session_ttl_minutes: int = 60
    claude_max_tokens: int = 4096
    claude_thinking_mode: str = "disabled"

    @classmethod
    def load(cls) -> "Settings":
        # PyCharm may use another working directory. Read .env next to run_bot.py.
        load_dotenv(PROJECT_ROOT / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise SettingsError("Заполните TELEGRAM_BOT_TOKEN в .env рядом с run_bot.py.")
        database = Path(os.getenv("DB_PATH", "").strip() or "data/ozon_bot.sqlite3").expanduser()
        if not database.is_absolute():
            database = PROJECT_ROOT / database
        try:
            ttl = int(os.getenv("SESSION_TTL_MINUTES", "60"))
        except ValueError:
            raise SettingsError(
                "SESSION_TTL_MINUTES должен быть целым числом от 5 до 1440."
            ) from None
        if not 5 <= ttl <= 1440:
            raise SettingsError("SESSION_TTL_MINUTES должен быть от 5 до 1440.")
        try:
            max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
        except ValueError:
            raise SettingsError(
                "CLAUDE_MAX_TOKENS должен быть целым числом от 256 до 16384."
            ) from None
        if not 256 <= max_tokens <= 16384:
            raise SettingsError("CLAUDE_MAX_TOKENS должен быть от 256 до 16384.")
        thinking_mode = os.getenv("CLAUDE_THINKING_MODE", "disabled").strip().lower()
        if thinking_mode not in ("disabled", "auto"):
            raise SettingsError("CLAUDE_THINKING_MODE: допустимы disabled и auto.")
        return cls(
            token,
            database.resolve(),
            os.getenv("CLAUDE_API_KEY", "").strip() or CLAUDE_API_KEY,
            os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            ttl,
            max_tokens,
            thinking_mode,
        )
