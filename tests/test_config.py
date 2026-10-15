import runpy
from pathlib import Path

import pytest

from ozon_bot import config


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "DB_PATH",
        "CLAUDE_API_KEY",
        "DATABASE_URL",
        "CLAUDE_MODEL",
        "CLAUDE_MAX_TOKENS",
        "CLAUDE_THINKING_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SESSION_TTL_MINUTES", "60")
    return tmp_path


def test_pycharm_reads_root_env_and_default_sqlite_path_from_another_directory(
    project, monkeypatch
):
    (project / ".env").write_text("TELEGRAM_BOT_TOKEN=test-token\n", encoding="utf-8")
    other = project / "another-working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    settings = config.Settings.load()
    assert settings.telegram_token == "test-token"
    assert settings.database_path == project / "data" / "ozon_bot.sqlite3"
    assert not settings.database_path.exists()  # Created by storage, not configuration.


def test_relative_db_path_is_project_relative_and_environment_has_priority(project, monkeypatch):
    (project / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=file-token\nDB_PATH=from-file.sqlite3\n", encoding="utf-8"
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "environment-token")
    monkeypatch.setenv("DB_PATH", "custom/local.sqlite3")
    monkeypatch.chdir(project.parent)
    settings = config.Settings.load()
    assert settings.telegram_token == "environment-token"
    assert settings.database_path == project / "custom" / "local.sqlite3"


def test_old_postgres_environment_does_not_require_server_for_sqlite(project, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DATABASE_URL", "postgresql://old-database-not-used")
    assert config.Settings.load().database_path == project / "data" / "ozon_bot.sqlite3"


def test_absolute_database_path_preserved(project, monkeypatch):
    target = project.parent / "outside-project.sqlite3"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DB_PATH", str(target))
    assert config.Settings.load().database_path == target


def test_missing_telegram_token_explains_local_env_setup(project):
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        config.Settings.load()


def test_selected_claude_model_and_budget_are_read_from_pycharm_env(project):
    (project / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=test-token\nCLAUDE_MODEL=claude-sonnet-5-5\n"
        "CLAUDE_MAX_TOKENS=8192\nCLAUDE_THINKING_MODE=disabled\n",
        encoding="utf-8",
    )
    settings = config.Settings.load()
    assert settings.claude_model == "claude-sonnet-5-5"
    assert settings.claude_max_tokens == 8192
    assert settings.claude_thinking_mode == "disabled"


def test_root_launcher_imports_sources_when_pycharm_working_directory_differs(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(root / "run_bot.py"), run_name="pycharm_import")
    assert namespace["PROJECT_ROOT"] == root
    assert callable(namespace["main"])
    assert not (tmp_path / "data").exists()
