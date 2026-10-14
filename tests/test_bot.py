import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, GetFile, SendDocument, SendMessage
from aiogram.types import CallbackQuery, Chat, Document, File, Message, Update, User

from ozon_bot.bot import BotApp, create_dispatcher, parse_period
from ozon_bot.config import Settings
from ozon_bot.domain import Snapshot
from ozon_bot.imports import SAMPLE_CSV
from ozon_bot.ozon import OzonAPIError
from ozon_bot.performance import PerformanceAPIError


class RecordingTelegram(BaseSession):
    """Exercises the aiogram dispatcher without a Telegram network connection."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.file_payload = b""
        self.fail_delete = False
        self.delete_started = asyncio.Event()
        self.release_delete = None

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, DeleteMessage):
            self.delete_started.set()
            if self.release_delete is not None:
                await self.release_delete.wait()
            if self.fail_delete:
                raise TelegramBadRequest(method=method, message="message cannot be deleted")
            return True
        if isinstance(method, AnswerCallbackQuery):
            return True
        if isinstance(method, GetFile):
            return File(
                file_id=method.file_id,
                file_unique_id="unique",
                file_size=len(self.file_payload),
                file_path="imports/upload.csv",
            )
        if isinstance(method, (SendMessage, SendDocument)):
            return Message(
                message_id=len(self.calls),
                date=datetime.now(UTC),
                chat=Chat(id=method.chat_id, type="private"),
                text=getattr(method, "text", None),
            )
        raise AssertionError(f"Unmocked Telegram operation: {type(method).__name__}")

    async def stream_content(
        self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True
    ):
        yield self.file_payload

    @property
    def messages(self):
        return [call for call in self.calls if isinstance(call, SendMessage)]


class FakeStorage:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.touched = []
        self.taxes = []
        self.imports = []
        self.touch_started = asyncio.Event()
        self.release_touch = None

    async def create_session(self, *args):
        self.created.append(args)

    async def delete_session(self, session_id):
        self.deleted.append(session_id)

    async def touch(self, *args):
        self.touched.append(args)
        self.touch_started.set()
        if self.release_touch is not None:
            await self.release_touch.wait()

    async def set_tax(self, *args):
        self.taxes.append(args)

    async def save_import(self, *args):
        self.imports.append(args)


class FakeSeller:
    def __init__(self):
        self.credentials = []
        self.validations = 0
        self.loads = 0
        self.reject = False
        self.block_load = False
        self.load_started = asyncio.Event()
        self.load_cancelled = False
        self.release_validate = None
        self.validate_started = asyncio.Event()

    def __call__(self, client_id, api_key):
        self.credentials.append((client_id, api_key))
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def validate(self):
        self.validations += 1
        self.validate_started.set()
        if self.release_validate is not None:
            await self.release_validate.wait()
        if self.reject:
            raise OzonAPIError("Ozon отклонил доступ.")

    async def load(self, start, end):
        self.loads += 1
        self.load_started.set()
        if self.block_load:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.load_cancelled = True
                raise
        return Snapshot()


class FakePerformance:
    def __init__(self):
        self.credentials = []
        self.reject = False

    def __call__(self, client_id, secret):
        self.credentials.append((client_id, secret))
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def validate(self):
        if self.reject:
            raise PerformanceAPIError("Нет доступа к рекламе.")

    async def load(self, start, end):
        return [], []


class FakeAI:
    def __init__(self):
        self.facts = []

    async def advise(self, facts):
        self.facts.append(facts)
        return "Проверьте комиссии и себестоимость."


@dataclass
class Harness:
    app: BotApp
    dispatcher: object
    bot: Bot
    telegram: RecordingTelegram
    storage: FakeStorage
    seller: FakeSeller
    performance: FakePerformance
    ai: FakeAI
    update_id: int = 0

    async def text(self, text, *, user_id=101, chat_id=None, chat_type="private"):
        self.update_id += 1
        message = self.message(user_id, chat_id, chat_type, text=text)
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=self.update_id, message=message)
        )
        return message

    def message(self, user_id, chat_id=None, chat_type="private", **kwargs):
        return Message(
            message_id=self.update_id + 100,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id or user_id, type=chat_type),
            from_user=User(id=user_id, is_bot=False, first_name="Тест"),
            **kwargs,
        )

    async def callback(self, data, *, user_id=101, chat_id=None, chat_type="private"):
        self.update_id += 1
        query = CallbackQuery(
            id=str(self.update_id),
            chat_instance="test-chat-instance",
            from_user=User(id=user_id, is_bot=False, first_name="Тест"),
            message=self.message(user_id, chat_id, chat_type),
            data=data,
        )
        return await self.dispatcher.feed_update(
            self.bot, Update(update_id=self.update_id, callback_query=query)
        )

    async def login(self, *, user_id=101):
        for value in ("/start", "Алексей", "123456", "seller-test-secret"):
            await self.text(value, user_id=user_id)
        return self.app.sessions[user_id]

    async def document(self, payload, filename="costs.csv", *, user_id=101):
        self.telegram.file_payload = payload
        self.update_id += 1
        message = self.message(
            user_id,
            document=Document(
                file_id="csv-file",
                file_unique_id="csv-unique",
                file_name=filename,
                file_size=len(payload),
                mime_type="text/csv",
            ),
        )
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=self.update_id, message=message)
        )


@pytest.fixture
async def h():
    telegram, storage = RecordingTelegram(), FakeStorage()
    seller, performance, ai = FakeSeller(), FakePerformance(), FakeAI()
    bot = Bot("123456:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO", session=telegram)
    app = BotApp(
        Settings(bot.token, "unused-test.sqlite3", "shared-ai-test-key"),
        storage,
        seller_factory=seller,
        performance_factory=performance,
        ai_client=ai,
    )
    harness = Harness(app, create_dispatcher(app), bot, telegram, storage, seller, performance, ai)
    yield harness
    for user_id in list(app.sessions):
        await app.end(user_id)
    await bot.session.close()


async def test_menu_gated_until_validated_login_and_keys_do_not_enter_database(h):
    await h.callback("report:finance")
    assert "Сначала" in h.telegram.messages[-1].text
    assert h.seller.loads == 0
    await h.text("/start")
    await h.text("Алексей")
    await h.text("123456")
    assert not h.app.sessions
    assert not h.storage.created
    h.seller.reject = True
    await h.text("wrong-key-value")
    assert not h.app.sessions
    assert h.app.pending[101].step == "api_key"
    assert "Попробуйте" in h.telegram.messages[-1].text
    assert not h.storage.created
    h.seller.reject = False
    key_message = await h.text("seller-test-secret")
    session = h.app.sessions[101]
    assert session.api_key == "seller-test-secret"
    assert h.telegram.messages[-1].reply_markup
    assert "выберите анализ" in h.telegram.messages[-1].text
    assert len(h.storage.created) == 1
    assert h.storage.created[0][1:4] == (101, 101, "Алексей")
    assert "seller-test-secret" not in repr(h.storage.created)
    assert "123456" not in repr(h.storage.created)
    assert any(
        isinstance(call, DeleteMessage) and call.message_id == key_message.message_id
        for call in h.telegram.calls
    )
    assert "seller-test-secret" not in repr(session)


async def test_group_messages_and_callbacks_cannot_login_or_dispatch_analysis(h):
    await h.text("/start", chat_id=-900, chat_type="group")
    await h.callback("report:sales", chat_id=-900, chat_type="group")
    assert not h.app.pending
    assert not h.app.sessions
    assert not h.telegram.calls
    assert h.seller.loads == 0


async def test_credential_deletion_failure_informs_user_without_echoing_secret(h):
    for value in ("/start", "Алексей", "123456"):
        await h.text(value)
    h.telegram.fail_delete = True
    await h.text("seller-test-secret")
    assert 101 in h.app.sessions
    assert any("Удалите его вручную" in call.text for call in h.telegram.messages)
    assert all("seller-test-secret" not in call.text for call in h.telegram.messages)


async def test_second_login_secret_is_deleted_while_first_is_being_validated(h):
    for value in ("/start", "Алексей", "123456"):
        await h.text(value)
    h.seller.release_validate = asyncio.Event()
    first_task = asyncio.create_task(h.text("first-seller-test-secret"))
    await asyncio.wait_for(h.seller.validate_started.wait(), 2)
    second_message = await h.text("second-seller-test-secret")
    h.seller.release_validate.set()
    await asyncio.wait_for(first_task, 2)
    assert any(
        isinstance(call, DeleteMessage) and call.message_id == second_message.message_id
        for call in h.telegram.calls
    )
    assert h.app.sessions[101].api_key == "first-seller-test-secret"
    assert h.seller.validations == 1
    assert all("second-seller-test-secret" not in call.text for call in h.telegram.messages)


async def test_logout_erases_both_credentials_and_imports_then_gates_old_menu(h):
    session = await h.login()
    await h.callback("connect_ads")
    await h.text("performance-client")
    await h.text("performance-test-secret")
    session.costs["001"] = 25
    session.seo.append({"sku": "001", "title": "Test"})
    await h.callback("logout")
    assert not h.app.sessions
    assert not h.app.pending
    assert session.api_key == session.client_id == ""
    assert session.performance_secret == session.performance_id == ""
    assert session.name == ""
    assert not session.costs and not session.seo
    assert h.storage.deleted == [session.id]
    await h.callback("report:finance")
    assert h.seller.loads == 0
    assert "Сначала" in h.telegram.messages[-1].text


async def test_logout_cancels_inflight_report_before_erasing_credentials(h):
    session = await h.login()
    h.seller.block_load = True
    report_task = asyncio.create_task(h.callback("report:sales"))
    await asyncio.wait_for(h.seller.load_started.wait(), 2)
    assert session.task is report_task
    await h.callback("logout")
    (result,) = await asyncio.gather(report_task, return_exceptions=True)
    assert isinstance(result, asyncio.CancelledError)
    assert h.seller.load_cancelled
    assert session.api_key == ""
    assert 101 not in h.app.sessions
    assert h.storage.deleted == [session.id]
    assert not any("Продажи\n" in call.text for call in h.telegram.messages)


async def test_expired_session_is_removed_on_next_update(h):
    session = await h.login()
    session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await h.callback("report:sales")
    assert 101 not in h.app.sessions
    assert session.api_key == ""
    assert h.storage.deleted == [session.id]
    assert not h.seller.loads


async def test_logout_during_database_touch_cannot_resume_report_on_ended_session(h):
    session = await h.login()
    h.storage.release_touch = asyncio.Event()
    report_task = asyncio.create_task(h.callback("report:sales"))
    await asyncio.wait_for(h.storage.touch_started.wait(), 2)
    await h.callback("logout")
    h.storage.release_touch.set()
    await asyncio.wait_for(report_task, 2)
    assert h.seller.loads == 0
    assert session.api_key == ""
    assert session.snapshot is None
    assert 101 not in h.app.sessions


async def test_logout_during_secret_deletion_cannot_reintroduce_ads_credentials(h):
    session = await h.login()
    await h.callback("connect_ads")
    await h.text("performance-client")
    h.telegram.delete_started.clear()
    h.telegram.release_delete = asyncio.Event()
    secret_task = asyncio.create_task(h.text("performance-test-secret"))
    await asyncio.wait_for(h.telegram.delete_started.wait(), 2)
    await h.callback("logout")
    h.telegram.release_delete.set()
    await asyncio.wait_for(secret_task, 2)
    assert h.performance.credentials == []
    assert session.performance_secret == session.performance_id == ""
    assert not h.app.sessions


async def test_user_and_chat_isolation_for_callbacks_and_logout(h):
    first = await h.login(user_id=101)
    second = await h.login(user_id=202)
    await h.callback("tax:15", user_id=202, chat_id=101)
    assert first.tax_rate == second.tax_rate == 6
    assert not h.storage.taxes
    await h.callback("logout", user_id=202)
    assert 101 in h.app.sessions and 202 not in h.app.sessions
    assert first.api_key == "seller-test-secret"
    assert second.api_key == ""
    assert h.storage.deleted == [second.id]


async def test_cost_csv_import_through_dispatcher_preserves_sku_and_updates_context(h):
    session = await h.login()
    await h.callback("import:costs")
    templates = [call for call in h.telegram.calls if isinstance(call, SendDocument)]
    assert templates[-1].document.filename == "costs_template.csv"
    await h.document("sku;cost\n001234;35,50\n000222;17\n".encode("utf-8-sig"))
    assert session.costs == {"001234": 35.5, "000222": 17.0}
    assert session.state is None
    assert h.storage.imports == [
        (
            session.id,
            "costs",
            [
                {"sku": "001234", "cost": 35.5},
                {"sku": "000222", "cost": 17.0},
            ],
        )
    ]
    assert any("2 строк" in call.text for call in h.telegram.messages)


async def test_invalid_csv_does_not_replace_previous_costs_or_database_data(h):
    session = await h.login()
    session.costs = {"001": 25.0}
    await h.callback("import:costs")
    await h.document(b"sku,cost,api_key\n001,8,secret\n")
    assert session.costs == {"001": 25.0}
    assert not h.storage.imports
    assert session.state == "import:costs"
    assert any("неизвестные столбцы" in call.text for call in h.telegram.messages)


async def test_ads_csv_disables_api_source_and_survives_api_disconnect(h):
    session = await h.login()
    await h.callback("connect_ads")
    await h.text("performance-client")
    await h.text("performance-test-secret")
    assert session.ads_source == "api"
    await h.callback("import:ads")
    await h.document(SAMPLE_CSV["ads"], filename="advertising.csv")
    imported = list(session.ads)
    assert imported
    assert session.ads_source == "csv"
    assert session.performance_id == session.performance_secret == ""
    await h.callback("report:ads")
    assert session.ads == imported
    assert len(h.performance.credentials) == 1  # Only authentication before the import.
    await h.callback("disconnect_ads")
    assert session.ads == imported
    assert session.ads_source == "csv"


async def test_reconnecting_ads_switches_from_csv_after_successful_auth_only(h):
    session = await h.login()
    await h.callback("import:ads")
    await h.document(SAMPLE_CSV["ads"], filename="advertising.csv")
    imported = list(session.ads)
    await h.callback("connect_ads")
    await h.text("performance-client")
    h.performance.reject = True
    await h.text("bad-performance-secret")
    assert session.ads == imported and session.ads_source == "csv"
    assert session.performance_secret == ""
    h.performance.reject = False
    await h.text("new-performance-secret")
    assert session.ads_source == "api"
    assert session.ads == []
    assert session.performance_secret == "new-performance-secret"


async def test_seo_pagination_buttons_select_next_page_and_reuse_snapshot(h):
    session = await h.login()
    session.snapshot = Snapshot(
        products=[{"sku": f"{i:03d}", "title": f"Карточка товара {i}"} for i in range(11)]
    )
    session.cache_time = datetime.now(UTC)
    await h.callback("report:seo")
    first = h.telegram.messages[-1]
    assert "Страница 1/2" in first.text
    assert any(
        button.callback_data == "page:seo:1"
        for row in first.reply_markup.inline_keyboard
        for button in row
    )
    await h.callback("page:seo:1")
    second = h.telegram.messages[-1]
    assert "Страница 2/2" in second.text
    assert "SKU 010" in second.text
    assert "SKU 000" not in second.text
    assert any(
        button.callback_data == "page:seo:0"
        for row in second.reply_markup.inline_keyboard
        for button in row
    )
    assert h.seller.loads == 0


async def test_tax_buttons_and_custom_input_reject_nan_without_mutating_state(h):
    session = await h.login()
    await h.callback("tax:15")
    assert session.tax_rate == 15 and session.tax_configured
    await h.callback("tax_custom")
    await h.text("NaN")
    assert session.tax_rate == 15
    assert h.storage.taxes == [(session.id, 15.0)]
    assert session.state == "tax"
    await h.text("7,5")
    assert session.tax_rate == 7.5 and session.state is None
    assert h.storage.taxes[-1] == (session.id, 7.5)


async def test_period_error_keeps_previous_period_and_allows_correction(h):
    session = await h.login()
    previous = session.period
    await h.callback("custom_period")
    tomorrow = date.today() + timedelta(days=1)
    await h.text(f"{date.today()} {tomorrow}")
    assert session.period == previous
    assert session.state == "period"
    await h.text("2026-01-01 2026-01-31")
    assert session.period.start == date(2026, 1, 1)
    assert session.period.end == date(2026, 1, 31)
    assert session.state is None


async def test_ai_dispatch_sends_business_summary_and_no_credentials_or_identity(h):
    session = await h.login()
    session.performance_id = "ads-test-client"
    session.performance_secret = "ads-test-secret"
    await h.callback("report:ai")
    (facts,) = h.ai.facts
    serialized = json.dumps(facts, ensure_ascii=False)
    for secret in (
        session.api_key,
        session.performance_id,
        session.performance_secret,
        session.name,
        session.id,
        "shared-ai-test-key",
    ):
        assert secret not in serialized
    assert {"finance", "sales", "ads", "seo", "competitors", "stocks", "categories"} <= set(facts)
    assert "Совет ИИ" in h.telegram.messages[-1].text


async def test_start_replaces_session_and_erases_previous_key(h):
    old = await h.login()
    await h.text("/start")
    assert old.api_key == ""
    assert h.storage.deleted == [old.id]
    assert not h.app.sessions
    assert h.app.pending[101].step == "name"


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-31 2026-01-01",
        "1999-12-31 2026-01-01",
        "2026-02-30 2026-03-01",
        "01.01.2026 31.01.2026",
        "2026-01-01",
        "2026-01-01 2026-01-31 extra",
    ],
)
def test_period_validation_rejects_unsupported_input(value):
    with pytest.raises(ValueError):
        parse_period(value)
