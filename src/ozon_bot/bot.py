import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from io import BytesIO

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .ai import ClaudeClient, ClaudeError
from .analytics import build_report
from .config import Settings
from .domain import Period
from .imports import MAX_BYTES, SAMPLE_CSV, parse_csv
from .ozon import OzonAPIError, OzonClient
from .performance import PerformanceAPIError, PerformanceClient
from .sessions import Session
from .storage import Storage

log = logging.getLogger(__name__)
REPORTS = {
    "finance": "💰 Финансы",
    "sales": "📈 Продажи",
    "ads": "📣 Реклама",
    "seo": "🔎 SEO",
    "competitors": "🏪 Конкуренты",
    "stocks": "📦 Остатки",
    "categories": "🗂 Категории",
    "ai": "🧠 Совет ИИ",
}
IMPORT_NAMES = {
    "costs": "Себестоимость",
    "ads": "Реклама",
    "seo": "SEO и позиции",
    "competitors": "Конкуренты",
}


def keyboard(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


def cancel_keyboard() -> InlineKeyboardMarkup:
    return keyboard([[("↩ Меню / отмена ввода", "menu")], [("🚪 Завершить", "logout")]])


def main_keyboard() -> InlineKeyboardMarkup:
    items = list(REPORTS.items())
    rows = [
        [(title, f"report:{kind}") for kind, title in items[i : i + 2]]
        for i in range(0, len(items), 2)
    ]
    return keyboard(
        rows
        + [
            [("🗓 Период", "period"), ("💵 Общая прибыль", "all_profit")],
            [("⚙ Налоги и CSV", "settings"), ("🔗 Подключить рекламу", "connect_ads")],
            [("🔄 Обновить данные", "refresh"), ("🚪 Завершить", "logout")],
        ]
    )


def parse_period(text: str) -> Period:
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError("Введите две даты через пробел: 2026-01-01 2026-01-31.")
    try:
        start, end = (date.fromisoformat(value) for value in parts)
    except ValueError:
        raise ValueError("Формат дат: ГГГГ-ММ-ДД ГГГГ-ММ-ДД.") from None
    if start < date(2000, 1, 1):
        raise ValueError("Начало периода должно быть не раньше 2000-01-01.")
    return Period(start, end, "Свой период")


@dataclass
class Pending:
    step: str = "name"
    name: str = ""
    client_id: str = ""
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(minutes=10))
    task: asyncio.Task | None = field(default=None, repr=False)


class BotApp:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        *,
        seller_factory=OzonClient,
        performance_factory=PerformanceClient,
        ai_client=None,
    ):
        self.settings = settings
        self.storage = storage
        self.seller_factory = seller_factory
        self.performance_factory = performance_factory
        self.ai = ai_client or ClaudeClient(settings.claude_api_key, settings.claude_model)
        self.sessions: dict[int, Session] = {}
        self.pending: dict[int, Pending] = {}
        self.router = Router()
        self.router.message.filter(F.chat.type == "private")
        self.router.callback_query.filter(F.message.chat.type == "private")
        self.router.message.register(self.start, CommandStart())
        self.router.message.register(self.help, Command("help"))
        self.router.message.register(self.document, F.document)
        self.router.message.register(self.text, F.text)
        self.router.callback_query.register(self.callback)

    async def send(self, bot: Bot, chat_id: int, text: str, markup=None) -> None:
        # Лимит Telegram применяется к каждому сообщению. Текст API не интерпретируется как HTML.
        while len(text) > 3600:
            split = text.rfind("\n", 0, 3600)
            split = split if split > 100 else 3600
            await bot.send_message(chat_id, text[:split], parse_mode=None)
            text = text[split:].lstrip("\n")
        await bot.send_message(chat_id, text or "Нет данных.", reply_markup=markup, parse_mode=None)

    async def remove_secret_message(self, message: Message) -> None:
        try:
            await message.delete()
        except (TelegramBadRequest, TelegramForbiddenError):
            await message.answer("Не удалось удалить сообщение с ключом. Удалите его вручную.")

    async def end(self, user_id: int) -> None:
        pending = self.pending.pop(user_id, None)
        session = self.sessions.pop(user_id, None)
        current = asyncio.current_task()
        tasks = [
            item.task for item in (pending, session) if item and item.task and item.task != current
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if session:
            session.clear()
            await self.storage.delete_session(session.id)

    async def active(self, user_id: int, chat_id: int) -> Session | None:
        session = self.sessions.get(user_id)
        if not session or session.chat_id != chat_id:
            return None
        if session.expires_at <= datetime.now(UTC):
            await self.end(user_id)
            return None
        session.touch()
        await self.storage.touch(session.id, session.expires_at)
        if self.sessions.get(user_id) is not session:
            return None
        return session

    async def menu(self, bot: Bot, session: Session) -> None:
        await self.send(
            bot,
            session.chat_id,
            f"{session.name}, выберите анализ.\n"
            f"Период: {session.period.start:%d.%m.%Y} — {session.period.end:%d.%m.%Y}.\n"
            f"Налог: {session.tax_rate:g}% от выручки"
            + (" (предварительно; укажите свою ставку)." if not session.tax_configured else ".")
            + f"\nСеанс истекает через {session.ttl_minutes} мин. бездействия.",
            main_keyboard(),
        )

    async def start(self, message: Message) -> None:
        await self.end(message.from_user.id)
        self.pending[message.from_user.id] = Pending()
        await message.answer(
            "Здравствуйте! Как к вам обращаться?\n"
            "Далее понадобятся Client-Id и Api-Key Ozon Seller. Используйте ключ только для чтения. "
            "Ключи хранятся только в памяти до завершения сеанса или перезапуска. "
            "Сообщения с ключами бот попробует удалить.",
            reply_markup=cancel_keyboard(),
        )

    async def help(self, message: Message) -> None:
        await message.answer(
            "Вход: /start → имя → Client-Id → Api-Key Ozon Seller. "
            "В меню выбирайте период и раздел кнопками. Себестоимость и внешние данные "
            "загружайте по CSV-шаблонам в настройках. Claude получает сводку только по кнопке "
            "«Совет ИИ»; это передаёт бизнес-показатели Anthropic. "
            "«Завершить» удаляет ключи из памяти, данные сеанса и импорты из локальной базы.",
            reply_markup=cancel_keyboard(),
        )

    async def text(self, message: Message) -> None:
        user_id = message.from_user.id
        pending = self.pending.get(user_id)
        if pending:
            await self.login_text(message, pending)
            return
        session = await self.active(user_id, message.chat.id)
        if not session:
            await message.answer("Сеанс не активен. Нажмите /start.")
            return
        if session.state == "ads_secret":
            await self.remove_secret_message(message)
        if session.lock.locked():
            await message.answer("Анализ ещё выполняется. Можно завершить сеанс кнопкой.")
            return
        text = message.text.strip()
        try:
            if session.state == "period":
                session.period = parse_period(text)
                session.snapshot = None
            elif session.state == "all_profit":
                session.period = parse_period(f"{text} {date.today().isoformat()}")
                session.snapshot = None
                session.state = None
                await self.run_report(message.bot, session, "finance")
                return
            elif session.state == "tax":
                rate = float(text.replace(",", ".").replace("%", ""))
                if not math.isfinite(rate) or not 0 <= rate <= 100:
                    raise ValueError("Налоговая ставка должна быть от 0 до 100%.")
                await self.storage.set_tax(session.id, rate)
                session.tax_rate, session.tax_configured = rate, True
            elif session.state == "ads_id":
                if not 1 <= len(text) <= 200:
                    raise ValueError("Укажите Client ID Ozon Performance (до 200 символов).")
                session.draft["ads_id"] = text
                session.state = "ads_secret"
                await message.answer(
                    "Теперь Client Secret Ozon Performance.", reply_markup=cancel_keyboard()
                )
                return
            elif session.state == "ads_secret":
                if not 8 <= len(text) <= 1000:
                    raise ValueError("Проверьте Client Secret Ozon Performance.")
                async with session.lock:
                    if self.sessions.get(user_id) is not session:
                        return
                    session.task = asyncio.current_task()
                    try:
                        async with self.performance_factory(session.draft["ads_id"], text) as api:
                            await api.validate()
                        session.performance_id = session.draft.pop("ads_id")
                        session.performance_secret = text
                        session.ads = []
                        session.ads_source = "api"
                        session.ads_period = None
                        session.ads_time = None
                    finally:
                        session.task = None
                await message.answer(
                    "Реклама подключена. Источник отчётов — Performance API; "
                    "предыдущий CSV не используется. Данные загрузятся при анализе рекламы."
                )
            elif session.state and session.state.startswith("import:"):
                await message.answer(
                    "Отправьте CSV как файл, а не сообщением.", reply_markup=cancel_keyboard()
                )
                return
            else:
                await self.menu(message.bot, session)
                return
        except (ValueError, OzonAPIError, PerformanceAPIError) as error:
            await message.answer(str(error), reply_markup=cancel_keyboard())
            return
        session.state = None
        await self.menu(message.bot, session)

    async def login_text(self, message: Message, pending: Pending) -> None:
        user_id = message.from_user.id
        if pending.expires_at <= datetime.now(UTC):
            await self.end(user_id)
            await message.answer("Время входа истекло. Нажмите /start.")
            return
        if pending.step == "api_key":
            await self.remove_secret_message(message)
            if self.pending.get(user_id) is not pending:
                return
        if pending.task:
            await message.answer("Проверяю доступ к магазину…")
            return
        text = message.text.strip()
        if pending.step == "name":
            if not 1 <= len(text) <= 120:
                await message.answer("Имя должно содержать от 1 до 120 символов.")
                return
            pending.name, pending.step = text, "client_id"
            await message.answer("Введите Client-Id Ozon Seller.", reply_markup=cancel_keyboard())
        elif pending.step == "client_id":
            if not text.isascii() or not text.isdigit() or len(text) > 30:
                await message.answer("Client-Id должен состоять из цифр.")
                return
            pending.client_id, pending.step = text, "api_key"
            await message.answer("Введите Api-Key Ozon Seller.", reply_markup=cancel_keyboard())
        else:
            if not 8 <= len(text) <= 1000:
                await message.answer("Проверьте Api-Key (от 8 до 1000 символов).")
                return
            pending.task = asyncio.current_task()
            session = None
            registered = False
            try:
                async with self.seller_factory(pending.client_id, text) as api:
                    await api.validate()
                session = Session(
                    user_id,
                    message.chat.id,
                    pending.name,
                    pending.client_id,
                    text,
                    self.settings.session_ttl_minutes,
                )
                session.touch()
                await self.storage.create_session(
                    session.id, user_id, message.chat.id, session.name, session.expires_at
                )
                self.sessions[user_id] = session
                registered = True
                self.pending.pop(user_id, None)
                await self.menu(message.bot, session)
            except OzonAPIError as error:
                await message.answer(
                    str(error) + "\nПопробуйте другой Api-Key.", reply_markup=cancel_keyboard()
                )
            finally:
                pending.task = None
                if session and not registered:
                    session.clear()
                    await asyncio.shield(self.storage.delete_session(session.id))

    async def callback(self, query: CallbackQuery) -> None:
        await query.answer()
        user_id, chat_id = query.from_user.id, query.message.chat.id
        action = query.data or ""
        if action == "logout":
            await self.end(user_id)
            await query.message.answer(
                "Сеанс завершён. Ключи и данные сеанса удалены. Для входа /start."
            )
            return
        session = await self.active(user_id, chat_id)
        if not session:
            await query.message.answer("Сначала завершите вход через /start.")
            return
        if session.lock.locked():
            await query.message.answer("Анализ ещё выполняется. Можно завершить сеанс кнопкой.")
            return
        if action == "menu":
            session.state = None
            session.draft.clear()
            await self.menu(query.bot, session)
        elif action == "period":
            await query.message.answer(
                "Выберите период (включая обе даты):",
                reply_markup=keyboard(
                    [
                        [("7 дней", "days:7"), ("30 дней", "days:30"), ("90 дней", "days:90")],
                        [("Текущий месяц", "month"), ("Ввести даты", "custom_period")],
                        [("↩ Меню", "menu")],
                    ]
                ),
            )
        elif action.startswith("days:") and action[5:] in ("7", "30", "90"):
            session.period, session.snapshot, session.state = (
                Period.last_days(int(action[5:])),
                None,
                None,
            )
            await self.menu(query.bot, session)
        elif action == "month":
            today = date.today()
            session.period = Period(today.replace(day=1), today, "Текущий месяц")
            session.snapshot, session.state = None, None
            await self.menu(query.bot, session)
        elif action in ("custom_period", "all_profit"):
            session.state = "period" if action == "custom_period" else "all_profit"
            await query.message.answer(
                "Введите период: ГГГГ-ММ-ДД ГГГГ-ММ-ДД."
                if action == "custom_period"
                else "Введите дату первой продажи: ГГГГ-ММ-ДД.\n"
                "Общая прибыль будет рассчитана от этой даты до сегодня, в пределах доступной "
                "истории Ozon. Большой период загружается дольше.",
                reply_markup=cancel_keyboard(),
            )
        elif action == "settings":
            session.state = None
            await query.message.answer(
                "Настройки и импорт CSV:",
                reply_markup=keyboard(
                    [
                        [("Налоговая ставка", "tax")],
                        *[
                            [(f"CSV: {title}", f"import:{kind}")]
                            for kind, title in IMPORT_NAMES.items()
                        ],
                        [("Отключить рекламу", "disconnect_ads")],
                        [("↩ Меню", "menu")],
                    ]
                ),
            )
        elif action == "tax":
            await query.message.answer(
                "Выберите ставку. Это упрощённый налог с выручки; "
                "НДС, УСН «доходы минус расходы», взносы и льготы требуют отдельного расчёта.",
                reply_markup=keyboard(
                    [
                        [("0%", "tax:0"), ("6%", "tax:6"), ("15%", "tax:15")],
                        [("Другая ставка", "tax_custom")],
                        [("↩ Меню", "menu")],
                    ]
                ),
            )
        elif action.startswith("tax:") and action[4:] in ("0", "6", "15"):
            rate = float(action[4:])
            await self.storage.set_tax(session.id, rate)
            session.tax_rate, session.tax_configured, session.state = rate, True, None
            await self.menu(query.bot, session)
        elif action == "tax_custom":
            session.state = "tax"
            await query.message.answer(
                "Введите процент налога с выручки, например 6 или 7,5.",
                reply_markup=cancel_keyboard(),
            )
        elif action == "connect_ads":
            session.state = "ads_id"
            await query.message.answer(
                "Введите Client ID из Ozon Performance API (это не Seller Client-Id).",
                reply_markup=cancel_keyboard(),
            )
        elif action == "disconnect_ads":
            session.performance_id, session.performance_secret = "", ""
            if session.ads_source == "api":
                session.ads, session.ads_time, session.ads_period = [], None, None
                session.ads_source = "none"
            session.state = None
            await query.message.answer("Подключение рекламы отключено, ключ удалён из памяти.")
            await self.menu(query.bot, session)
        elif action.startswith("import:") and action[7:] in IMPORT_NAMES:
            kind = action[7:]
            session.state = action
            await query.message.answer_document(
                BufferedInputFile(SAMPLE_CSV[kind], filename=f"{kind}_template.csv"),
                caption="Замените пример своими данными и отправьте CSV файлом. "
                "Максимум 2 МиБ / 20 000 строк. Новый импорт заменяет прежний для этого раздела. "
                "Пример не загружается автоматически.",
                reply_markup=cancel_keyboard(),
            )
        elif action == "refresh":
            session.snapshot, session.ads_time = None, None
            await self.run_report(query.bot, session, "sales")
        elif action.startswith("report:") and action[7:] in REPORTS:
            session.state = None
            await self.run_report(query.bot, session, action[7:])
        elif action.startswith("category:"):
            try:
                page = int(action[9:])
                if not 0 <= page <= 10000:
                    raise ValueError
            except ValueError:
                return
            await self.run_report(query.bot, session, "categories", page=page)
        elif action.startswith("page:"):
            parts = action.split(":")
            if len(parts) != 3 or parts[1] not in ("seo", "stocks", "competitors", "categories"):
                return
            try:
                page = int(parts[2])
                if not 0 <= page <= 10000:
                    return
            except ValueError:
                return
            await self.run_report(query.bot, session, parts[1], page=page)

    async def document(self, message: Message) -> None:
        session = await self.active(message.from_user.id, message.chat.id)
        if not session:
            await message.answer("Для импорта сначала войдите через /start.")
            return
        if session.lock.locked():
            await message.answer("Дождитесь завершения анализа.")
            return
        if not session.state or not session.state.startswith("import:"):
            await message.answer("Сначала выберите вид CSV в «Налоги и CSV».")
            return
        if not message.document.file_name or not message.document.file_name.lower().endswith(
            ".csv"
        ):
            await message.answer("Нужен файл с расширением .csv.")
            return
        if message.document.file_size is None or message.document.file_size > MAX_BYTES:
            await message.answer("Файл должен быть не больше 2 МиБ.")
            return
        kind = session.state[7:]
        async with session.lock:
            if self.sessions.get(session.user_id) is not session:
                return
            session.task = asyncio.current_task()
            try:
                buffer = BytesIO()
                await message.bot.download(message.document, destination=buffer)
                rows = await asyncio.to_thread(parse_csv, kind, buffer.getvalue())
                await self.storage.save_import(session.id, kind, rows)
                if kind == "costs":
                    session.costs = {row["sku"]: row["cost"] for row in rows}
                else:
                    setattr(session, kind, rows)
                    if kind == "ads":
                        session.performance_id, session.performance_secret = "", ""
                        session.ads_source = "csv"
                        session.ads_period, session.ads_time = session.period, datetime.now(UTC)
                session.state = None
                await message.answer(
                    f"Импорт «{IMPORT_NAMES[kind]}»: {len(rows)} строк. Данные сохранены на время сеанса."
                )
                if kind == "ads":
                    await message.answer(
                        "Реклама переключена на CSV. Подключение Performance отключено, "
                        "его ключ удалён из памяти. Новый API-отчёт не перезапишет CSV."
                    )
            except ValueError as error:
                await message.answer(str(error), reply_markup=cancel_keyboard())
            finally:
                session.task = None
        await self.menu(message.bot, session)

    async def run_report(self, bot: Bot, session: Session, kind: str, page: int = 0) -> None:
        async with session.lock:
            if self.sessions.get(session.user_id) is not session:
                return
            session.task = asyncio.current_task()
            try:
                await self.send(
                    bot,
                    session.chat_id,
                    "Загружаю данные и рассчитываю показатели… "
                    "Можно остановить запрос кнопкой «Завершить».",
                    cancel_keyboard(),
                )
                now = datetime.now(UTC)
                if (
                    session.snapshot is None
                    or session.cache_time is None
                    or now - session.cache_time > timedelta(minutes=5)
                ):
                    async with self.seller_factory(session.client_id, session.api_key) as api:
                        session.snapshot = await api.load(session.period.start, session.period.end)
                    session.cache_time = datetime.now(UTC)
                if kind in ("ads", "ai") and session.performance_secret:
                    if (
                        session.ads_period != session.period
                        or session.ads_time is None
                        or now - session.ads_time > timedelta(minutes=5)
                    ):
                        try:
                            async with self.performance_factory(
                                session.performance_id, session.performance_secret
                            ) as api:
                                session.ads, warnings = await api.load(
                                    session.period.start, session.period.end
                                )
                            session.snapshot.warnings.extend(warnings)
                            session.ads_period, session.ads_time = session.period, datetime.now(UTC)
                        except PerformanceAPIError as error:
                            # Старый период рекламы не смешивается с новым отчётом.
                            session.ads = []
                            session.snapshot.warnings.append(f"Реклама недоступна: {error}")
                context = session.context()
                if kind == "ai":
                    facts = {
                        "period": {
                            "start": str(session.period.start),
                            "end": str(session.period.end),
                        },
                        "warnings": context.snapshot.warnings,
                    }
                    for report_kind in REPORTS:
                        if report_kind != "ai":
                            report = await asyncio.to_thread(build_report, report_kind, context)
                            facts[report_kind] = {"facts": report.facts, "notes": report.text}
                    await self.send(
                        bot,
                        session.chat_id,
                        "Отправляю сводные показатели Claude для рекомендаций.",
                    )
                    answer = await self.ai.advise(facts)
                    await self.send(bot, session.chat_id, "Совет ИИ\n\n" + answer, main_keyboard())
                else:
                    report = await asyncio.to_thread(build_report, kind, context, page)
                    markup = main_keyboard()
                    if kind in ("categories", "seo", "stocks", "competitors"):
                        page = report.facts.get("page", page)
                        nav = []
                        if page > 0:
                            nav.append(("← Назад", f"page:{kind}:{page - 1}"))
                        pages = report.facts.get("pages", 1)
                        if page + 1 < pages:
                            nav.append(("Далее →", f"page:{kind}:{page + 1}"))
                        markup = keyboard(
                            ([nav] if nav else [])
                            + [[("↩ Меню", "menu"), ("🚪 Завершить", "logout")]]
                        )
                    await self.send(
                        bot,
                        session.chat_id,
                        f"{report.title}\n"
                        f"{session.period.start:%d.%m.%Y} — {session.period.end:%d.%m.%Y}\n\n{report.text}",
                        markup,
                    )
            except (OzonAPIError, PerformanceAPIError, ClaudeError, ValueError) as error:
                await self.send(bot, session.chat_id, str(error), main_keyboard())
            finally:
                session.task = None

    async def sweep(self, bot: Bot) -> None:
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            for user_id, session in list(self.sessions.items()):
                if session.expires_at <= now:
                    try:
                        await self.end(user_id)
                        await self.send(
                            bot, session.chat_id, "Сеанс истёк. Ключи удалены. Для входа /start."
                        )
                    except Exception as error:
                        log.error("Session cleanup failed (%s)", type(error).__name__)
            for user_id, pending in list(self.pending.items()):
                if pending.expires_at <= now:
                    await self.end(user_id)
            try:
                expired_ids = set(await self.storage.expire_sessions(now))
                for user_id, session in list(self.sessions.items()):
                    if session.id in expired_ids:
                        await self.end(user_id)
            except Exception as error:
                log.error("Database cleanup failed (%s)", type(error).__name__)


def create_dispatcher(app: BotApp) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.include_router(app.router)

    @dispatcher.errors()
    async def safe_error(event):
        # Не выводим exception text, update или traceback: они могут содержать секреты.
        log.error("Bot operation failed (%s)", type(event.exception).__name__)
        message = event.update.message or (
            event.update.callback_query.message if event.update.callback_query else None
        )
        if message:
            try:
                await message.answer(
                    "Не удалось выполнить действие. Повторите позже; проверьте доступ к файлу базы и API.",
                    reply_markup=cancel_keyboard(),
                )
            except Exception:
                pass
        return True

    return dispatcher
