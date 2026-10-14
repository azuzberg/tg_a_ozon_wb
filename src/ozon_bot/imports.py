"""Bounded, strict CSV imports for data unavailable through Ozon Seller API."""

from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import polars as pl

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 20_000

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "costs": frozenset({"sku", "cost"}),
    "ads": frozenset({"date", "campaign_id", "campaign_name", "spend", "revenue", "orders"}),
    "seo": frozenset({"sku", "title"}),
    "competitors": frozenset({"date", "competitor_sku", "name", "price"}),
}
IMPORT_COLUMNS: dict[str, frozenset[str]] = {
    "costs": REQUIRED_COLUMNS["costs"],
    "ads": REQUIRED_COLUMNS["ads"] | {"sku", "clicks", "impressions"},
    "seo": REQUIRED_COLUMNS["seo"]
    | {"description", "keywords", "position", "previous_position", "date"},
    "competitors": REQUIRED_COLUMNS["competitors"]
    | {"sku", "category", "rating", "reviews", "position"},
}

SAMPLE_CSV: dict[str, bytes] = {
    "costs": "sku;cost\n001234567;350.00\n".encode("utf-8-sig"),
    "ads": (
        "date;campaign_id;campaign_name;spend;revenue;orders;sku;clicks;impressions\n"
        "2026-01-15;00123;Продвижение товара;500;5000;10;001234567;100;5000\n"
    ).encode("utf-8-sig"),
    "seo": (
        "sku;title;description;keywords;position;previous_position;date\n"
        "001234567;Термокружка 500 мл;Стальная термокружка;термокружка, кружка;12;18;2026-01-15\n"
    ).encode("utf-8-sig"),
    "competitors": (
        "date;competitor_sku;name;price;sku;category;rating;reviews;position\n"
        "2026-01-15;009876543;Термокружка конкурента;990;001234567;Термокружки;4.7;150;8\n"
    ).encode("utf-8-sig"),
}

_MONEY = frozenset({"cost", "spend", "revenue", "price"})
_COUNTS = frozenset({"orders", "clicks", "impressions", "reviews"})
_POSITIONS = frozenset({"position", "previous_position"})
_IDENTIFIERS = frozenset({"sku", "competitor_sku", "campaign_id"})


def _error(row: int | None, message: str) -> ValueError:
    prefix = f"Строка {row}: " if row is not None else ""
    return ValueError(prefix + message)


def _text(value: Any, column: str, row: int) -> str:
    if not isinstance(value, str):
        # Identifiers must stay strings; accepting an integer would silently
        # discard leading zeroes that the importing caller already lost.
        if column in _IDENTIFIERS:
            raise _error(row, f"поле {column} должно быть текстовым идентификатором.")
        value = str(value)
    value = value.strip()
    maximum = 20_000 if column == "description" else 5_000 if column == "keywords" else 1_000
    if column in _IDENTIFIERS:
        maximum = 120
    if len(value) > maximum:
        raise _error(row, f"поле {column} слишком длинное (максимум {maximum} символов).")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
        raise _error(row, f"поле {column} содержит недопустимые символы.")
    if column in _IDENTIFIERS and any(character in "\n\r\t" for character in value):
        raise _error(row, f"поле {column} содержит недопустимые символы.")
    return value


def _number(value: Any, column: str, row: int) -> int | float:
    if isinstance(value, bool):
        raise _error(row, f"поле {column} должно быть числом.")
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, ValueError):
        raise _error(row, f"поле {column} должно быть числом.") from None
    if not number.is_finite():
        raise _error(row, f"поле {column} должно быть конечным числом.")
    maximum = Decimal(5) if column == "rating" else Decimal(1000000000000)
    minimum = Decimal(1) if column in _POSITIONS else Decimal(0)
    if number < minimum or number > maximum:
        raise _error(row, f"поле {column} вне допустимого диапазона {minimum}–{maximum}.")
    if column in _COUNTS or column in _POSITIONS:
        if number != number.to_integral_value():
            raise _error(row, f"поле {column} должно быть целым числом.")
        return int(number)
    result = float(number)
    if not math.isfinite(result):
        raise _error(row, f"поле {column} должно быть конечным числом.")
    return result


def _date(value: Any, row: int) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    raise _error(row, "дата должна быть существующей датой YYYY-MM-DD или ДД.ММ.ГГГГ.")


def normalize_rows(kind: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate imported records again before persisting their JSON payload.

    The allowlist intentionally excludes credentials and arbitrary extra fields.
    The same validation applies to CSV input and calls to the storage layer.
    """
    if kind not in IMPORT_COLUMNS:
        raise ValueError("Неизвестный тип импорта.")
    if not rows:
        raise ValueError("CSV должен содержать хотя бы одну строку данных.")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"CSV содержит больше {MAX_ROWS} строк. Разделите файл.")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw in enumerate(rows, start=2):
        if not isinstance(raw, dict) or set(raw) - IMPORT_COLUMNS[kind]:
            raise _error(index, "есть неизвестные столбцы. Используйте шаблон этого раздела.")
        if REQUIRED_COLUMNS[kind] - set(raw):
            raise _error(
                index, "отсутствуют обязательные столбцы. Используйте шаблон этого раздела."
            )
        normalized: dict[str, Any] = {}
        for column, value in raw.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                if column in REQUIRED_COLUMNS[kind]:
                    raise _error(index, f"обязательное поле {column} пустое.")
                continue
            if column == "date":
                normalized[column] = _date(value, index)
            elif (
                column in _MONEY or column in _COUNTS or column in _POSITIONS or column == "rating"
            ):
                normalized[column] = _number(value, column, index)
            else:
                normalized[column] = _text(value, column, index)
        if kind == "costs":
            key = (normalized["sku"],)
        elif kind == "ads":
            key = (normalized["date"], normalized["campaign_id"], normalized.get("sku", ""))
        elif kind == "seo":
            key = (normalized.get("date", ""), normalized["sku"])
        else:
            key = (normalized["date"], normalized["competitor_sku"])
        if key in seen:
            raise _error(index, "повторяется запись с тем же идентификатором и датой.")
        seen.add(key)
        result.append(normalized)
    return result


def parse_csv(kind: str, data: bytes) -> list[dict[str, Any]]:
    """Parse UTF-8/CP1251 CSV; never infer numeric types for identifier columns."""
    if kind not in IMPORT_COLUMNS:
        raise ValueError("Неизвестный тип импорта.")
    if len(data) > MAX_BYTES:
        raise ValueError("CSV больше 2 МиБ. Разделите файл.")
    if not data:
        raise ValueError("CSV пустой. Скачайте шаблон и добавьте строки данных.")
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = data.decode("cp1251")
        except UnicodeDecodeError:
            raise ValueError(
                "Не удалось прочитать кодировку CSV. Используйте UTF-8 или Windows-1251."
            ) from None
    if "\x00" in decoded:
        raise ValueError("Недопустимый CSV: обнаружены нулевые байты.")
    try:
        delimiter = csv.Sniffer().sniff(decoded[:16_384], delimiters=",;").delimiter
    except csv.Error:
        first_line = decoded.splitlines()[0] if decoded.splitlines() else ""
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    try:
        header = next(csv.reader(io.StringIO(decoded), delimiter=delimiter, strict=True))
    except (StopIteration, csv.Error):
        raise ValueError("Не удалось прочитать заголовок CSV.") from None
    columns = [column.strip().lower() for column in header]
    if len(columns) != len(set(columns)):
        raise ValueError("В CSV повторяются названия столбцов.")
    if set(columns) - IMPORT_COLUMNS[kind]:
        raise ValueError("CSV содержит неизвестные столбцы. Используйте шаблон этого раздела.")
    missing = REQUIRED_COLUMNS[kind] - set(columns)
    if missing:
        raise ValueError(
            "В CSV отсутствуют обязательные столбцы: " + ", ".join(sorted(missing)) + "."
        )
    try:
        frame = pl.read_csv(
            io.BytesIO(decoded.encode("utf-8")),
            separator=delimiter,
            infer_schema=False,
            n_rows=MAX_ROWS + 1,
            raise_if_empty=True,
            truncate_ragged_lines=False,
        )
        frame.columns = columns
    except (pl.exceptions.PolarsError, ValueError):
        raise ValueError(
            "Не удалось прочитать строки CSV. Проверьте разделители, кавычки и число столбцов."
        ) from None
    return normalize_rows(kind, frame.to_dicts())
