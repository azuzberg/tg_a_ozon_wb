from datetime import date

import pytest

from ozon_bot.imports import MAX_BYTES, MAX_ROWS, SAMPLE_CSV, normalize_rows, parse_csv


@pytest.mark.parametrize("kind", ["costs", "ads", "seo", "competitors"])
def test_samples_are_valid(kind):
    rows = parse_csv(kind, SAMPLE_CSV[kind])
    assert len(rows) == 1
    assert rows[0].get("sku", "001234567") == "001234567"


def test_costs_semicolon_decimal_comma_and_leading_zeroes():
    rows = parse_csv("costs", "sku;cost\n00123;35,50\n".encode("utf-8-sig"))
    assert rows == [{"sku": "00123", "cost": 35.5}]


def test_comma_delimiter_and_quoted_decimal():
    rows = parse_csv("costs", b'sku,cost\n00123,"35,50"\n')
    assert rows == [{"sku": "00123", "cost": 35.5}]


def test_windows_1251_and_date_normalization():
    payload = (
        "date;competitor_sku;name;price;rating;reviews\n15.01.2026;00100;Термокружка;900;4,5;42\n"
    ).encode("cp1251")
    row = parse_csv("competitors", payload)[0]
    assert row == {
        "date": "2026-01-15",
        "competitor_sku": "00100",
        "name": "Термокружка",
        "price": 900.0,
        "rating": 4.5,
        "reviews": 42,
    }


@pytest.mark.parametrize("number", ["NaN", "inf", "-Infinity", "-1", "1000000000001"])
def test_rejects_nonfinite_negative_and_excessive_money(number):
    with pytest.raises(ValueError):
        parse_csv("costs", f"sku,cost\n00123,{number}\n".encode())


@pytest.mark.parametrize(
    "column,value", [("orders", "1.5"), ("clicks", "-1"), ("impressions", "inf")]
)
def test_counts_are_nonnegative_integers(column, value):
    fields = {
        "date": "2026-01-15",
        "campaign_id": "001",
        "campaign_name": "Ad",
        "spend": "10",
        "revenue": "50",
        "orders": "1",
        "clicks": "1",
        "impressions": "5",
    }
    fields[column] = value
    with pytest.raises(ValueError):
        normalize_rows("ads", [fields])


def test_rating_has_maximum_five():
    with pytest.raises(ValueError):
        parse_csv(
            "competitors", b"date,competitor_sku,name,price,rating\n2026-01-15,01,Item,5,5.1\n"
        )


@pytest.mark.parametrize("position", ["0", "-1", "1.5", "NaN"])
def test_positions_are_positive_integers(position):
    with pytest.raises(ValueError):
        parse_csv("seo", f"sku,title,position\n00123,Item,{position}\n".encode())


@pytest.mark.parametrize("day", ["2026-02-30", "2026-13-01", "tomorrow", "2026-01-01T12:00:00"])
def test_rejects_bad_dates(day):
    with pytest.raises(ValueError):
        parse_csv("seo", f"sku,title,date\n00123,Item,{day}\n".encode())


def test_date_objects_normalize_for_storage():
    assert (
        normalize_rows("seo", [{"sku": "001", "title": "Item", "date": date(2026, 1, 15)}])[0][
            "date"
        ]
        == "2026-01-15"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"sku\n00123\n",
        b"cost\n5\n",
        b"sku,cost,api_key\n00123,5,secret\n",
        b"sku,cost,cost\n00123,5,5\n",
    ],
)
def test_missing_duplicate_and_unknown_headers_rejected(payload):
    with pytest.raises(ValueError):
        parse_csv("costs", payload)


def test_unknown_column_cannot_enter_storage():
    with pytest.raises(ValueError, match="неизвестные"):
        normalize_rows("costs", [{"sku": "001", "cost": 5, "api_key": "secret"}])


def test_identifiers_must_remain_strings():
    with pytest.raises(ValueError, match="идентификатор"):
        normalize_rows("costs", [{"sku": 123, "cost": 5}])


@pytest.mark.parametrize("kind", ["costs", "ads", "seo", "competitors"])
def test_duplicates_rejected(kind):
    rows = parse_csv(kind, SAMPLE_CSV[kind])
    with pytest.raises(ValueError, match="повторяется"):
        normalize_rows(kind, rows + rows)


def test_ads_allow_separate_dates_and_skus():
    row = parse_csv("ads", SAMPLE_CSV["ads"])[0]
    other_sku = dict(row, sku="000000002")
    other_date = dict(row, date="2026-01-16")
    assert len(normalize_rows("ads", [row, other_sku, other_date])) == 3


def test_seo_allows_position_history_per_date():
    rows = [
        {"sku": "001", "title": "Item", "date": "2026-01-15"},
        {"sku": "001", "title": "Item", "date": "2026-01-16"},
    ]
    assert len(normalize_rows("seo", rows)) == 2


def test_optional_blank_fields_are_omitted():
    row = parse_csv("seo", b"sku,title,description,keywords,position,date\n001,Item,,,,\n")[0]
    assert row == {"sku": "001", "title": "Item"}


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"sku,cost\n",
        b"sku,cost\n001,\n",
        b"sku,cost\n,5\n",
        b"sku,cost\n001,5,extra\n",
        b"sku,cost\n001,5\x00\n",
    ],
)
def test_empty_malformed_and_null_data_rejected(payload):
    with pytest.raises(ValueError):
        parse_csv("costs", payload)


def test_byte_limit():
    with pytest.raises(ValueError, match="2 МиБ"):
        parse_csv("costs", b"x" * (MAX_BYTES + 1))


def test_row_limit():
    rows = "sku,cost\n" + "".join(f"{index:06d},1\n" for index in range(MAX_ROWS + 1))
    with pytest.raises(ValueError, match="20000"):
        parse_csv("costs", rows.encode())


def test_unknown_import_kind():
    with pytest.raises(ValueError, match="тип"):
        parse_csv("secrets", b"sku,cost\n001,5\n")
