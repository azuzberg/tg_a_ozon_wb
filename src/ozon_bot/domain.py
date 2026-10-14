from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Начало периода позже окончания.")
        if self.end > date.today():
            raise ValueError("Период не может заканчиваться в будущем.")

    @classmethod
    def last_days(cls, days: int) -> "Period":
        today = date.today()
        return cls(today - timedelta(days=days - 1), today, f"{days} дней")


@dataclass
class Snapshot:
    transactions: list[dict[str, Any]] = field(default_factory=list)
    postings: list[dict[str, Any]] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    stocks: list[dict[str, Any]] = field(default_factory=list)
    daily: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisContext:
    snapshot: Snapshot
    period: Period
    costs: dict[str, float] = field(default_factory=dict)
    tax_rate: float = 0.0
    ads: list[dict[str, Any]] = field(default_factory=list)
    seo: list[dict[str, Any]] = field(default_factory=list)
    competitors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Report:
    title: str
    text: str
    facts: dict[str, Any] = field(default_factory=dict)
