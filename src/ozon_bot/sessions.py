import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .domain import AnalysisContext, Period, Snapshot


@dataclass
class Session:
    user_id: int
    chat_id: int
    name: str
    client_id: str
    api_key: str = field(repr=False)
    ttl_minutes: int = 60
    id: str = field(default_factory=lambda: str(uuid4()))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    period: Period = field(default_factory=lambda: Period.last_days(30))
    tax_rate: float = 6.0
    tax_configured: bool = False
    costs: dict[str, float] = field(default_factory=dict)
    ads: list[dict] = field(default_factory=list)
    seo: list[dict] = field(default_factory=list)
    competitors: list[dict] = field(default_factory=list)
    performance_id: str = ""
    performance_secret: str = field(default="", repr=False)
    snapshot: Snapshot | None = None
    cache_time: datetime | None = None
    ads_period: Period | None = None
    ads_time: datetime | None = None
    ads_source: str = "none"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    state: str | None = None
    draft: dict[str, str] = field(default_factory=dict, repr=False)

    def touch(self) -> None:
        self.expires_at = datetime.now(UTC) + timedelta(minutes=self.ttl_minutes)

    def context(self) -> AnalysisContext:
        if self.snapshot is None:
            raise ValueError("Данные магазина ещё не загружены.")
        snapshot = self.snapshot
        if not self.tax_configured:
            snapshot = replace(
                snapshot,
                warnings=[
                    *snapshot.warnings,
                    "Налоговая ставка не задана: предварительно используется 6% от выручки.",
                ],
            )
        return AnalysisContext(
            snapshot,
            self.period,
            self.costs,
            self.tax_rate,
            self.ads,
            self.seo,
            self.competitors,
        )

    def clear(self) -> None:
        self.api_key = ""
        self.client_id = ""
        self.performance_secret = ""
        self.performance_id = ""
        self.name = ""
        self.snapshot = None
        self.costs.clear()
        self.ads.clear()
        self.seo.clear()
        self.competitors.clear()
        self.draft.clear()
        self.state = None
