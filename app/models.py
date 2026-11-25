"""Database models."""
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, String, Text, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Store datetimes as naive UTC and return them timezone-aware.

    SQLite has no timezone type: DateTime(timezone=True) silently returns
    naive values after a reload, so we normalise explicitly.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; pass a timezone-aware value")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return None if value is None else value.replace(tzinfo=timezone.utc)


STATUSES = ("open", "in_progress", "resolved")
PRIORITIES = ("low", "normal", "high")


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_confidence_range"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(254))
    message: Mapped[str] = mapped_column(Text)
    # Team-assigned; no automatic priority rule.
    priority: Mapped[str] = mapped_column(String(16), default="normal")
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    predicted_category: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float]  # top-class probability from the classifier (not calibrated)
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, index=True)
    # Part B reply draft. reply_status: generated | no_match | error | disabled.
    suggested_reply: Mapped[str | None] = mapped_column(Text)
    reply_error: Mapped[str | None] = mapped_column(Text)
    reply_status: Mapped[str | None] = mapped_column(String(16))
    reply_sources: Mapped[str | None] = mapped_column(Text)  # JSON list of {id, question, score}
