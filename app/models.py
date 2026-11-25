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


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_confidence_range"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message: Mapped[str] = mapped_column(Text)
    predicted_category: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float]  # top-class probability from the classifier (not calibrated)
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, index=True)
    # Filled in by Part B (FAQ-grounded LLM reply); nullable so Part A needs no schema change later.
    suggested_reply: Mapped[str | None] = mapped_column(Text)
    reply_error: Mapped[str | None] = mapped_column(Text)
