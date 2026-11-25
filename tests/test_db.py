from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, StatementError

from app.config import DEFAULT_DATABASE_URL, PROJECT_ROOT, Settings
from app.db import init_db, make_engine, make_session_factory
from app.models import Ticket


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


def _ticket(**overrides):
    fields = {"name": "Ana", "email": "ana@example.com", "message": "Where is my refund?", "predicted_category": "REFUND",
              "confidence": 0.87, "model_version": "test"}
    return Ticket(**(fields | overrides))


def test_settings_default_and_env_override(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert Settings.from_env().database_url == DEFAULT_DATABASE_URL
    assert DEFAULT_DATABASE_URL.endswith(f"{PROJECT_ROOT.as_posix()}/support_desk.db")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///other.db")
    assert Settings.from_env().database_url == "sqlite:///other.db"


def test_init_db_creates_tickets_table(session_factory):
    cols = {c["name"]: c for c in inspect(session_factory.kw["bind"]).get_columns("tickets")}
    assert set(cols) == {"id", "name", "email", "message", "priority", "status", "predicted_category",
                         "confidence", "model_version", "created_at", "suggested_reply", "reply_error",
                         "reply_status", "reply_sources"}
    for c in ("suggested_reply", "reply_error", "reply_status", "reply_sources"):
        assert cols[c]["nullable"]
    assert not cols["message"]["nullable"] and not cols["confidence"]["nullable"]


def test_ticket_roundtrip_defaults(session_factory):
    with session_factory() as s:
        s.add(_ticket())
        s.commit()
    with session_factory() as s:
        t = s.scalars(select(Ticket)).one()
    assert (t.message, t.predicted_category, t.confidence) == ("Where is my refund?", "REFUND", 0.87)
    assert t.suggested_reply is None and t.reply_error is None
    assert (t.priority, t.status) == ("normal", "open")
    assert t.created_at is not None


def test_created_at_is_utc_aware_after_reload(session_factory):
    # Must reload in a fresh session: the in-memory object keeps its tzinfo even
    # when the database round-trip would lose it.
    with session_factory() as s:
        s.add(_ticket())
        s.commit()
    with session_factory() as s:
        t = s.scalars(select(Ticket)).one()
    assert t.created_at.tzinfo is not None and t.created_at.utcoffset() == timedelta(0)


def test_non_utc_aware_datetime_is_converted_to_utc(session_factory):
    ist = timezone(timedelta(hours=5, minutes=30))
    with session_factory() as s:
        s.add(_ticket(created_at=datetime(2026, 1, 1, 12, 0, tzinfo=ist)))
        s.commit()
    with session_factory() as s:
        assert s.scalars(select(Ticket)).one().created_at == datetime(2026, 1, 1, 6, 30, tzinfo=timezone.utc)


def test_naive_datetime_is_rejected(session_factory):
    with session_factory() as s:
        s.add(_ticket(created_at=datetime(2026, 1, 1, 12, 0)))
        with pytest.raises(StatementError):
            s.commit()


@pytest.mark.parametrize("missing", ["name", "email", "message", "predicted_category", "confidence", "model_version"])
def test_required_columns_enforced(session_factory, missing):
    with session_factory() as s:
        s.add(_ticket(**{missing: None}))
        with pytest.raises(IntegrityError):
            s.commit()


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_confidence_range_enforced(session_factory, bad):
    with session_factory() as s:
        s.add(_ticket(confidence=bad))
        with pytest.raises(IntegrityError):
            s.commit()
