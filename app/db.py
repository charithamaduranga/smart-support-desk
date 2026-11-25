"""SQLAlchemy engine/session setup. Engines are created per app (not at import
time) so tests can point each app instance at its own temporary database."""
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    # FastAPI runs sync routes in a threadpool; SQLite connections must be
    # usable from a thread other than the one that created them.
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create tables that do not exist yet (no migrations; see NOTES.md)."""
    import app.models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(engine)
