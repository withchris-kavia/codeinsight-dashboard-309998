"""
Database connectivity helpers for the FastAPI service.

Uses SQLAlchemy 2.0 style engine and sessions.
"""

from __future__ import annotations

import os
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# SQLAlchemy ORM base for model declarations.
Base = declarative_base()


# PUBLIC_INTERFACE
def get_database_url() -> str:
    """Resolve the database connection string.

    Precedence:
      1) DATABASE_URL env var (recommended for deployments)
      2) Preview/default local dev connection string.

    Note:
      - The canonical postgresql_db/db_connection.txt in this repo points to port 5000.
        We default to that value for local/dev smoke tests, while still allowing override
        via DATABASE_URL for other environments.
    """
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://appuser:dbuser123@localhost:5000/myapp",
    )


def _create_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine."""
    # pool_pre_ping helps avoid stale pooled connections.
    return create_engine(database_url, pool_pre_ping=True, future=True)


_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


# PUBLIC_INTERFACE
def init_engine() -> Engine:
    """Initialize (or return existing) SQLAlchemy Engine and session factory."""
    global _ENGINE, _SessionLocal
    if _ENGINE is None:
        database_url = get_database_url()
        _ENGINE = _create_engine(database_url)
        _SessionLocal = sessionmaker(bind=_ENGINE, autocommit=False, autoflush=False)
    return _ENGINE


# PUBLIC_INTERFACE
def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy Session."""
    init_engine()
    assert _SessionLocal is not None  # for type checkers
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
