"""
SQLAlchemy ORM models for the CodeInsight Dashboard backend.

These models map to tables created by the PostgreSQL schema in
`postgresql_db/schema/001_init_schema.sql`.

We only define fields needed for OAuth token persistence and minimal user linkage.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db import Base


class User(Base):
    """ORM mapping for `users` table (minimal fields used by OAuth flows)."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Note: org_id exists in schema but is not required for OAuth persistence.
    org_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True, unique=True)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    oauth_identities: Mapped[List["OAuthIdentity"]] = relationship(
        "OAuthIdentity",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class OAuthIdentity(Base):
    """ORM mapping for `oauth_identities` table."""

    __tablename__ = "oauth_identities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    provider: Mapped[str] = mapped_column(String, nullable=False)  # github|gitlab|bitbucket
    provider_user_id: Mapped[str] = mapped_column(Text, nullable=False)
    username: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    access_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    scopes: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user: Mapped[User] = relationship("User", back_populates="oauth_identities")
