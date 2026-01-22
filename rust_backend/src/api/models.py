"""
SQLAlchemy ORM models for the CodeInsight Dashboard backend.

These models map to tables created by the PostgreSQL schema in
`postgresql_db/schema/001_init_schema.sql`.

We include:
- Orgs / Users (minimal)
- OAuth identities (for login flows)
- Repos + GitEvents (for webhook ingestion)

Design notes:
- The webhook ingestion needs `org_id` and `repo_id` for `git_events` (NOT NULL in schema).
  When a webhook arrives for a repo we have not seen before, we create:
  - a default Org (slug=default) if one does not exist, then
  - the Repo row in that org, then
  - GitEvent rows linked to that repo.

This keeps ingestion functional even before an explicit "connect org/repo" UI exists.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db import Base


class Org(Base):
    """ORM mapping for `orgs` table."""

    __tablename__ = "orgs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    plan_tier: Mapped[str] = mapped_column(Text, nullable=False, default="free")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    users: Mapped[List["User"]] = relationship("User", back_populates="org")
    repos: Mapped[List["Repo"]] = relationship("Repo", back_populates="org")


class User(Base):
    """ORM mapping for `users` table (minimal fields used by OAuth flows)."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    org_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("orgs.id"), nullable=True)

    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True, unique=True)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    org: Mapped[Optional[Org]] = relationship("Org", back_populates="users")

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


class Repo(Base):
    """ORM mapping for `repos` table."""

    __tablename__ = "repos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orgs.id", ondelete="CASCADE"), nullable=False)

    provider: Mapped[str] = mapped_column(Text, nullable=False)  # github|gitlab|bitbucket
    external_id: Mapped[str] = mapped_column(Text, nullable=False)  # provider repo id
    owner: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)  # owner/name
    default_branch: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_private: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    org: Mapped[Org] = relationship("Org", back_populates="repos")
    git_events: Mapped[List["GitEvent"]] = relationship("GitEvent", back_populates="repo")


class GitEvent(Base):
    """ORM mapping for `git_events` table (normalized event records)."""

    __tablename__ = "git_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orgs.id", ondelete="CASCADE"), nullable=False)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)

    provider: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)  # commit | pull_request | merge | ...
    external_event_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    actor_provider_user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_username: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    head_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    commit_sha: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    merge_commit_sha: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pr_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pr_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_state: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    raw_payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    repo: Mapped[Repo] = relationship("Repo", back_populates="git_events")


class AnalyticsDevDaily(Base):
    """ORM mapping for `analytics_dev_daily` table (per-developer daily rollups)."""

    __tablename__ = "analytics_dev_daily"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orgs.id", ondelete="CASCADE"), nullable=False)
    repo_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=True)

    # Store as date (SQLAlchemy Date works well with Postgres DATE)
    from sqlalchemy import Date  # local import to avoid reordering existing imports

    day: Mapped[Any] = mapped_column(Date, nullable=False)

    actor_provider_user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_username: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    commits_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prs_opened_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prs_merged_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    merges_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    lines_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lines_deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AnalyticsRepoDaily(Base):
    """ORM mapping for `analytics_repo_daily` table (per-repo daily rollups)."""

    __tablename__ = "analytics_repo_daily"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orgs.id", ondelete="CASCADE"), nullable=False)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)

    from sqlalchemy import Date  # local import to avoid reordering existing imports

    day: Mapped[Any] = mapped_column(Date, nullable=False)

    commits_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prs_opened_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prs_merged_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    merges_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    active_devs_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AiSummary(Base):
    """ORM mapping for persisted AI-generated summaries.

    Notes:
      - The DB schema may be created by a separate DB container migration.
      - The API router also ensures the table exists at runtime (CREATE TABLE IF NOT EXISTS),
        so local/dev previews keep working even when migrations lag.
    """

    __tablename__ = "ai_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Nullable scope pointers (requested fields)
    repo_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    org_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    subject_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="One of: repo|org|user|system",
    )
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    summary_text: Mapped[str] = mapped_column(Text, nullable=False)

    provider: Mapped[str] = mapped_column(Text, nullable=False, default="openai")
    model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    tokens_in: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
