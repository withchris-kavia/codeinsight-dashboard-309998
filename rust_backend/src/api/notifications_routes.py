"""
Notification configuration and dispatch routes.

This module provides:
- persistence of per-org/per-user notification configurations (Slack/email),
- a test endpoint for quick validation,
- a dispatch endpoint intended for internal hooks (e.g., analytics/AI summary triggers).

Provider behavior:
- Slack: posts to an incoming webhook URL.
- Email: sends a text email via SMTP.

If provider environment configuration is missing, we return a stubbed response:
  {"dispatched": false, "reason": "missing provider config"}
and persist last_error for observability.

Note: This backend template currently has no auth middleware; callers must supply org_id/user_id.
"""

from __future__ import annotations

import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, or_, text
from sqlalchemy.orm import Session

from src.api.db import get_db_session
from src.api.models import NotificationConfig

router = APIRouter(prefix="/notifications", tags=["Notifications"])

_PROVIDER = Literal["slack", "email"]


def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def _is_missing_env(*names: str) -> bool:
    """Return True if any env var in names is missing/blank."""
    for name in names:
        if not os.getenv(name):
            return True
    return False


def _ensure_notification_configs_table(db: Session) -> None:
    """Create the notification_configs table if it does not exist.

    This keeps preview/dev working even when DB migrations are not wired up.
    """
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS notification_configs (
              id uuid PRIMARY KEY,
              org_id uuid NULL,
              user_id uuid NULL,
              provider text NOT NULL,
              enabled boolean NOT NULL DEFAULT true,
              slack_webhook_url text NULL,
              email_address text NULL,
              last_dispatch_at timestamptz NULL,
              last_error text NULL,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
    )
    # Helpful indexes for lookups
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_notification_configs_org_id ON notification_configs(org_id);"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_notification_configs_user_id ON notification_configs(user_id);"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_notification_configs_provider ON notification_configs(provider);"))
    # Ensure upserts behave sensibly (provider + scope)
    db.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_notification_configs_scope
            ON notification_configs(provider, COALESCE(org_id, '00000000-0000-0000-0000-000000000000'::uuid),
                                   COALESCE(user_id,'00000000-0000-0000-0000-000000000000'::uuid));
            """
        )
    )
    db.commit()


class NotificationConfigUpsertRequest(BaseModel):
    """Request to create or update a notification configuration."""

    provider: _PROVIDER = Field(..., description="Notification provider: slack|email.")
    enabled: bool = Field(True, description="Whether this config is enabled.")

    org_id: Optional[uuid.UUID] = Field(None, description="Optional organization UUID scope.")
    user_id: Optional[uuid.UUID] = Field(None, description="Optional user UUID scope.")

    slack_webhook_url: Optional[str] = Field(
        None,
        description="Slack incoming webhook URL (optional; can fall back to SLACK_WEBHOOK_DEFAULT env var).",
        examples=["https://hooks.slack.com/services/T000/B000/XXX"],
    )
    email_address: Optional[str] = Field(
        None,
        description="Destination email address for provider=email.",
        examples=["alerts@example.com"],
    )

    @field_validator("email_address")
    @classmethod
    def _validate_email_for_provider(cls, v: Optional[str], info):  # type: ignore[override]
        provider = info.data.get("provider")
        if provider == "email" and not v:
            raise ValueError("email_address is required when provider=email")
        return v

    @field_validator("slack_webhook_url")
    @classmethod
    def _validate_slack_webhook_for_provider(cls, v: Optional[str], info):  # type: ignore[override]
        provider = info.data.get("provider")
        # Slack webhook URL can be omitted to rely on SLACK_WEBHOOK_DEFAULT.
        if provider == "slack" and v is not None and not v.startswith("http"):
            raise ValueError("slack_webhook_url must be a valid URL")
        return v

    @field_validator("provider")
    @classmethod
    def _validate_scope_present(cls, v: _PROVIDER, info):  # type: ignore[override]
        # Validation that at least one of org_id/user_id is present is done after model creation (below),
        # because org_id/user_id are not necessarily in info.data at this point depending on parse order.
        return v

    def validate_scope(self) -> None:
        """Ensure at least one scope identifier is supplied."""
        if self.org_id is None and self.user_id is None:
            raise ValueError("At least one of org_id or user_id must be provided.")


class NotificationConfigOut(BaseModel):
    """Response model for a notification configuration row."""

    id: uuid.UUID = Field(..., description="Notification config UUID.")
    provider: _PROVIDER = Field(..., description="Provider: slack|email.")
    enabled: bool = Field(..., description="Whether this config is enabled.")
    org_id: Optional[uuid.UUID] = Field(None, description="Org scope (if any).")
    user_id: Optional[uuid.UUID] = Field(None, description="User scope (if any).")
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL (if set).")
    email_address: Optional[str] = Field(None, description="Email address (if set).")
    last_dispatch_at: Optional[datetime] = Field(None, description="Last successful dispatch timestamp (UTC).")
    last_error: Optional[str] = Field(None, description="Last dispatch error message, if any.")
    created_at: datetime = Field(..., description="Creation timestamp (UTC).")
    updated_at: datetime = Field(..., description="Update timestamp (UTC).")


class NotificationDispatchRequest(BaseModel):
    """Request payload for dispatching a message to configured channels."""

    org_id: Optional[uuid.UUID] = Field(None, description="Optional org UUID to target.")
    user_id: Optional[uuid.UUID] = Field(None, description="Optional user UUID to target.")
    message: str = Field(..., description="Message content to dispatch.", min_length=1, max_length=4000)
    subject: Optional[str] = Field(None, description="Optional email subject.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for observability/debug.")


class NotificationTestRequest(BaseModel):
    """Request payload for sending a test notification."""

    org_id: Optional[uuid.UUID] = Field(None, description="Optional org UUID to target.")
    user_id: Optional[uuid.UUID] = Field(None, description="Optional user UUID to target.")
    message: Optional[str] = Field("Test notification from CodeInsight Dashboard.", description="Optional test message.")


class NotificationDispatchResult(BaseModel):
    """Per-config dispatch result."""

    config_id: uuid.UUID = Field(..., description="Notification config UUID.")
    provider: _PROVIDER = Field(..., description="Provider used.")
    dispatched: bool = Field(..., description="Whether dispatch succeeded.")
    reason: Optional[str] = Field(None, description="If not dispatched, a reason string.")
    error: Optional[str] = Field(None, description="If failed, error string (if any).")


class NotificationDispatchResponse(BaseModel):
    """Dispatch response summarizing what happened across all channels."""

    dispatched_any: bool = Field(..., description="True if any channel dispatched successfully.")
    results: List[NotificationDispatchResult] = Field(..., description="Per-config results.")


def _configs_query(org_id: Optional[uuid.UUID], user_id: Optional[uuid.UUID]) -> Any:
    """Build a SQLAlchemy filter that matches configs for the provided scopes."""
    filters = []
    if org_id is not None:
        filters.append(NotificationConfig.org_id == org_id)
    if user_id is not None:
        filters.append(NotificationConfig.user_id == user_id)

    if not filters:
        # If caller doesn't scope, we don't want to return everything.
        raise HTTPException(status_code=400, detail="At least one of org_id or user_id must be provided.")

    return or_(*filters)


async def _dispatch_slack(
    webhook_url: str,
    message: str,
) -> None:
    """Send a Slack message via incoming webhook."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(webhook_url, json={"text": message})
        resp.raise_for_status()


def _smtp_settings() -> Optional[Dict[str, Any]]:
    """Resolve SMTP settings from environment variables.

    Required env vars:
      - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

    Returns None if any are missing.
    """
    if _is_missing_env("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"):
        return None
    return {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "0") or "0"),
        "user": os.getenv("SMTP_USER"),
        "password": os.getenv("SMTP_PASSWORD"),
        "from": os.getenv("SMTP_FROM"),
    }


def _send_email_smtp(to_addr: str, subject: str, body: str) -> None:
    """Send a text email via SMTP using env configuration."""
    settings = _smtp_settings()
    if settings is None:
        raise RuntimeError("missing provider config")

    msg = EmailMessage()
    msg["From"] = settings["from"]
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    # For simplicity, use STARTTLS on standard ports. This can be adjusted later if needed.
    with smtplib.SMTP(settings["host"], settings["port"], timeout=10) as server:
        server.ehlo()
        try:
            server.starttls()
            server.ehlo()
        except smtplib.SMTPException:
            # Some SMTP servers don't support STARTTLS on their port; proceed without it.
            pass

        server.login(settings["user"], settings["password"])
        server.send_message(msg)


async def _dispatch_one_config(
    db: Session,
    cfg: NotificationConfig,
    message: str,
    subject: Optional[str] = None,
) -> NotificationDispatchResult:
    """Dispatch to a single config and persist observability fields."""
    if not cfg.enabled:
        return NotificationDispatchResult(
            config_id=cfg.id,
            provider=cfg.provider,  # type: ignore[arg-type]
            dispatched=False,
            reason="disabled",
        )

    try:
        if cfg.provider == "slack":
            webhook_url = cfg.slack_webhook_url or os.getenv("SLACK_WEBHOOK_DEFAULT")
            if not webhook_url:
                cfg.last_error = "missing provider config"
                db.add(cfg)
                db.commit()
                return NotificationDispatchResult(
                    config_id=cfg.id,
                    provider="slack",
                    dispatched=False,
                    reason="missing provider config",
                    error=None,
                )
            await _dispatch_slack(webhook_url, message)

        elif cfg.provider == "email":
            if not cfg.email_address:
                cfg.last_error = "missing destination email_address"
                db.add(cfg)
                db.commit()
                return NotificationDispatchResult(
                    config_id=cfg.id,
                    provider="email",
                    dispatched=False,
                    reason="missing destination email_address",
                )

            if _smtp_settings() is None:
                cfg.last_error = "missing provider config"
                db.add(cfg)
                db.commit()
                return NotificationDispatchResult(
                    config_id=cfg.id,
                    provider="email",
                    dispatched=False,
                    reason="missing provider config",
                )

            _send_email_smtp(
                to_addr=cfg.email_address,
                subject=subject or "CodeInsight Notification",
                body=message,
            )
        else:
            cfg.last_error = f"unsupported provider: {cfg.provider}"
            db.add(cfg)
            db.commit()
            return NotificationDispatchResult(
                config_id=cfg.id,
                provider=cfg.provider,  # type: ignore[arg-type]
                dispatched=False,
                reason="unsupported provider",
                error=cfg.last_error,
            )

        cfg.last_dispatch_at = _utcnow()
        cfg.last_error = None
        db.add(cfg)
        db.commit()

        return NotificationDispatchResult(
            config_id=cfg.id,
            provider=cfg.provider,  # type: ignore[arg-type]
            dispatched=True,
        )
    except Exception as e:
        cfg.last_error = str(e)
        db.add(cfg)
        db.commit()
        return NotificationDispatchResult(
            config_id=cfg.id,
            provider=cfg.provider,  # type: ignore[arg-type]
            dispatched=False,
            reason="dispatch failed",
            error=str(e),
        )


@router.post(
    "/configs",
    summary="Create or update notification config",
    description="Creates or updates a notification configuration for an org and/or user scope.",
    response_model=NotificationConfigOut,
)
async def upsert_notification_config(
    payload: NotificationConfigUpsertRequest,
    db: Session = Depends(get_db_session),
) -> NotificationConfigOut:
    """Upsert a notification configuration row."""
    try:
        payload.validate_scope()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    _ensure_notification_configs_table(db)

    # Find existing config for same scope+provider.
    # We intentionally allow either org_id or user_id to be NULL and treat NULL as part of the key.
    existing = (
        db.query(NotificationConfig)
        .filter(
            and_(
                NotificationConfig.provider == payload.provider,
                NotificationConfig.org_id.is_(None) if payload.org_id is None else NotificationConfig.org_id == payload.org_id,
                NotificationConfig.user_id.is_(None) if payload.user_id is None else NotificationConfig.user_id == payload.user_id,
            )
        )
        .one_or_none()
    )

    cfg = existing or NotificationConfig(
        id=uuid.uuid4(),
        provider=payload.provider,
        org_id=payload.org_id,
        user_id=payload.user_id,
        enabled=payload.enabled,
    )

    cfg.enabled = payload.enabled
    cfg.slack_webhook_url = payload.slack_webhook_url if payload.provider == "slack" else None
    cfg.email_address = payload.email_address if payload.provider == "email" else None
    cfg.updated_at = _utcnow()

    db.add(cfg)
    db.commit()
    db.refresh(cfg)

    return NotificationConfigOut.model_validate(cfg, from_attributes=True)


@router.get(
    "/configs",
    summary="List notification configs",
    description="Lists current notification configurations for a given org_id and/or user_id scope.",
    response_model=List[NotificationConfigOut],
)
async def list_notification_configs(
    org_id: Optional[uuid.UUID] = Query(None, description="Optional org UUID to filter."),
    user_id: Optional[uuid.UUID] = Query(None, description="Optional user UUID to filter."),
    db: Session = Depends(get_db_session),
) -> List[NotificationConfigOut]:
    """List configs for the requested scope."""
    _ensure_notification_configs_table(db)

    q = db.query(NotificationConfig).filter(_configs_query(org_id, user_id)).order_by(NotificationConfig.created_at.asc())
    rows = q.all()
    return [NotificationConfigOut.model_validate(r, from_attributes=True) for r in rows]


@router.post(
    "/test",
    summary="Send a test notification",
    description="Sends a test notification using current enabled configs for the provided scope. "
    "If provider env configuration is missing, returns stubbed results and persists last_error.",
    response_model=NotificationDispatchResponse,
)
async def test_notification(
    payload: NotificationTestRequest,
    db: Session = Depends(get_db_session),
) -> NotificationDispatchResponse:
    """Send a test message across enabled configs for scope."""
    _ensure_notification_configs_table(db)
    message = payload.message or "Test notification from CodeInsight Dashboard."
    dispatch_payload = NotificationDispatchRequest(org_id=payload.org_id, user_id=payload.user_id, message=message)
    return await dispatch_notification(dispatch_payload, db=db)


@router.post(
    "/dispatch",
    summary="Dispatch message to configured channels",
    description="Internal hook to dispatch a message payload to all enabled notification configs for the given org/user scope. "
    "If provider configuration is missing, returns stubbed results and persists last_error.",
    response_model=NotificationDispatchResponse,
)
async def dispatch_notification(
    payload: NotificationDispatchRequest,
    db: Session = Depends(get_db_session),
) -> NotificationDispatchResponse:
    """Dispatch a message to all enabled configs matching the provided scope."""
    _ensure_notification_configs_table(db)

    # Pull matching configs. We allow either org_id or user_id filtering (OR), so caller can target org or user.
    q = db.query(NotificationConfig).filter(_configs_query(payload.org_id, payload.user_id)).order_by(NotificationConfig.created_at.asc())
    configs = q.all()

    if not configs:
        return NotificationDispatchResponse(
            dispatched_any=False,
            results=[],
        )

    results: List[NotificationDispatchResult] = []
    for cfg in configs:
        results.append(await _dispatch_one_config(db, cfg, payload.message, subject=payload.subject))

    return NotificationDispatchResponse(
        dispatched_any=any(r.dispatched for r in results),
        results=results,
    )
