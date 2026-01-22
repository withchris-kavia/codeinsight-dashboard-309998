"""
Webhook endpoints for GitHub, GitLab, and Bitbucket.

Implements:
- POST /webhooks/github
- POST /webhooks/gitlab
- POST /webhooks/bitbucket

Security:
- GitHub: verify X-Hub-Signature-256 (HMAC SHA-256) using GITHUB_WEBHOOK_SECRET.
- GitLab: verify X-Gitlab-Token matches GITLAB_WEBHOOK_TOKEN.
- Bitbucket: if BITBUCKET_WEBHOOK_SECRET is configured, verify HMAC SHA-256 against X-Hub-Signature.

Persistence:
- Normalizes incoming webhook payloads into `git_events` rows linked to `repos` and `orgs`.
- Uses idempotency where providers supply a delivery UUID/ID by storing it as `external_event_id`.
  The DB schema includes a partial unique index on (provider, external_event_id) when external_event_id is not null.

Performance:
- Returns 2xx quickly after best-effort persistence. Any DB failures are logged and return 202 to avoid
  repeated redeliveries causing outages (the event can be retried later).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.api.db import get_db_session
from src.api.models import GitEvent, Org, Repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def _get_env_optional(name: str) -> Optional[str]:
    """Get environment variable if set and non-empty."""
    val = os.getenv(name)
    return val if val and val.strip() else None


def _get_env_required(name: str) -> str:
    """Get an environment variable or raise HTTP 500 with clear message."""
    val = _get_env_optional(name)
    if not val:
        raise HTTPException(status_code=500, detail=f"Server webhook is not configured: missing {name}.")
    return val


def _hmac_sha256_hex(secret: str, body: bytes) -> str:
    """Compute lowercase hex digest for HMAC-SHA256."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _parse_signature_header(signature_header: str) -> Tuple[Optional[str], str]:
    """
    Parse a signature header into (algorithm, hex_digest).

    Supports:
    - "sha256=<hex>"
    - "<hex>" (algorithm unknown)
    """
    value = (signature_header or "").strip()
    if not value:
        return None, ""
    if "=" in value:
        alg, digest = value.split("=", 1)
        return alg.strip().lower(), digest.strip().lower()
    return None, value.lower()


def _verify_github_signature(body: bytes, signature_header: Optional[str]) -> None:
    """Verify GitHub X-Hub-Signature-256 using GITHUB_WEBHOOK_SECRET."""
    secret = _get_env_required("GITHUB_WEBHOOK_SECRET")
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header.")
    alg, got = _parse_signature_header(signature_header)
    if alg not in (None, "sha256"):
        raise HTTPException(status_code=401, detail="Unsupported GitHub signature algorithm.")
    expected = _hmac_sha256_hex(secret, body)
    if not hmac.compare_digest(expected, got):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature.")


def _verify_gitlab_token(token_header: Optional[str]) -> None:
    """Verify GitLab X-Gitlab-Token matches configured GITLAB_WEBHOOK_TOKEN."""
    expected = _get_env_required("GITLAB_WEBHOOK_TOKEN")
    got = (token_header or "").strip()
    if not got:
        raise HTTPException(status_code=401, detail="Missing X-Gitlab-Token header.")
    if not hmac.compare_digest(expected, got):
        raise HTTPException(status_code=401, detail="Invalid GitLab webhook token.")


def _verify_bitbucket_signature_if_configured(body: bytes, signature_header: Optional[str]) -> None:
    """
    Verify Bitbucket Cloud signature if BITBUCKET_WEBHOOK_SECRET is configured.

    Bitbucket Cloud sends HMAC SHA-256 as X-Hub-Signature (commonly "sha256=<hex>").
    If the secret is not configured, we accept the request (per user instruction).
    """
    secret = _get_env_optional("BITBUCKET_WEBHOOK_SECRET")
    if not secret:
        return

    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature header.")
    alg, got = _parse_signature_header(signature_header)
    if alg not in (None, "sha256"):
        raise HTTPException(status_code=401, detail="Unsupported Bitbucket signature algorithm.")
    expected = _hmac_sha256_hex(secret, body)
    if not hmac.compare_digest(expected, got):
        raise HTTPException(status_code=401, detail="Invalid Bitbucket webhook signature.")


def _ensure_default_org(db: Session) -> Org:
    """Create or return a default org to associate webhook events to, if no org mapping is available."""
    org = db.scalar(select(Org).where(Org.slug == "default"))
    if org:
        return org
    now = datetime.now(timezone.utc)
    org = Org(slug="default", name="Default", plan_tier="free", updated_at=now)
    db.add(org)
    db.flush()
    return org


def _get_or_create_repo(
    db: Session,
    *,
    provider: str,
    external_id: str,
    owner: str,
    name: str,
    full_name: str,
    default_branch: Optional[str],
    is_private: bool,
) -> Repo:
    """Lookup repo by (provider, external_id) else create it under the default org."""
    repo = db.scalar(select(Repo).where(Repo.provider == provider, Repo.external_id == external_id))
    if repo:
        # Best-effort refresh of mutable fields.
        repo.owner = owner
        repo.name = name
        repo.full_name = full_name
        repo.default_branch = default_branch
        repo.is_private = bool(is_private)
        repo.updated_at = datetime.now(timezone.utc)
        return repo

    org = _ensure_default_org(db)
    repo = Repo(
        org_id=org.id,
        provider=provider,
        external_id=external_id,
        owner=owner,
        name=name,
        full_name=full_name,
        default_branch=default_branch,
        is_private=bool(is_private),
        is_active=True,
        updated_at=datetime.now(timezone.utc),
    )
    db.add(repo)
    db.flush()
    return repo


def _safe_dt(value: Any) -> datetime:
    """Parse best-effort timestamp; falls back to now(UTC) if unknown/unparseable."""
    if isinstance(value, str) and value:
        # Handle common ISO8601 forms; Python's fromisoformat doesn't accept Z in older variants reliably.
        # We'll do simple normalization.
        v = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _insert_event_idempotent(db: Session, event: GitEvent) -> bool:
    """
    Insert a GitEvent row with idempotency via DB unique index.

    Returns True if inserted, False if duplicate (by provider+external_event_id).
    """
    try:
        db.add(event)
        db.commit()
        return True
    except IntegrityError:
        db.rollback()
        # Likely (provider, external_event_id) duplicate.
        return False
    except Exception:
        db.rollback()
        raise


def _bulk_insert_events(db: Session, events: List[GitEvent]) -> Dict[str, int]:
    """Insert multiple events with per-row idempotency behavior."""
    inserted = 0
    duplicates = 0
    errors = 0

    for ev in events:
        try:
            ok = _insert_event_idempotent(db, ev)
            if ok:
                inserted += 1
            else:
                duplicates += 1
        except Exception as e:
            errors += 1
            logger.exception("Failed to insert webhook event: %s", e)

    return {"inserted": inserted, "duplicates": duplicates, "errors": errors}


def _github_repo_info(payload: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str], bool]:
    """Extract repo identity from GitHub payload."""
    repo = payload.get("repository") or {}
    external_id = str(repo.get("id") or "")
    full_name = str(repo.get("full_name") or "")
    name = str(repo.get("name") or "")
    owner = ""
    owner_obj = repo.get("owner")
    if isinstance(owner_obj, dict):
        owner = str(owner_obj.get("login") or owner_obj.get("name") or "")
    if not owner and "/" in full_name:
        owner = full_name.split("/", 1)[0]
    default_branch = repo.get("default_branch") if isinstance(repo.get("default_branch"), str) else None
    is_private = bool(repo.get("private")) if "private" in repo else True

    if not external_id or not full_name or not name or not owner:
        raise HTTPException(status_code=400, detail="GitHub payload missing repository identification fields.")
    return external_id, owner, name, full_name, default_branch, is_private


def _gitlab_repo_info(payload: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str], bool]:
    """Extract repo identity from GitLab payload (project object)."""
    project = payload.get("project") or payload.get("repository") or {}
    external_id = str(project.get("id") or "")
    full_name = str(project.get("path_with_namespace") or project.get("name_with_namespace") or "")
    name = str(project.get("name") or "")
    owner = ""
    namespace = project.get("namespace")
    if isinstance(namespace, dict):
        owner = str(namespace.get("full_path") or namespace.get("path") or "")
    if not owner and "/" in full_name:
        owner = full_name.split("/", 1)[0]
    default_branch = project.get("default_branch") if isinstance(project.get("default_branch"), str) else None
    visibility = (project.get("visibility") if isinstance(project.get("visibility"), str) else "") or ""
    is_private = visibility.lower() != "public"

    if not external_id or not full_name or not name or not owner:
        raise HTTPException(status_code=400, detail="GitLab payload missing project identification fields.")
    return external_id, owner, name, full_name, default_branch, is_private


def _bitbucket_repo_info(payload: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str], bool]:
    """Extract repo identity from Bitbucket Cloud payload."""
    repo = payload.get("repository") or {}
    external_id = str(repo.get("uuid") or repo.get("full_name") or "")
    full_name = str(repo.get("full_name") or "")
    name = str(repo.get("name") or "")
    owner = ""
    owner_obj = repo.get("owner")
    if isinstance(owner_obj, dict):
        owner = str(owner_obj.get("username") or owner_obj.get("nickname") or owner_obj.get("display_name") or "")
    if not owner and "/" in full_name:
        owner = full_name.split("/", 1)[0]

    # Bitbucket doesn't always include default branch on webhook payload.
    default_branch = None
    is_private = bool(repo.get("is_private")) if "is_private" in repo else True

    if not external_id or not full_name or not name or not owner:
        raise HTTPException(status_code=400, detail="Bitbucket payload missing repository identification fields.")
    return external_id, owner, name, full_name, default_branch, is_private


def _github_events(
    *,
    repo: Repo,
    payload: Dict[str, Any],
    event_name: str,
    delivery_id: Optional[str],
) -> List[GitEvent]:
    """Convert GitHub webhook payload into one or more GitEvent rows."""
    now = datetime.now(timezone.utc)
    events: List[GitEvent] = []

    if event_name == "push":
        ref = payload.get("ref") if isinstance(payload.get("ref"), str) else None
        pusher = payload.get("pusher") if isinstance(payload.get("pusher"), dict) else {}
        actor_username = pusher.get("name") if isinstance(pusher.get("name"), str) else None
        actor_email = pusher.get("email") if isinstance(pusher.get("email"), str) else None

        commits = payload.get("commits") if isinstance(payload.get("commits"), list) else []
        # If no commits list, still record a single push event.
        if not commits:
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="github",
                    event_type="commit",
                    external_event_id=delivery_id,
                    actor_username=actor_username,
                    actor_email=actor_email,
                    event_timestamp=_safe_dt(payload.get("head_commit", {}).get("timestamp") if isinstance(payload.get("head_commit"), dict) else None),
                    ref=ref,
                    raw_payload=payload,
                    metadata={"github_event": "push", "commit_count": 0},
                )
            )
            return events

        for c in commits:
            if not isinstance(c, dict):
                continue
            sha = c.get("id") if isinstance(c.get("id"), str) else None
            author = c.get("author") if isinstance(c.get("author"), dict) else {}
            ev_ts = _safe_dt(c.get("timestamp"))
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="github",
                    event_type="commit",
                    external_event_id=f"{delivery_id}:{sha}" if (delivery_id and sha) else delivery_id,
                    actor_username=author.get("username") if isinstance(author.get("username"), str) else actor_username,
                    actor_email=author.get("email") if isinstance(author.get("email"), str) else actor_email,
                    event_timestamp=ev_ts,
                    ref=ref,
                    commit_sha=sha,
                    raw_payload=c,  # store per-commit payload slice to keep rows smaller
                    metadata={"github_event": "push"},
                )
            )
        return events

    if event_name == "pull_request":
        action = payload.get("action") if isinstance(payload.get("action"), str) else None
        pr = payload.get("pull_request") if isinstance(payload.get("pull_request"), dict) else {}
        number = payload.get("number")
        pr_number = int(number) if isinstance(number, int) else None
        pr_title = pr.get("title") if isinstance(pr.get("title"), str) else None
        pr_url = pr.get("html_url") if isinstance(pr.get("html_url"), str) else None
        merged = bool(pr.get("merged")) if "merged" in pr else False
        pr_state = "merged" if merged else (pr.get("state") if isinstance(pr.get("state"), str) else None)

        user = pr.get("user") if isinstance(pr.get("user"), dict) else {}
        actor_provider_user_id = str(user.get("id")) if user.get("id") is not None else None
        actor_username = user.get("login") if isinstance(user.get("login"), str) else None

        base_ref = None
        head_ref = None
        base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
        head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
        if isinstance(base.get("ref"), str):
            base_ref = base.get("ref")
        if isinstance(head.get("ref"), str):
            head_ref = head.get("ref")

        ev_ts = _safe_dt(pr.get("updated_at") or pr.get("created_at")) if isinstance(pr, dict) else now

        # Always store a pull_request event record.
        events.append(
            GitEvent(
                org_id=repo.org_id,
                repo_id=repo.id,
                provider="github",
                event_type="pull_request",
                external_event_id=delivery_id,
                actor_provider_user_id=actor_provider_user_id,
                actor_username=actor_username,
                event_timestamp=ev_ts,
                base_ref=base_ref,
                head_ref=head_ref,
                pr_number=pr_number,
                pr_title=pr_title,
                pr_state=pr_state,
                pr_url=pr_url,
                raw_payload=payload,
                metadata={"github_event": "pull_request", "action": action, "merged": merged},
            )
        )

        # If merged, optionally add a separate "merge" event (useful for analytics).
        if merged or action == "closed" and pr_state == "merged":
            merge_sha = pr.get("merge_commit_sha") if isinstance(pr.get("merge_commit_sha"), str) else None
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="github",
                    event_type="merge",
                    external_event_id=f"{delivery_id}:merge" if delivery_id else None,
                    actor_provider_user_id=actor_provider_user_id,
                    actor_username=actor_username,
                    event_timestamp=ev_ts,
                    base_ref=base_ref,
                    head_ref=head_ref,
                    merge_commit_sha=merge_sha,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    pr_state="merged",
                    pr_url=pr_url,
                    raw_payload=payload,
                    metadata={"github_event": "pull_request", "action": action, "merged": True},
                )
            )

        return events

    # Default: store raw event as metadata-only row (still useful for audit/debug).
    events.append(
        GitEvent(
            org_id=repo.org_id,
            repo_id=repo.id,
            provider="github",
            event_type="unknown",
            external_event_id=delivery_id,
            event_timestamp=now,
            raw_payload=payload,
            metadata={"github_event": event_name},
        )
    )
    return events


def _gitlab_events(
    *,
    repo: Repo,
    payload: Dict[str, Any],
    event_name: str,
    delivery_id: Optional[str],
) -> List[GitEvent]:
    """Convert GitLab webhook payload into one or more GitEvent rows."""
    now = datetime.now(timezone.utc)
    events: List[GitEvent] = []

    # GitLab push events
    if event_name.lower() in ("push hook", "push_hook", "push"):
        ref = payload.get("ref") if isinstance(payload.get("ref"), str) else None
        user_username = payload.get("user_username") if isinstance(payload.get("user_username"), str) else None
        user_email = payload.get("user_email") if isinstance(payload.get("user_email"), str) else None
        user_id = str(payload.get("user_id")) if payload.get("user_id") is not None else None

        commits = payload.get("commits") if isinstance(payload.get("commits"), list) else []
        if not commits:
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="gitlab",
                    event_type="commit",
                    external_event_id=delivery_id,
                    actor_provider_user_id=user_id,
                    actor_username=user_username,
                    actor_email=user_email,
                    event_timestamp=_safe_dt(payload.get("timestamp")),
                    ref=ref,
                    raw_payload=payload,
                    metadata={"gitlab_event": "push", "commit_count": 0},
                )
            )
            return events

        for c in commits:
            if not isinstance(c, dict):
                continue
            sha = c.get("id") if isinstance(c.get("id"), str) else None
            author_name = c.get("author", {}).get("name") if isinstance(c.get("author"), dict) else None
            author_email = c.get("author", {}).get("email") if isinstance(c.get("author"), dict) else None
            ev_ts = _safe_dt(c.get("timestamp") or payload.get("timestamp"))
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="gitlab",
                    event_type="commit",
                    external_event_id=f"{delivery_id}:{sha}" if (delivery_id and sha) else delivery_id,
                    actor_provider_user_id=user_id,
                    actor_username=author_name or user_username,
                    actor_email=author_email or user_email,
                    event_timestamp=ev_ts,
                    ref=ref,
                    commit_sha=sha,
                    raw_payload=c,
                    metadata={"gitlab_event": "push"},
                )
            )
        return events

    # Merge request events
    if event_name.lower() in ("merge request hook", "merge_request_hook", "merge request"):
        obj = payload.get("object_attributes") if isinstance(payload.get("object_attributes"), dict) else {}
        action = payload.get("object_attributes", {}).get("action") if isinstance(payload.get("object_attributes"), dict) else None
        state = obj.get("state") if isinstance(obj.get("state"), str) else None
        merged = bool(obj.get("state") == "merged") or bool(obj.get("merged_at"))
        pr_state = "merged" if merged else state

        iid = obj.get("iid")
        pr_number = int(iid) if isinstance(iid, int) else None
        pr_title = obj.get("title") if isinstance(obj.get("title"), str) else None
        pr_url = obj.get("url") if isinstance(obj.get("url"), str) else None

        last_updated = obj.get("updated_at") or obj.get("created_at")
        ev_ts = _safe_dt(last_updated)

        user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
        actor_provider_user_id = str(user.get("id")) if user.get("id") is not None else None
        actor_username = user.get("username") if isinstance(user.get("username"), str) else None
        actor_email = user.get("email") if isinstance(user.get("email"), str) else None

        source_branch = obj.get("source_branch") if isinstance(obj.get("source_branch"), str) else None
        target_branch = obj.get("target_branch") if isinstance(obj.get("target_branch"), str) else None
        merge_sha = obj.get("merge_commit_sha") if isinstance(obj.get("merge_commit_sha"), str) else None

        events.append(
            GitEvent(
                org_id=repo.org_id,
                repo_id=repo.id,
                provider="gitlab",
                event_type="pull_request",
                external_event_id=delivery_id,
                actor_provider_user_id=actor_provider_user_id,
                actor_username=actor_username,
                actor_email=actor_email,
                event_timestamp=ev_ts,
                base_ref=target_branch,
                head_ref=source_branch,
                merge_commit_sha=merge_sha,
                pr_number=pr_number,
                pr_title=pr_title,
                pr_state=pr_state,
                pr_url=pr_url,
                raw_payload=payload,
                metadata={"gitlab_event": "merge_request", "action": action, "merged": merged},
            )
        )

        if merged:
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="gitlab",
                    event_type="merge",
                    external_event_id=f"{delivery_id}:merge" if delivery_id else None,
                    actor_provider_user_id=actor_provider_user_id,
                    actor_username=actor_username,
                    actor_email=actor_email,
                    event_timestamp=ev_ts,
                    base_ref=target_branch,
                    head_ref=source_branch,
                    merge_commit_sha=merge_sha,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    pr_state="merged",
                    pr_url=pr_url,
                    raw_payload=payload,
                    metadata={"gitlab_event": "merge_request", "merged": True},
                )
            )

        return events

    events.append(
        GitEvent(
            org_id=repo.org_id,
            repo_id=repo.id,
            provider="gitlab",
            event_type="unknown",
            external_event_id=delivery_id,
            event_timestamp=now,
            raw_payload=payload,
            metadata={"gitlab_event": event_name},
        )
    )
    return events


def _bitbucket_events(
    *,
    repo: Repo,
    payload: Dict[str, Any],
    event_key: str,
    delivery_id: Optional[str],
) -> List[GitEvent]:
    """Convert Bitbucket Cloud webhook payload into one or more GitEvent rows."""
    now = datetime.now(timezone.utc)
    events: List[GitEvent] = []

    if event_key == "repo:push":
        push = payload.get("push") if isinstance(payload.get("push"), dict) else {}
        changes = push.get("changes") if isinstance(push.get("changes"), list) else []

        # Best-effort actor from "actor" object.
        actor = payload.get("actor") if isinstance(payload.get("actor"), dict) else {}
        actor_provider_user_id = str(actor.get("account_id")) if actor.get("account_id") is not None else None
        actor_username = actor.get("username") if isinstance(actor.get("username"), str) else actor.get("nickname")

        for ch in changes:
            if not isinstance(ch, dict):
                continue
            new = ch.get("new") if isinstance(ch.get("new"), dict) else {}
            ref = new.get("name") if isinstance(new.get("name"), str) else None
            # commits are not fully included; store as one event per change.
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="bitbucket",
                    event_type="commit",
                    external_event_id=f"{delivery_id}:{ref}" if (delivery_id and ref) else delivery_id,
                    actor_provider_user_id=actor_provider_user_id,
                    actor_username=actor_username if isinstance(actor_username, str) else None,
                    event_timestamp=now,
                    ref=ref,
                    raw_payload=payload,
                    metadata={"bitbucket_event": "repo:push"},
                )
            )

        if not events:
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="bitbucket",
                    event_type="commit",
                    external_event_id=delivery_id,
                    actor_provider_user_id=actor_provider_user_id,
                    actor_username=actor_username if isinstance(actor_username, str) else None,
                    event_timestamp=now,
                    raw_payload=payload,
                    metadata={"bitbucket_event": "repo:push", "changes": 0},
                )
            )
        return events

    if event_key.startswith("pullrequest:"):
        pr = payload.get("pullrequest") if isinstance(payload.get("pullrequest"), dict) else {}
        pr_id = pr.get("id")
        pr_number = int(pr_id) if isinstance(pr_id, int) else None
        pr_title = pr.get("title") if isinstance(pr.get("title"), str) else None
        pr_url = None
        links = pr.get("links") if isinstance(pr.get("links"), dict) else {}
        html = links.get("html") if isinstance(links.get("html"), dict) else {}
        pr_url = html.get("href") if isinstance(html.get("href"), str) else None

        state = pr.get("state") if isinstance(pr.get("state"), str) else None
        # Bitbucket uses "MERGED"/"OPEN"/"DECLINED"
        pr_state = state.lower() if isinstance(state, str) else None

        actor = payload.get("actor") if isinstance(payload.get("actor"), dict) else {}
        actor_provider_user_id = str(actor.get("account_id")) if actor.get("account_id") is not None else None
        actor_username = actor.get("username") if isinstance(actor.get("username"), str) else actor.get("nickname")

        source = pr.get("source") if isinstance(pr.get("source"), dict) else {}
        dest = pr.get("destination") if isinstance(pr.get("destination"), dict) else {}
        head_ref = source.get("branch", {}).get("name") if isinstance(source.get("branch"), dict) else None
        base_ref = dest.get("branch", {}).get("name") if isinstance(dest.get("branch"), dict) else None

        events.append(
            GitEvent(
                org_id=repo.org_id,
                repo_id=repo.id,
                provider="bitbucket",
                event_type="pull_request",
                external_event_id=delivery_id,
                actor_provider_user_id=actor_provider_user_id,
                actor_username=actor_username if isinstance(actor_username, str) else None,
                event_timestamp=now,
                base_ref=base_ref if isinstance(base_ref, str) else None,
                head_ref=head_ref if isinstance(head_ref, str) else None,
                pr_number=pr_number,
                pr_title=pr_title,
                pr_state=pr_state,
                pr_url=pr_url,
                raw_payload=payload,
                metadata={"bitbucket_event": event_key},
            )
        )

        if event_key == "pullrequest:fulfilled" or pr_state == "merged":
            events.append(
                GitEvent(
                    org_id=repo.org_id,
                    repo_id=repo.id,
                    provider="bitbucket",
                    event_type="merge",
                    external_event_id=f"{delivery_id}:merge" if delivery_id else None,
                    actor_provider_user_id=actor_provider_user_id,
                    actor_username=actor_username if isinstance(actor_username, str) else None,
                    event_timestamp=now,
                    base_ref=base_ref if isinstance(base_ref, str) else None,
                    head_ref=head_ref if isinstance(head_ref, str) else None,
                    pr_number=pr_number,
                    pr_title=pr_title,
                    pr_state="merged",
                    pr_url=pr_url,
                    raw_payload=payload,
                    metadata={"bitbucket_event": event_key, "merged": True},
                )
            )

        return events

    events.append(
        GitEvent(
            org_id=repo.org_id,
            repo_id=repo.id,
            provider="bitbucket",
            event_type="unknown",
            external_event_id=delivery_id,
            event_timestamp=now,
            raw_payload=payload,
            metadata={"bitbucket_event": event_key},
        )
    )
    return events


# PUBLIC_INTERFACE
@router.post(
    "/github",
    summary="GitHub webhook receiver",
    description=(
        "Receives GitHub webhooks and persists normalized git events into Postgres.\n\n"
        "Security: verifies X-Hub-Signature-256 using GITHUB_WEBHOOK_SECRET.\n"
        "Idempotency: uses X-GitHub-Delivery as external_event_id (and per-commit suffixes)."
    ),
    operation_id="webhook_github_post",
)
async def github_webhook(request: Request, db: Session = Depends(get_db_session)) -> Dict[str, Any]:
    """Handle GitHub webhooks (push, pull_request, etc.)."""
    body = await request.body()
    _verify_github_signature(body, request.headers.get("X-Hub-Signature-256"))

    delivery_id = request.headers.get("X-GitHub-Delivery")
    event_name = request.headers.get("X-GitHub-Event") or "unknown"

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    external_id, owner, name, full_name, default_branch, is_private = _github_repo_info(payload)
    repo = _get_or_create_repo(
        db,
        provider="github",
        external_id=external_id,
        owner=owner,
        name=name,
        full_name=full_name,
        default_branch=default_branch,
        is_private=is_private,
    )

    events = _github_events(repo=repo, payload=payload, event_name=str(event_name), delivery_id=delivery_id)

    try:
        stats = _bulk_insert_events(db, events)
        logger.info("GitHub webhook ingested: event=%s delivery=%s repo=%s stats=%s", event_name, delivery_id, full_name, stats)
    except Exception:
        logger.exception("GitHub webhook ingestion failed (returning 202 to avoid provider retries storm).")
        return {"status": "accepted", "provider": "github"}

    return {"status": "ok", "provider": "github", "event": event_name, "delivery_id": delivery_id, **stats}


# PUBLIC_INTERFACE
@router.post(
    "/gitlab",
    summary="GitLab webhook receiver",
    description=(
        "Receives GitLab webhooks and persists normalized git events into Postgres.\n\n"
        "Security: verifies X-Gitlab-Token matches GITLAB_WEBHOOK_TOKEN.\n"
        "Idempotency: uses X-Gitlab-Event-UUID when present."
    ),
    operation_id="webhook_gitlab_post",
)
async def gitlab_webhook(request: Request, db: Session = Depends(get_db_session)) -> Dict[str, Any]:
    """Handle GitLab webhooks (Push Hook, Merge Request Hook, etc.)."""
    _verify_gitlab_token(request.headers.get("X-Gitlab-Token"))

    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    event_name = request.headers.get("X-Gitlab-Event") or "unknown"
    delivery_id = request.headers.get("X-Gitlab-Event-UUID") or request.headers.get("X-Request-Id")

    external_id, owner, name, full_name, default_branch, is_private = _gitlab_repo_info(payload)
    repo = _get_or_create_repo(
        db,
        provider="gitlab",
        external_id=external_id,
        owner=owner,
        name=name,
        full_name=full_name,
        default_branch=default_branch,
        is_private=is_private,
    )

    events = _gitlab_events(repo=repo, payload=payload, event_name=str(event_name), delivery_id=delivery_id)

    try:
        stats = _bulk_insert_events(db, events)
        logger.info("GitLab webhook ingested: event=%s delivery=%s repo=%s stats=%s", event_name, delivery_id, full_name, stats)
    except Exception:
        logger.exception("GitLab webhook ingestion failed (returning 202 to avoid provider retries storm).")
        return {"status": "accepted", "provider": "gitlab"}

    return {"status": "ok", "provider": "gitlab", "event": event_name, "delivery_id": delivery_id, **stats}


# PUBLIC_INTERFACE
@router.post(
    "/bitbucket",
    summary="Bitbucket webhook receiver",
    description=(
        "Receives Bitbucket Cloud webhooks and persists normalized git events into Postgres.\n\n"
        "Security: if BITBUCKET_WEBHOOK_SECRET is configured, verifies X-Hub-Signature (HMAC SHA-256).\n"
        "Idempotency: uses X-Request-UUID when present."
    ),
    operation_id="webhook_bitbucket_post",
)
async def bitbucket_webhook(request: Request, db: Session = Depends(get_db_session)) -> Dict[str, Any]:
    """Handle Bitbucket webhooks (repo:push, pullrequest:created, pullrequest:fulfilled, etc.)."""
    body = await request.body()
    _verify_bitbucket_signature_if_configured(body, request.headers.get("X-Hub-Signature"))

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    event_key = request.headers.get("X-Event-Key") or "unknown"
    delivery_id = request.headers.get("X-Request-UUID") or request.headers.get("X-Request-Id")

    external_id, owner, name, full_name, default_branch, is_private = _bitbucket_repo_info(payload)
    repo = _get_or_create_repo(
        db,
        provider="bitbucket",
        external_id=external_id,
        owner=owner,
        name=name,
        full_name=full_name,
        default_branch=default_branch,
        is_private=is_private,
    )

    events = _bitbucket_events(repo=repo, payload=payload, event_key=str(event_key), delivery_id=delivery_id)

    try:
        stats = _bulk_insert_events(db, events)
        logger.info("Bitbucket webhook ingested: event=%s delivery=%s repo=%s stats=%s", event_key, delivery_id, full_name, stats)
    except Exception:
        logger.exception("Bitbucket webhook ingestion failed (returning 202 to avoid provider retries storm).")
        return {"status": "accepted", "provider": "bitbucket"}

    return {"status": "ok", "provider": "bitbucket", "event": event_key, "delivery_id": delivery_id, **stats}
