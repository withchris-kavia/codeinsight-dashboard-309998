"""
Analytics routes and on-demand aggregation for the CodeInsight Dashboard.

This module provides:
- On-demand aggregation that upserts into analytics tables:
  - analytics_repo_daily
  - analytics_dev_daily
- Dashboard-friendly read endpoints:
  - commits per day
  - PRs merged per repo
  - active contributors
  - recent activity feed

Background scheduling is intentionally not assumed; aggregation is triggered via
a compute endpoint.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.db import get_db_session

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _ensure_default_org_and_get_id(db: Session) -> UUID:
    """
    Ensure the 'default' org exists and return its UUID.

    This supports smoke tests where the caller uses org_id=1 as a shorthand.
    """
    row = db.execute(text("SELECT id FROM orgs WHERE slug = 'default' LIMIT 1")).first()
    if row and row[0]:
        return UUID(str(row[0]))

    created = db.execute(
        text(
            """
            INSERT INTO orgs (slug, name, plan_tier)
            VALUES ('default', 'Default', 'free')
            ON CONFLICT (slug) DO UPDATE SET updated_at = now()
            RETURNING id
            """
        )
    ).first()
    if not created or not created[0]:
        # Extremely defensive; should not happen.
        raise HTTPException(status_code=500, detail="Failed to create default org.")
    db.commit()
    return UUID(str(created[0]))


def _parse_org_id(value: str, db: Session) -> UUID:
    """
    Parse org identifier.

    Accepted formats:
      - UUID string
      - "1" (legacy/smoke test shorthand) => maps to default org UUID
      - "default" => maps to default org UUID
    """
    v = (value or "").strip()
    if v in ("1", "default"):
        return _ensure_default_org_and_get_id(db)
    try:
        return UUID(v)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"org_id must be a UUID (or '1'/'default'): {e}") from e


def _utc_dt_range_from_dates(
    start_date: Optional[date],
    end_date: Optional[date],
) -> tuple[datetime, datetime]:
    """
    Convert date boundaries into an [start, end) UTC datetime range.

    If only start_date is provided: end defaults to tomorrow (UTC).
    If only end_date is provided: start defaults to (end_date - 30 days).
    If neither is provided: last 30 days through tomorrow (UTC).
    """
    today_utc = datetime.now(timezone.utc).date()

    if start_date is None and end_date is None:
        start_date = today_utc - timedelta(days=30)
        end_date = today_utc + timedelta(days=1)
    elif start_date is None and end_date is not None:
        start_date = end_date - timedelta(days=30)
    elif start_date is not None and end_date is None:
        end_date = today_utc + timedelta(days=1)

    assert start_date is not None and end_date is not None
    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, time.min, tzinfo=timezone.utc)

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_date must be after start_date.")

    return start_dt, end_dt


class AnalyticsComputeRequest(BaseModel):
    org_id: str = Field(..., description="Organization UUID (or legacy '1' for default org) to compute analytics for.")
    repo_id: Optional[UUID] = Field(default=None, description="Optional repo UUID to scope aggregation to.")
    start_date: Optional[date] = Field(
        default=None,
        description="Optional start date (inclusive, UTC). Defaults to last 30 days if not provided.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Optional end date (exclusive, UTC). Defaults to tomorrow if not provided.",
    )


class AnalyticsComputeResponse(BaseModel):
    org_id: UUID = Field(..., description="Organization UUID analytics were computed for.")
    repo_id: Optional[UUID] = Field(default=None, description="Repo UUID if aggregation was scoped to a repo.")
    start_ts: datetime = Field(..., description="UTC start timestamp used for aggregation (inclusive).")
    end_ts: datetime = Field(..., description="UTC end timestamp used for aggregation (exclusive).")
    repo_daily_upserts: int = Field(..., description="Number of repo_daily rows upserted (inserted or updated).")
    dev_daily_upserts: int = Field(..., description="Number of dev_daily rows upserted (inserted or updated).")


class DayCountPoint(BaseModel):
    day: date = Field(..., description="Day (UTC date).")
    count: int = Field(..., description="Count for that day.")


class PullRequestsDayPoint(BaseModel):
    day: date = Field(..., description="Day (UTC date).")
    opened: int = Field(..., description="PRs opened (count of pull_request events) for that day.")
    merged: int = Field(..., description="PRs merged (count of merge events) for that day.")


class DashboardMetricsResponse(BaseModel):
    org_id: UUID = Field(..., description="Organization UUID.")
    repo_id: Optional[UUID] = Field(default=None, description="Repo UUID if scoped.")
    start_ts: datetime = Field(..., description="UTC start timestamp used (inclusive).")
    end_ts: datetime = Field(..., description="UTC end timestamp used (exclusive).")

    commits_per_day: List[DayCountPoint] = Field(..., description="Commits time-series for charts.")
    prs_per_day: List[PullRequestsDayPoint] = Field(..., description="PRs opened/merged time-series for charts.")
    prs_merged_per_repo: List[RepoCountPoint] = Field(..., description="Top repos by merged PRs for charts/tables.")

    active_contributors: int = Field(..., description="Distinct active contributors in the time range.")
    recent_activity: List[RecentActivityItem] = Field(..., description="Recent activity feed items.")


class RepoCountPoint(BaseModel):
    repo_id: UUID = Field(..., description="Repo UUID.")
    repo_full_name: str = Field(..., description="Repo full name (owner/name).")
    count: int = Field(..., description="Aggregate count.")


class RecentActivityItem(BaseModel):
    git_event_id: UUID = Field(..., description="Git event UUID.")
    repo_id: UUID = Field(..., description="Repo UUID.")
    repo_full_name: str = Field(..., description="Repo full name (owner/name).")
    provider: str = Field(..., description="Provider (github|gitlab|bitbucket).")
    event_type: str = Field(..., description="Event type (commit|pull_request|merge|unknown).")
    actor_username: Optional[str] = Field(default=None, description="Actor username if present.")
    actor_email: Optional[str] = Field(default=None, description="Actor email if present.")
    event_timestamp: datetime = Field(..., description="Event timestamp.")
    pr_number: Optional[int] = Field(default=None, description="PR number if present.")
    pr_state: Optional[str] = Field(default=None, description="PR state if present.")
    commit_sha: Optional[str] = Field(default=None, description="Commit SHA if present.")
    merge_commit_sha: Optional[str] = Field(default=None, description="Merge commit SHA if present.")


class ActiveContributorsResponse(BaseModel):
    org_id: UUID = Field(..., description="Organization UUID.")
    repo_id: Optional[UUID] = Field(default=None, description="Repo UUID if scoped.")
    start_ts: datetime = Field(..., description="UTC start timestamp used (inclusive).")
    end_ts: datetime = Field(..., description="UTC end timestamp used (exclusive).")
    active_contributors: int = Field(..., description="Distinct active contributors in the time range.")


def _validate_org_repo_scope(org_id: UUID, repo_id: Optional[UUID], db: Session) -> None:
    """Validate that repo_id (if provided) belongs to org_id."""
    if repo_id is None:
        return
    row = db.execute(
        text("SELECT 1 FROM repos WHERE id = :repo_id AND org_id = :org_id"),
        {"repo_id": str(repo_id), "org_id": str(org_id)},
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Repo not found in org.")


def _compute_repo_daily(db: Session, *, org_id: UUID, repo_id: Optional[UUID], start_ts: datetime, end_ts: datetime) -> int:
    """
    Upsert analytics_repo_daily rows from git_events for the given scope/range.

    Note: PR merged counts are approximated using `merge` events (since ingestion normalizes
    merges into event_type='merge').

    Important:
      - SQLAlchemy/psycopg `rowcount` is not reliable for INSERT .. ON CONFLICT across drivers.
      - We use `RETURNING 1` and count returned rows for a stable, deterministic upsert count.
    """
    sql = text(
        """
        WITH filtered AS (
          SELECT
            ge.org_id,
            ge.repo_id,
            (ge.event_timestamp AT TIME ZONE 'UTC')::date AS day,
            ge.event_type,
            ge.actor_provider_user_id,
            ge.actor_username,
            ge.actor_email
          FROM git_events ge
          WHERE ge.org_id = :org_id
            AND (:repo_id::uuid IS NULL OR ge.repo_id = :repo_id::uuid)
            AND ge.event_timestamp >= :start_ts
            AND ge.event_timestamp < :end_ts
        ),
        daily AS (
          SELECT
            org_id,
            repo_id,
            day,
            SUM(CASE WHEN event_type = 'commit' THEN 1 ELSE 0 END)::int AS commits_count,
            SUM(CASE WHEN event_type = 'pull_request' THEN 1 ELSE 0 END)::int AS prs_opened_count,
            -- Approximation: merges represent "PRs merged" in our normalized event stream
            SUM(CASE WHEN event_type = 'merge' THEN 1 ELSE 0 END)::int AS prs_merged_count,
            SUM(CASE WHEN event_type = 'merge' THEN 1 ELSE 0 END)::int AS merges_count,
            COUNT(
              DISTINCT COALESCE(
                NULLIF(actor_provider_user_id, ''),
                NULLIF(actor_username, ''),
                NULLIF(actor_email, '')
              )
            ) FILTER (
              WHERE COALESCE(
                NULLIF(actor_provider_user_id, ''),
                NULLIF(actor_username, ''),
                NULLIF(actor_email, '')
              ) IS NOT NULL
            )::int AS active_devs_count
          FROM filtered
          GROUP BY org_id, repo_id, day
        )
        INSERT INTO analytics_repo_daily (
          org_id,
          repo_id,
          day,
          commits_count,
          prs_opened_count,
          prs_merged_count,
          merges_count,
          active_devs_count,
          updated_at
        )
        SELECT
          org_id,
          repo_id,
          day,
          commits_count,
          prs_opened_count,
          prs_merged_count,
          merges_count,
          active_devs_count,
          now()
        FROM daily
        ON CONFLICT (repo_id, day)
        DO UPDATE SET
          commits_count = EXCLUDED.commits_count,
          prs_opened_count = EXCLUDED.prs_opened_count,
          prs_merged_count = EXCLUDED.prs_merged_count,
          merges_count = EXCLUDED.merges_count,
          active_devs_count = EXCLUDED.active_devs_count,
          updated_at = now()
        RETURNING 1
        """
    )
    rows = db.execute(
        sql,
        {
            "org_id": str(org_id),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).fetchall()
    return int(len(rows))


def _compute_dev_daily(db: Session, *, org_id: UUID, repo_id: Optional[UUID], start_ts: datetime, end_ts: datetime) -> int:
    """
    Upsert analytics_dev_daily rows from git_events for the given scope/range.

    We only aggregate rows where at least one actor identifier is present, otherwise
    uniqueness/attribution becomes unreliable.

    Important:
      - SQLAlchemy/psycopg `rowcount` is not reliable for INSERT .. ON CONFLICT across drivers.
      - We use `RETURNING 1` and count returned rows for a stable, deterministic upsert count.
    """
    sql = text(
        """
        WITH filtered AS (
          SELECT
            ge.org_id,
            ge.repo_id,
            (ge.event_timestamp AT TIME ZONE 'UTC')::date AS day,
            ge.event_type,
            ge.actor_provider_user_id,
            ge.actor_username,
            ge.actor_email
          FROM git_events ge
          WHERE ge.org_id = :org_id
            AND (:repo_id::uuid IS NULL OR ge.repo_id = :repo_id::uuid)
            AND ge.event_timestamp >= :start_ts
            AND ge.event_timestamp < :end_ts
            AND (
              NULLIF(ge.actor_provider_user_id, '') IS NOT NULL OR
              NULLIF(ge.actor_username, '') IS NOT NULL OR
              NULLIF(ge.actor_email, '') IS NOT NULL
            )
        ),
        daily AS (
          SELECT
            org_id,
            repo_id,
            day,
            actor_provider_user_id,
            actor_username,
            actor_email,
            SUM(CASE WHEN event_type = 'commit' THEN 1 ELSE 0 END)::int AS commits_count,
            SUM(CASE WHEN event_type = 'pull_request' THEN 1 ELSE 0 END)::int AS prs_opened_count,
            SUM(CASE WHEN event_type = 'merge' THEN 1 ELSE 0 END)::int AS prs_merged_count,
            SUM(CASE WHEN event_type = 'merge' THEN 1 ELSE 0 END)::int AS merges_count
          FROM filtered
          GROUP BY org_id, repo_id, day, actor_provider_user_id, actor_username, actor_email
        )
        INSERT INTO analytics_dev_daily (
          org_id,
          repo_id,
          day,
          actor_provider_user_id,
          actor_username,
          actor_email,
          commits_count,
          prs_opened_count,
          prs_merged_count,
          merges_count,
          lines_added,
          lines_deleted,
          updated_at
        )
        SELECT
          org_id,
          repo_id,
          day,
          actor_provider_user_id,
          actor_username,
          actor_email,
          commits_count,
          prs_opened_count,
          prs_merged_count,
          merges_count,
          0,
          0,
          now()
        FROM daily
        ON CONFLICT ON CONSTRAINT analytics_dev_daily_unique
        DO UPDATE SET
          commits_count = EXCLUDED.commits_count,
          prs_opened_count = EXCLUDED.prs_opened_count,
          prs_merged_count = EXCLUDED.prs_merged_count,
          merges_count = EXCLUDED.merges_count,
          updated_at = now()
        RETURNING 1
        """
    )
    rows = db.execute(
        sql,
        {
            "org_id": str(org_id),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).fetchall()
    return int(len(rows))


# PUBLIC_INTERFACE
@router.post(
    "/compute",
    summary="Compute analytics aggregates (on-demand)",
    description=(
        "Computes analytics aggregates from `git_events` and upserts results into:\n"
        "- analytics_repo_daily\n"
        "- analytics_dev_daily\n\n"
        "This endpoint is intended to be called by the frontend before rendering dashboards, "
        "or by a future scheduler/cron.\n\n"
        "Time range is specified by start_date/end_date (UTC dates). end_date is treated as exclusive."
    ),
    operation_id="analytics_compute_post",
    response_model=AnalyticsComputeResponse,
)
def compute_analytics(req: AnalyticsComputeRequest, db: Session = Depends(get_db_session)) -> AnalyticsComputeResponse:
    """Compute daily analytics aggregates and store them in analytics tables."""
    org_uuid = _parse_org_id(req.org_id, db)
    _validate_org_repo_scope(org_uuid, req.repo_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(req.start_date, req.end_date)

    try:
        repo_upserts = _compute_repo_daily(db, org_id=org_uuid, repo_id=req.repo_id, start_ts=start_ts, end_ts=end_ts)
        dev_upserts = _compute_dev_daily(db, org_id=org_uuid, repo_id=req.repo_id, start_ts=start_ts, end_ts=end_ts)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail={"error": "analytics_compute_failed", "message": str(e)})

    return AnalyticsComputeResponse(
        org_id=org_uuid,
        repo_id=req.repo_id,
        start_ts=start_ts,
        end_ts=end_ts,
        repo_daily_upserts=repo_upserts,
        dev_daily_upserts=dev_upserts,
    )


# PUBLIC_INTERFACE
@router.get(
    "/commits-per-day",
    summary="Commits per day",
    description=(
        "Returns daily commits counts based on analytics_repo_daily.\n\n"
        "If repo_id is omitted, sums commits across all repos in the org."
    ),
    operation_id="analytics_commits_per_day_get",
    response_model=List[DayCountPoint],
)
def commits_per_day(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID."),
    start_date: Optional[date] = Query(default=None, description="Start date (inclusive, UTC)."),
    end_date: Optional[date] = Query(default=None, description="End date (exclusive, UTC)."),
    db: Session = Depends(get_db_session),
) -> List[DayCountPoint]:
    """Fetch commits per day for org/repo using precomputed analytics."""
    org_uuid = _parse_org_id(org_id, db)
    _validate_org_repo_scope(org_uuid, repo_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(start_date, end_date)

    sql = text(
        """
        SELECT
          ard.day AS day,
          SUM(ard.commits_count)::int AS count
        FROM analytics_repo_daily ard
        WHERE ard.org_id = :org_id
          AND (:repo_id::uuid IS NULL OR ard.repo_id = :repo_id::uuid)
          AND ard.day >= (:start_ts AT TIME ZONE 'UTC')::date
          AND ard.day <  (:end_ts AT TIME ZONE 'UTC')::date
        GROUP BY ard.day
        ORDER BY ard.day ASC
        """
    )
    rows = db.execute(
        sql,
        {
            "org_id": str(org_uuid),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).mappings().all()

    return [DayCountPoint(day=r["day"], count=int(r["count"])) for r in rows]


# PUBLIC_INTERFACE
@router.get(
    "/prs-per-day",
    summary="PRs opened and merged per day",
    description=(
        "Returns daily PR opened and PR merged counts based on analytics_repo_daily.\\n\\n"
        "If repo_id is omitted, sums across all repos in the org.\\n\\n"
        "Note: merged counts are derived from normalized `merge` events."
    ),
    operation_id="analytics_prs_per_day_get",
    response_model=List[PullRequestsDayPoint],
)
def prs_per_day(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID."),
    start_date: Optional[date] = Query(default=None, description="Start date (inclusive, UTC)."),
    end_date: Optional[date] = Query(default=None, description="End date (exclusive, UTC)."),
    db: Session = Depends(get_db_session),
) -> List[PullRequestsDayPoint]:
    """Fetch PR opened/merged per day for org/repo using precomputed analytics."""
    org_uuid = _parse_org_id(org_id, db)
    _validate_org_repo_scope(org_uuid, repo_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(start_date, end_date)

    sql = text(
        """
        SELECT
          ard.day AS day,
          SUM(ard.prs_opened_count)::int AS opened,
          SUM(ard.prs_merged_count)::int AS merged
        FROM analytics_repo_daily ard
        WHERE ard.org_id = :org_id
          AND (:repo_id::uuid IS NULL OR ard.repo_id = :repo_id::uuid)
          AND ard.day >= (:start_ts AT TIME ZONE 'UTC')::date
          AND ard.day <  (:end_ts AT TIME ZONE 'UTC')::date
        GROUP BY ard.day
        ORDER BY ard.day ASC
        """
    )

    rows = db.execute(
        sql,
        {
            "org_id": str(org_id),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).mappings().all()

    return [
        PullRequestsDayPoint(
            day=r["day"],
            opened=int(r["opened"] or 0),
            merged=int(r["merged"] or 0),
        )
        for r in rows
    ]


# PUBLIC_INTERFACE
@router.get(
    "/dashboard",
    summary="Dashboard metrics (single call)",
    description=(
        "Returns a compact, dashboard-ready bundle of analytics metrics in a single request.\\n\\n"
        "This is designed for the frontend dashboard view to minimize round-trips.\\n\\n"
        "Data sources:\\n"
        "- commits_per_day: analytics_repo_daily\\n"
        "- prs_per_day: analytics_repo_daily\\n"
        "- prs_merged_per_repo: analytics_repo_daily joined with repos\\n"
        "- active_contributors: analytics_dev_daily (with git_events fallback)\\n"
        "- recent_activity: git_events joined with repos\\n\\n"
        "Tip: call POST /analytics/compute first (same date window) to ensure aggregates are up-to-date."
    ),
    operation_id="analytics_dashboard_get",
    response_model=DashboardMetricsResponse,
)
def dashboard_metrics(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID."),
    start_date: Optional[date] = Query(default=None, description="Start date (inclusive, UTC)."),
    end_date: Optional[date] = Query(default=None, description="End date (exclusive, UTC)."),
    recent_limit: int = Query(default=50, ge=1, le=200, description="Max recent activity items."),
    top_repos_limit: int = Query(default=10, ge=1, le=200, description="Max repos in merged-PR ranking."),
    db: Session = Depends(get_db_session),
) -> DashboardMetricsResponse:
    """Fetch a dashboard-ready set of analytics metrics."""
    org_uuid = _parse_org_id(org_id, db)
    _validate_org_repo_scope(org_uuid, repo_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(start_date, end_date)

    commits_series = commits_per_day(
        org_id=str(org_uuid),
        repo_id=repo_id,
        start_date=start_date,
        end_date=end_date,
        db=db,
    )

    prs_series = prs_per_day(
        org_id=str(org_uuid),
        repo_id=repo_id,
        start_date=start_date,
        end_date=end_date,
        db=db,
    )

    # If dashboard is scoped to a repo, still return "merged per repo" as just that repo for consistency.
    if repo_id is not None:
        sql = text(
            """
            SELECT
              r.id AS repo_id,
              r.full_name AS repo_full_name,
              COALESCE(SUM(ard.prs_merged_count), 0)::int AS count
            FROM repos r
            LEFT JOIN analytics_repo_daily ard
              ON ard.repo_id = r.id
              AND ard.day >= (:start_ts AT TIME ZONE 'UTC')::date
              AND ard.day <  (:end_ts AT TIME ZONE 'UTC')::date
            WHERE r.org_id = :org_id
              AND r.id = :repo_id::uuid
            GROUP BY r.id, r.full_name
            ORDER BY count DESC, r.full_name ASC
            """
        )
        rows = db.execute(
            sql,
            {"org_id": str(org_uuid), "repo_id": str(repo_id), "start_ts": start_ts, "end_ts": end_ts},
        ).mappings().all()
        merged_per_repo = [
            RepoCountPoint(repo_id=r["repo_id"], repo_full_name=r["repo_full_name"], count=int(r["count"])) for r in rows
        ]
    else:
        merged_per_repo = prs_merged_per_repo(
            org_id=str(org_uuid),
            start_date=start_date,
            end_date=end_date,
            limit=int(top_repos_limit),
            db=db,
        )

    active = active_contributors(
        org_id=str(org_uuid),
        repo_id=repo_id,
        start_date=start_date,
        end_date=end_date,
        db=db,
    )

    recent = recent_activity(org_id=str(org_uuid), repo_id=repo_id, limit=int(recent_limit), db=db)

    return DashboardMetricsResponse(
        org_id=org_uuid,
        repo_id=repo_id,
        start_ts=start_ts,
        end_ts=end_ts,
        commits_per_day=commits_series,
        prs_per_day=prs_series,
        prs_merged_per_repo=merged_per_repo,
        active_contributors=active.active_contributors,
        recent_activity=recent,
    )


# PUBLIC_INTERFACE
@router.get(
    "/prs-merged-per-repo",
    summary="PRs merged per repo",
    description=(
        "Returns merged PR counts per repo, derived from analytics_repo_daily.prs_merged_count.\n\n"
        "Note: In this system, prs_merged_count is computed from normalized `merge` events."
    ),
    operation_id="analytics_prs_merged_per_repo_get",
    response_model=List[RepoCountPoint],
)
def prs_merged_per_repo(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    start_date: Optional[date] = Query(default=None, description="Start date (inclusive, UTC)."),
    end_date: Optional[date] = Query(default=None, description="End date (exclusive, UTC)."),
    limit: int = Query(default=50, ge=1, le=200, description="Max repos returned (sorted by merged count desc)."),
    db: Session = Depends(get_db_session),
) -> List[RepoCountPoint]:
    """Fetch merged PR counts per repo for an org."""
    org_uuid = _parse_org_id(org_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(start_date, end_date)

    sql = text(
        """
        SELECT
          r.id AS repo_id,
          r.full_name AS repo_full_name,
          COALESCE(SUM(ard.prs_merged_count), 0)::int AS count
        FROM repos r
        LEFT JOIN analytics_repo_daily ard
          ON ard.repo_id = r.id
          AND ard.day >= (:start_ts AT TIME ZONE 'UTC')::date
          AND ard.day <  (:end_ts AT TIME ZONE 'UTC')::date
        WHERE r.org_id = :org_id
        GROUP BY r.id, r.full_name
        ORDER BY count DESC, r.full_name ASC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {"org_id": str(org_uuid), "start_ts": start_ts, "end_ts": end_ts, "limit": int(limit)},
    ).mappings().all()

    return [RepoCountPoint(repo_id=r["repo_id"], repo_full_name=r["repo_full_name"], count=int(r["count"])) for r in rows]


# PUBLIC_INTERFACE
@router.get(
    "/active-contributors",
    summary="Active contributors",
    description=(
        "Returns distinct active contributor count for the range.\n\n"
        "Uses analytics_dev_daily if available, falling back to git_events distinct actor ids if needed."
    ),
    operation_id="analytics_active_contributors_get",
    response_model=ActiveContributorsResponse,
)
def active_contributors(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID."),
    start_date: Optional[date] = Query(default=None, description="Start date (inclusive, UTC)."),
    end_date: Optional[date] = Query(default=None, description="End date (exclusive, UTC)."),
    db: Session = Depends(get_db_session),
) -> ActiveContributorsResponse:
    """Compute distinct active contributor count."""
    org_uuid = _parse_org_id(org_id, db)
    _validate_org_repo_scope(org_uuid, repo_id, db)
    start_ts, end_ts = _utc_dt_range_from_dates(start_date, end_date)

    # Preferred: analytics_dev_daily (smaller than git_events).
    sql = text(
        """
        SELECT COUNT(DISTINCT COALESCE(
          NULLIF(add.actor_provider_user_id, ''),
          NULLIF(add.actor_username, ''),
          NULLIF(add.actor_email, '')
        ))::int AS cnt
        FROM analytics_dev_daily add
        WHERE add.org_id = :org_id
          AND (:repo_id::uuid IS NULL OR add.repo_id = :repo_id::uuid)
          AND add.day >= (:start_ts AT TIME ZONE 'UTC')::date
          AND add.day <  (:end_ts AT TIME ZONE 'UTC')::date
        """
    )
    row = db.execute(
        sql,
        {
            "org_id": str(org_uuid),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "start_ts": start_ts,
            "end_ts": end_ts,
        },
    ).mappings().first()

    cnt = int(row["cnt"] if row and row.get("cnt") is not None else 0)

    # Fallback for cases where analytics tables are empty (e.g., compute endpoint not called yet).
    if cnt == 0:
        fallback_sql = text(
            """
            SELECT COUNT(DISTINCT COALESCE(
              NULLIF(ge.actor_provider_user_id, ''),
              NULLIF(ge.actor_username, ''),
              NULLIF(ge.actor_email, '')
            ))::int AS cnt
            FROM git_events ge
            WHERE ge.org_id = :org_id
              AND (:repo_id::uuid IS NULL OR ge.repo_id = :repo_id::uuid)
              AND ge.event_timestamp >= :start_ts
              AND ge.event_timestamp < :end_ts
              AND COALESCE(
                NULLIF(ge.actor_provider_user_id, ''),
                NULLIF(ge.actor_username, ''),
                NULLIF(ge.actor_email, '')
              ) IS NOT NULL
            """
        )
        fb = db.execute(
            fallback_sql,
            {
                "org_id": str(org_uuid),
                "repo_id": (str(repo_id) if repo_id is not None else None),
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        ).mappings().first()
        cnt = int(fb["cnt"] if fb and fb.get("cnt") is not None else 0)

    return ActiveContributorsResponse(
        org_id=org_uuid,
        repo_id=repo_id,
        start_ts=start_ts,
        end_ts=end_ts,
        active_contributors=cnt,
    )


# PUBLIC_INTERFACE
@router.get(
    "/recent-activity",
    summary="Recent activity feed",
    description=(
        "Returns recent git_events joined with repo metadata for an org.\n\n"
        "This is intended for dashboard activity feeds."
    ),
    operation_id="analytics_recent_activity_get",
    response_model=List[RecentActivityItem],
)
def recent_activity(
    org_id: str = Query(..., description="Organization UUID (or legacy '1'/'default' for default org)."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID."),
    limit: int = Query(default=50, ge=1, le=200, description="Max events returned."),
    db: Session = Depends(get_db_session),
) -> List[RecentActivityItem]:
    """Fetch latest git events for activity feed."""
    org_uuid = _parse_org_id(org_id, db)
    _validate_org_repo_scope(org_uuid, repo_id, db)

    sql = text(
        """
        SELECT
          ge.id AS git_event_id,
          ge.repo_id,
          r.full_name AS repo_full_name,
          ge.provider,
          ge.event_type,
          ge.actor_username,
          ge.actor_email,
          ge.event_timestamp,
          ge.pr_number,
          ge.pr_state,
          ge.commit_sha,
          ge.merge_commit_sha
        FROM git_events ge
        JOIN repos r ON r.id = ge.repo_id
        WHERE ge.org_id = :org_id
          AND (:repo_id::uuid IS NULL OR ge.repo_id = :repo_id::uuid)
        ORDER BY ge.event_timestamp DESC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {"org_id": str(org_uuid), "repo_id": (str(repo_id) if repo_id is not None else None), "limit": int(limit)},
    ).mappings().all()

    return [
        RecentActivityItem(
            git_event_id=r["git_event_id"],
            repo_id=r["repo_id"],
            repo_full_name=r["repo_full_name"],
            provider=r["provider"],
            event_type=r["event_type"],
            actor_username=r.get("actor_username"),
            actor_email=r.get("actor_email"),
            event_timestamp=r["event_timestamp"],
            pr_number=r.get("pr_number"),
            pr_state=r.get("pr_state"),
            commit_sha=r.get("commit_sha"),
            merge_commit_sha=r.get("merge_commit_sha"),
        )
        for r in rows
    ]
