"""
AI summary endpoints for the CodeInsight Dashboard.

This router provides:
- POST /ai/summaries/compute: compute (or simulate) a summary and persist it.
- GET  /ai/summaries/recent: list recent summaries for dashboard display.
- GET  /ai/summaries/{id}: fetch a single persisted summary.

External AI calls are gated behind provider keys in environment variables.
If keys are missing, the endpoint returns a simulated summary (and persists it)
so the app remains functional in preview/dev environments.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.db import get_db_session
from src.api.models import AiSummary

router = APIRouter(prefix="/ai/summaries", tags=["AI"])


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _require_period(period_start: date, period_end: date) -> None:
    """Validate [start, end) period range for date inputs."""
    if period_end <= period_start:
        raise HTTPException(status_code=400, detail="period_end must be after period_start.")


def _openai_api_key() -> Optional[str]:
    """Read OpenAI API key from environment; returns None if not configured."""
    return os.getenv("OPENAI_API_KEY") or None


def _default_openai_model() -> str:
    """Read default OpenAI model from env; uses a safe default."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _ensure_ai_summaries_table(db: Session) -> None:
    """Ensure `ai_summaries` table exists.

    This keeps previews/dev working even if the DB schema migration has not been applied yet.
    """
    # Lightweight DDL (no heavy joins) and safe if the schema already exists.
    ddl = text(
        """
        CREATE TABLE IF NOT EXISTS ai_summaries (
          id uuid PRIMARY KEY,
          repo_id uuid NULL,
          org_id uuid NULL,
          subject_type text NOT NULL,
          subject_id uuid NULL,
          period_start date NOT NULL,
          period_end date NOT NULL,
          summary_text text NOT NULL,
          provider text NOT NULL,
          model text NULL,
          tokens_in int NULL,
          tokens_out int NULL,
          cost_usd numeric(12,6) NULL,
          created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ai_summaries_created_at_idx ON ai_summaries(created_at DESC);
        CREATE INDEX IF NOT EXISTS ai_summaries_org_id_idx ON ai_summaries(org_id);
        CREATE INDEX IF NOT EXISTS ai_summaries_repo_id_idx ON ai_summaries(repo_id);
        CREATE INDEX IF NOT EXISTS ai_summaries_subject_idx ON ai_summaries(subject_type, subject_id);
        """
    )
    db.execute(ddl)
    db.commit()


Scope = Literal["repo", "org", "user", "system"]


class AiSummaryComputeRequest(BaseModel):
    scope: Scope = Field(..., description="Summary scope: repo|org|user|system.")
    scope_id: Optional[UUID] = Field(
        default=None,
        description="UUID of the scoped entity (repo/org/user). Not used for scope=system.",
    )
    period_start: date = Field(..., description="Start date (inclusive, UTC).")
    period_end: date = Field(..., description="End date (exclusive, UTC).")

    provider: Optional[str] = Field(
        default=None,
        description="Optional AI provider override. Currently supported: openai. Defaults to openai.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override. Defaults to OPENAI_MODEL or a safe default.",
    )


class AiSummaryOut(BaseModel):
    id: UUID = Field(..., description="Summary UUID.")
    repo_id: Optional[UUID] = Field(default=None, description="Repo UUID (nullable).")
    org_id: Optional[UUID] = Field(default=None, description="Org UUID (nullable).")

    subject_type: str = Field(..., description="repo|org|user|system.")
    subject_id: Optional[UUID] = Field(default=None, description="Scoped entity UUID (nullable).")

    period_start: date = Field(..., description="Start date (inclusive, UTC).")
    period_end: date = Field(..., description="End date (exclusive, UTC).")

    summary_text: str = Field(..., description="Generated summary text.")
    provider: str = Field(..., description="AI provider used.")
    model: Optional[str] = Field(default=None, description="Model used (if any).")

    tokens_in: Optional[int] = Field(default=None, description="Input tokens (if available).")
    tokens_out: Optional[int] = Field(default=None, description="Output tokens (if available).")
    cost_usd: Optional[float] = Field(default=None, description="Estimated USD cost (if available).")

    created_at: datetime = Field(..., description="Creation timestamp (UTC).")


def _summary_to_out(s: AiSummary) -> AiSummaryOut:
    """Convert ORM model to response model."""
    # cost_usd may be Decimal depending on driver; cast to float when present.
    cost_val = float(s.cost_usd) if s.cost_usd is not None else None
    return AiSummaryOut(
        id=s.id,
        repo_id=s.repo_id,
        org_id=s.org_id,
        subject_type=s.subject_type,
        subject_id=s.subject_id,
        period_start=s.period_start,
        period_end=s.period_end,
        summary_text=s.summary_text,
        provider=s.provider,
        model=s.model,
        tokens_in=s.tokens_in,
        tokens_out=s.tokens_out,
        cost_usd=cost_val,
        created_at=s.created_at,
    )


def _resolve_scope_refs(db: Session, scope: Scope, scope_id: Optional[UUID]) -> Dict[str, Optional[UUID]]:
    """Resolve org_id/repo_id/subject_type/subject_id based on request scope."""
    if scope == "system":
        return {"org_id": None, "repo_id": None, "subject_type": "system", "subject_id": None}

    if scope_id is None:
        raise HTTPException(status_code=422, detail="scope_id is required for scope=repo|org|user.")

    if scope == "repo":
        row = db.execute(
            text("SELECT id AS repo_id, org_id FROM repos WHERE id = :repo_id"),
            {"repo_id": str(scope_id)},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Repo not found.")
        return {
            "org_id": UUID(row["org_id"]),
            "repo_id": UUID(row["repo_id"]),
            "subject_type": "repo",
            "subject_id": UUID(row["repo_id"]),
        }

    if scope == "org":
        # Validate org exists (cheap lookup)
        row = db.execute(
            text("SELECT id AS org_id FROM orgs WHERE id = :org_id"),
            {"org_id": str(scope_id)},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Org not found.")
        return {"org_id": UUID(row["org_id"]), "repo_id": None, "subject_type": "org", "subject_id": UUID(row["org_id"])}

    if scope == "user":
        row = db.execute(
            text("SELECT id AS user_id, org_id FROM users WHERE id = :user_id"),
            {"user_id": str(scope_id)},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        # user.org_id can be null in this codebase
        org_id = UUID(row["org_id"]) if row.get("org_id") else None
        return {"org_id": org_id, "repo_id": None, "subject_type": "user", "subject_id": UUID(row["user_id"])}

    raise HTTPException(status_code=400, detail="Invalid scope.")


def _fetch_lightweight_metrics(
    db: Session,
    *,
    org_id: Optional[UUID],
    repo_id: Optional[UUID],
    period_start: date,
    period_end: date,
) -> Dict[str, Any]:
    """Fetch lightweight top metrics for prompt context.

    Intentionally avoids heavy joins. Uses analytics tables if present; does not require that
    the analytics compute endpoint has been called (numbers may be 0 if tables are empty).
    """
    if org_id is None:
        return {}

    metrics_sql = text(
        """
        SELECT
          COALESCE(SUM(ard.commits_count), 0)::int AS commits,
          COALESCE(SUM(ard.prs_opened_count), 0)::int AS prs_opened,
          COALESCE(SUM(ard.prs_merged_count), 0)::int AS prs_merged,
          COALESCE(MAX(ard.active_devs_count), 0)::int AS max_active_devs_daily
        FROM analytics_repo_daily ard
        WHERE ard.org_id = :org_id
          AND (:repo_id::uuid IS NULL OR ard.repo_id = :repo_id::uuid)
          AND ard.day >= :period_start::date
          AND ard.day <  :period_end::date
        """
    )
    row = db.execute(
        metrics_sql,
        {
            "org_id": str(org_id),
            "repo_id": (str(repo_id) if repo_id is not None else None),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
        },
    ).mappings().first()

    if not row:
        return {}

    return {
        "commits": int(row.get("commits") or 0),
        "prs_opened": int(row.get("prs_opened") or 0),
        "prs_merged": int(row.get("prs_merged") or 0),
        "max_active_devs_daily": int(row.get("max_active_devs_daily") or 0),
    }


def _build_prompt(*, scope: Scope, period_start: date, period_end: date, metrics: Dict[str, Any]) -> str:
    """Build a compact prompt for a weekly/monthly-ish activity summary."""
    metrics_lines = ""
    if metrics:
        metrics_lines = (
            "\nTop metrics (from analytics aggregates):\n"
            f"- Commits: {metrics.get('commits', 0)}\n"
            f"- PRs opened: {metrics.get('prs_opened', 0)}\n"
            f"- PRs merged: {metrics.get('prs_merged', 0)}\n"
            f"- Max active devs (single day): {metrics.get('max_active_devs_daily', 0)}\n"
        )

    return (
        "You are a concise engineering analytics assistant.\n"
        f"Generate a short summary for scope={scope}.\n"
        f"Period: [{period_start.isoformat()}, {period_end.isoformat()}) UTC.\n"
        f"{metrics_lines}\n"
        "Output:\n"
        "- 4-8 bullet points.\n"
        "- Mention trends/peaks if metrics imply them.\n"
        "- Add a short 'What to do next' section with 2 actionable suggestions.\n"
        "Do not invent numbers not present above."
    ).strip()


async def _call_openai_chat(*, api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call OpenAI Chat Completions via HTTP.

    Returns:
      Dict with keys: text, tokens_in, tokens_out
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate concise activity summaries for git analytics dashboards."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code >= 400:
        # Fail gracefully: do not break app; caller will handle fallback.
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

    data = resp.json()
    text_out = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return {
        "text": text_out,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }


def _simulated_summary_text(*, scope: Scope, period_start: date, period_end: date, metrics: Dict[str, Any]) -> str:
    """Generate a deterministic, clearly-marked simulated summary."""
    return (
        "[SIMULATED SUMMARY — no AI provider key configured]\n\n"
        f"Scope: {scope}\n"
        f"Period: [{period_start.isoformat()}, {period_end.isoformat()}) UTC\n\n"
        "Highlights (based on available aggregates):\n"
        f"- Commits: {metrics.get('commits', 0)}\n"
        f"- PRs opened: {metrics.get('prs_opened', 0)}\n"
        f"- PRs merged: {metrics.get('prs_merged', 0)}\n"
        f"- Max active devs (single day): {metrics.get('max_active_devs_daily', 0)}\n\n"
        "What to do next:\n"
        "- Ensure analytics are up-to-date by calling POST /analytics/compute for this window.\n"
        "- Configure OPENAI_API_KEY to enable real AI-generated narrative summaries.\n"
    ).strip()


# PUBLIC_INTERFACE
@router.post(
    "/compute",
    summary="Compute and persist an AI summary",
    description=(
        "Computes an AI-generated summary for the requested scope and period, persists it in `ai_summaries`, "
        "and returns the stored record.\n\n"
        "If provider keys (e.g., OPENAI_API_KEY) are missing, returns a simulated summary and still persists it."
    ),
    operation_id="ai_summaries_compute_post",
    response_model=AiSummaryOut,
)
async def compute_summary(req: AiSummaryComputeRequest, db: Session = Depends(get_db_session)) -> AiSummaryOut:
    """Compute (or simulate) an AI summary for a scope + period and persist it."""
    _ensure_ai_summaries_table(db)
    _require_period(req.period_start, req.period_end)

    provider = (req.provider or "openai").lower().strip()
    if provider != "openai":
        # Keep surface area small for now; still allow simulated persistence.
        provider = "openai"

    refs = _resolve_scope_refs(db, req.scope, req.scope_id)
    metrics = _fetch_lightweight_metrics(
        db,
        org_id=refs["org_id"],
        repo_id=refs["repo_id"],
        period_start=req.period_start,
        period_end=req.period_end,
    )
    prompt = _build_prompt(scope=req.scope, period_start=req.period_start, period_end=req.period_end, metrics=metrics)

    model = (req.model or _default_openai_model()).strip()
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None

    api_key = _openai_api_key()
    if api_key:
        try:
            result = await _call_openai_chat(api_key=api_key, model=model, prompt=prompt)
            summary_text = result["text"]
            tokens_in = result.get("tokens_in")
            tokens_out = result.get("tokens_out")
            # Cost calculation is provider/model specific; keep optional/blank.
            cost_usd = None
        except Exception as e:
            # Graceful fallback to simulated output if external call fails.
            summary_text = _simulated_summary_text(
                scope=req.scope, period_start=req.period_start, period_end=req.period_end, metrics=metrics
            )
            summary_text += f"\n\n[NOTE: External AI call failed; simulated fallback used: {str(e)}]"
            model = None
            tokens_in = None
            tokens_out = None
            cost_usd = None
    else:
        summary_text = _simulated_summary_text(scope=req.scope, period_start=req.period_start, period_end=req.period_end, metrics=metrics)
        model = None  # Clearly indicates we did not run an external model.

    s = AiSummary(
        repo_id=refs["repo_id"],
        org_id=refs["org_id"],
        subject_type=refs["subject_type"] or req.scope,
        subject_id=refs["subject_id"],
        period_start=req.period_start,
        period_end=req.period_end,
        summary_text=summary_text,
        provider=provider,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        created_at=_utcnow(),
    )

    try:
        db.add(s)
        db.commit()
        db.refresh(s)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail={"error": "ai_summary_persist_failed", "message": str(e)})

    return _summary_to_out(s)


# PUBLIC_INTERFACE
@router.get(
    "/recent",
    summary="List recent AI summaries",
    description="Returns the most recent persisted summaries for dashboard rendering.",
    operation_id="ai_summaries_recent_get",
    response_model=List[AiSummaryOut],
)
def recent_summaries(
    org_id: Optional[UUID] = Query(default=None, description="Optional org UUID to filter."),
    repo_id: Optional[UUID] = Query(default=None, description="Optional repo UUID to filter."),
    subject_type: Optional[str] = Query(default=None, description="Optional subject_type filter (repo|org|user|system)."),
    limit: int = Query(default=20, ge=1, le=200, description="Max summaries returned."),
    db: Session = Depends(get_db_session),
) -> List[AiSummaryOut]:
    """List recent summaries, optionally filtered by org/repo/subject_type."""
    _ensure_ai_summaries_table(db)

    # Build dynamic WHERE safely.
    wheres: List[str] = []
    params: Dict[str, Any] = {"limit": int(limit)}

    if org_id is not None:
        wheres.append("org_id = :org_id::uuid")
        params["org_id"] = str(org_id)
    if repo_id is not None:
        wheres.append("repo_id = :repo_id::uuid")
        params["repo_id"] = str(repo_id)
    if subject_type is not None:
        wheres.append("subject_type = :subject_type")
        params["subject_type"] = subject_type

    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    sql = text(
        f"""
        SELECT
          id, repo_id, org_id, subject_type, subject_id, period_start, period_end,
          summary_text, provider, model, tokens_in, tokens_out, cost_usd, created_at
        FROM ai_summaries
        {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    rows = db.execute(sql, params).mappings().all()
    return [
        AiSummaryOut(
            id=r["id"],
            repo_id=r.get("repo_id"),
            org_id=r.get("org_id"),
            subject_type=r["subject_type"],
            subject_id=r.get("subject_id"),
            period_start=r["period_start"],
            period_end=r["period_end"],
            summary_text=r["summary_text"],
            provider=r["provider"],
            model=r.get("model"),
            tokens_in=r.get("tokens_in"),
            tokens_out=r.get("tokens_out"),
            cost_usd=(float(r["cost_usd"]) if r.get("cost_usd") is not None else None),
            created_at=r["created_at"],
        )
        for r in rows
    ]


# PUBLIC_INTERFACE
@router.get(
    "/{summary_id}",
    summary="Get a persisted AI summary by id",
    description="Fetch a single persisted summary record by its UUID.",
    operation_id="ai_summaries_get_by_id",
    response_model=AiSummaryOut,
)
def get_summary_by_id(
    summary_id: UUID = Path(..., description="AI summary UUID."),
    db: Session = Depends(get_db_session),
) -> AiSummaryOut:
    """Fetch a single persisted AI summary by id."""
    _ensure_ai_summaries_table(db)

    row = db.execute(
        text(
            """
            SELECT
              id, repo_id, org_id, subject_type, subject_id, period_start, period_end,
              summary_text, provider, model, tokens_in, tokens_out, cost_usd, created_at
            FROM ai_summaries
            WHERE id = :id::uuid
            """
        ),
        {"id": str(summary_id)},
    ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Summary not found.")

    return AiSummaryOut(
        id=row["id"],
        repo_id=row.get("repo_id"),
        org_id=row.get("org_id"),
        subject_type=row["subject_type"],
        subject_id=row.get("subject_id"),
        period_start=row["period_start"],
        period_end=row["period_end"],
        summary_text=row["summary_text"],
        provider=row["provider"],
        model=row.get("model"),
        tokens_in=row.get("tokens_in"),
        tokens_out=row.get("tokens_out"),
        cost_usd=(float(row["cost_usd"]) if row.get("cost_usd") is not None else None),
        created_at=row["created_at"],
    )
