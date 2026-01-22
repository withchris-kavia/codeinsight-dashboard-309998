import os
from typing import List

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.ai_routes import router as ai_router
from src.api.analytics_routes import router as analytics_router
from src.api.db import get_db_session, init_engine
from src.api.notifications_routes import router as notifications_router
from src.api.oauth_routes import router as oauth_router
from src.api.webhook_routes import router as webhook_router

openapi_tags = [
    {"name": "Health", "description": "Service and database health checks."},
    {"name": "Auth", "description": "OAuth login flows for GitHub/GitLab/Bitbucket."},
    {"name": "Webhooks", "description": "Provider webhook ingestion endpoints (GitHub/GitLab/Bitbucket)."},
    {"name": "Analytics", "description": "On-demand analytics aggregation and dashboard endpoints."},
    {"name": "AI", "description": "AI summary generation and retrieval endpoints."},
    {"name": "Notifications", "description": "Notification configuration and dispatch endpoints (Slack/email)."},
]


def _split_csv_env(name: str) -> List[str]:
    """Split an env var by commas, trimming whitespace, and dropping empty items."""
    raw = os.getenv(name, "") or ""
    return [item.strip() for item in raw.split(",") if item and item.strip()]


# PUBLIC_INTERFACE
def get_allowed_cors_origins() -> List[str]:
    """
    Compute allowed CORS origins for the API.

    Defaults are aligned to local Next.js preview on port 3000.

    You can override/extend via either:
      - FRONTEND_ORIGIN="https://preview-host:3000"               (single origin)
      - FRONTEND_ORIGINS="http://localhost:3000,https://...:3000" (comma-separated)

    Note:
      - With allow_credentials=True, you cannot use "*" for allow_origins.
    """
    env_origin_single = (os.getenv("FRONTEND_ORIGIN") or "").strip()
    env_origins_csv = _split_csv_env("FRONTEND_ORIGINS")

    # Default Next.js preview origins (per task instruction).
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Preserve order while de-duplicating.
    seen = set()
    merged: List[str] = []
    for o in [env_origin_single, *env_origins_csv, *defaults]:
        if not o:
            continue
        if o not in seen:
            merged.append(o)
            seen.add(o)

    return merged


app = FastAPI(
    title="CodeInsight Dashboard API",
    description="Backend API for the CodeInsight dashboard (Git analytics + AI summaries).",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# NOTE:
# - With allow_credentials=True, allow_origins cannot be ["*"].
# - We also allow any https? origin *ending in :3000* via regex to support preview URLs
#   (while staying aligned with the preview port requirement).
# - Optionally override the regex via FRONTEND_ORIGIN_REGEX if your preview runs on a different port.
cors_origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX", r"^https?://.*:3000$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_cors_origins(),
    allow_origin_regex=cors_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# OAuth routes
app.include_router(oauth_router)

# Webhook routes
app.include_router(webhook_router)

# Analytics routes
app.include_router(analytics_router)

# AI summary routes
app.include_router(ai_router)

# Notifications routes
app.include_router(notifications_router)


@app.on_event("startup")
def _startup_init_db() -> None:
    """Initialize DB engine early so connection issues surface on startup logs."""
    init_engine()


@app.get(
    "/",
    tags=["Health"],
    summary="Health check (legacy)",
    description="Legacy health check endpoint. Prefer GET /health.",
)
def health_check_root():
    """Health check endpoint (legacy root path)."""
    return {"status": "ok", "message": "Healthy"}


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Returns OK if the API process is running.",
)
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get(
    "/db/health",
    tags=["Health"],
    summary="Database health check",
    description="Runs a trivial SELECT 1 query to verify database connectivity.",
)
def db_health_check(db: Session = Depends(get_db_session)):
    """Database connectivity check using SELECT 1."""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "reachable"}
