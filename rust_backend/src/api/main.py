from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.ai_routes import router as ai_router
from src.api.analytics_routes import router as analytics_router
from src.api.db import get_db_session, init_engine
from src.api.oauth_routes import router as oauth_router
from src.api.webhook_routes import router as webhook_router

openapi_tags = [
    {"name": "Health", "description": "Service and database health checks."},
    {"name": "Auth", "description": "OAuth login flows for GitHub/GitLab/Bitbucket."},
    {"name": "Webhooks", "description": "Provider webhook ingestion endpoints (GitHub/GitLab/Bitbucket)."},
    {"name": "Analytics", "description": "On-demand analytics aggregation and dashboard endpoints."},
    {"name": "AI", "description": "AI summary generation and retrieval endpoints."},
]

app = FastAPI(
    title="CodeInsight Dashboard API",
    description="Backend API for the CodeInsight dashboard (Git analytics + AI summaries).",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
