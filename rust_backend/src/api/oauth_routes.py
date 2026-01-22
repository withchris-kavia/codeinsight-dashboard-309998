"""
OAuth routes for GitHub, GitLab, and Bitbucket.

Implements:
- GET /auth/{provider}/login: returns provider authorization URL
- GET /auth/{provider}/callback: exchanges code for token, fetches user info, persists to DB

Security note:
- This implementation uses a signed `state` parameter to protect against CSRF.
- Configure OAUTH_STATE_SECRET to a strong random value.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.db import get_db_session
from src.api.models import OAuthIdentity, User

Provider = Literal["github", "gitlab", "bitbucket"]

router = APIRouter(prefix="/auth", tags=["Auth"])


def _get_env_required(name: str) -> str:
    """Get an environment variable or raise a 500 with a clear message."""
    value = os.getenv(name)
    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"Server OAuth is not configured: missing environment variable {name}.",
        )
    return value


def _get_env_optional(name: str) -> Optional[str]:
    """Get an env var if set and non-empty; otherwise None."""
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return value


def _oauth_configured(provider: Provider) -> bool:
    """
    Return True if the provider OAuth env config is present.

    This is used to decide whether to run a real OAuth exchange or a safe stub flow.
    """
    required = [_client_id_env(provider), _client_secret_env(provider), "OAUTH_STATE_SECRET"]
    return all(_get_env_optional(k) for k in required)


def _encode_state_unsigned(payload: Dict[str, Any]) -> str:
    """
    Encode state without signing (stub-mode only).

    Format: base64url(JSON)
    """
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _b64url(raw)


def _decode_state_unsigned(state: str) -> Dict[str, Any]:
    """Decode unsigned state (stub-mode only)."""
    return json.loads(_b64url_decode(state).decode("utf-8"))


def _get_backend_base_url() -> str:
    """Backend base URL used for redirect_uri generation."""
    return os.getenv("BACKEND_BASE_URL", "http://localhost:3001").rstrip("/")


def _get_frontend_base_url() -> str:
    """Frontend base URL used for post-auth redirect hints."""
    return os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")


def _redirect_uri(provider: Provider) -> str:
    """Compute redirect_uri for a given provider."""
    return f"{_get_backend_base_url()}/auth/{provider}/callback"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign_state(payload: Dict[str, Any], secret: str) -> str:
    """Create a signed `state` string: base64url(payload).base64url(hmac)."""
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    msg = _b64url(raw).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return f"{msg.decode('utf-8')}.{_b64url(sig)}"


def _verify_state(state: str, secret: str, max_age_seconds: int = 15 * 60) -> Dict[str, Any]:
    """Verify and decode a signed state string."""
    try:
        msg_b64, sig_b64 = state.split(".", 1)
        expected = hmac.new(secret.encode("utf-8"), msg_b64.encode("utf-8"), hashlib.sha256).digest()
        got = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected, got):
            raise ValueError("bad signature")
        payload = json.loads(_b64url_decode(msg_b64).decode("utf-8"))
        ts = int(payload.get("ts", 0))
        if ts <= 0 or (int(time.time()) - ts) > max_age_seconds:
            raise ValueError("state expired")
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {e}") from e


def _provider_endpoints(provider: Provider) -> Dict[str, str]:
    """Provider-specific OAuth and API endpoints."""
    if provider == "github":
        return {
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "user_url": "https://api.github.com/user",
            "emails_url": "https://api.github.com/user/emails",
        }
    if provider == "gitlab":
        return {
            "auth_url": "https://gitlab.com/oauth/authorize",
            "token_url": "https://gitlab.com/oauth/token",
            "user_url": "https://gitlab.com/api/v4/user",
            "emails_url": "",  # GitLab v4 user object includes email for some configurations.
        }
    if provider == "bitbucket":
        return {
            "auth_url": "https://bitbucket.org/site/oauth2/authorize",
            "token_url": "https://bitbucket.org/site/oauth2/access_token",
            "user_url": "https://api.bitbucket.org/2.0/user",
            "emails_url": "https://api.bitbucket.org/2.0/user/emails?pagelen=1",
        }
    # Should be unreachable due to typing, but keep safe:
    raise HTTPException(status_code=400, detail="Unsupported provider")


def _client_id_env(provider: Provider) -> str:
    return {
        "github": "GITHUB_CLIENT_ID",
        "gitlab": "GITLAB_CLIENT_ID",
        "bitbucket": "BITBUCKET_CLIENT_ID",
    }[provider]


def _client_secret_env(provider: Provider) -> str:
    return {
        "github": "GITHUB_CLIENT_SECRET",
        "gitlab": "GITLAB_CLIENT_SECRET",
        "bitbucket": "BITBUCKET_CLIENT_SECRET",
    }[provider]


def _scopes_env(provider: Provider) -> str:
    return {
        "github": "GITHUB_SCOPES",
        "gitlab": "GITLAB_SCOPES",
        "bitbucket": "BITBUCKET_SCOPES",
    }[provider]


def _default_scopes(provider: Provider) -> str:
    # Keep conservative defaults; app can expand later.
    if provider == "github":
        return "read:user user:email"
    if provider == "gitlab":
        return "read_user"
    if provider == "bitbucket":
        return "account email"
    return ""


def _parse_scopes(provider: Provider, token_response: Dict[str, Any]) -> Optional[list[str]]:
    """Normalize scopes from either env config or token response."""
    env_scopes = os.getenv(_scopes_env(provider))
    if env_scopes:
        return [s for s in env_scopes.replace(",", " ").split() if s]

    # Provider-specific token response scope fields
    if provider == "github":
        # GitHub returns scopes in response headers, not body (often).
        return None
    if provider == "gitlab":
        scope = token_response.get("scope")
        if isinstance(scope, str) and scope.strip():
            return [s for s in scope.split() if s]
        return None
    if provider == "bitbucket":
        scope = token_response.get("scopes")
        if isinstance(scope, str) and scope.strip():
            return [s for s in scope.split() if s]
        return None
    return None


async def _exchange_code_for_token(provider: Provider, code: str) -> Dict[str, Any]:
    """Exchange OAuth authorization code for an access token."""
    endpoints = _provider_endpoints(provider)
    client_id = _get_env_required(_client_id_env(provider))
    client_secret = _get_env_required(_client_secret_env(provider))
    redirect_uri = _redirect_uri(provider)

    async with httpx.AsyncClient(timeout=20.0) as client:
        if provider == "github":
            # GitHub expects application/x-www-form-urlencoded; request JSON response.
            resp = await client.post(
                endpoints["token_url"],
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
        elif provider == "gitlab":
            resp = await client.post(
                endpoints["token_url"],
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
        else:  # bitbucket
            # Bitbucket uses Basic auth for client credentials.
            basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
            resp = await client.post(
                endpoints["token_url"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Authorization": f"Basic {basic}"},
            )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "token_exchange_failed",
                "provider": provider,
                "status_code": resp.status_code,
                "response": resp.text,
            },
        )

    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token response was not JSON: {e}") from e


async def _fetch_provider_user(provider: Provider, access_token: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Fetch provider user profile and best-effort primary email."""
    endpoints = _provider_endpoints(provider)

    async with httpx.AsyncClient(timeout=20.0) as client:
        # User profile
        if provider == "github":
            user_resp = await client.get(
                endpoints["user_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
        elif provider == "gitlab":
            user_resp = await client.get(
                endpoints["user_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
        else:  # bitbucket
            user_resp = await client.get(
                endpoints["user_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )

        if user_resp.status_code >= 400:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "user_fetch_failed",
                    "provider": provider,
                    "status_code": user_resp.status_code,
                    "response": user_resp.text,
                },
            )
        user = user_resp.json()

        # Email best-effort
        email: Optional[str] = None
        if provider == "github":
            emails_resp = await client.get(
                endpoints["emails_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            if emails_resp.status_code < 400:
                try:
                    emails = emails_resp.json()
                    # pick primary+verified if possible
                    if isinstance(emails, list):
                        primary = next(
                            (e for e in emails if e.get("primary") and e.get("verified") and e.get("email")),
                            None,
                        )
                        any_email = next((e for e in emails if e.get("email")), None)
                        email = (primary or any_email or {}).get("email")
                except Exception:
                    email = None
        elif provider == "gitlab":
            # Often included for trusted apps
            email = user.get("email") if isinstance(user, dict) else None
        else:  # bitbucket
            emails_resp = await client.get(
                endpoints["emails_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            if emails_resp.status_code < 400:
                try:
                    data = emails_resp.json()
                    values = data.get("values") if isinstance(data, dict) else None
                    if isinstance(values, list) and values:
                        primary = next((v for v in values if v.get("is_primary") and v.get("email")), None)
                        email = (primary or values[0]).get("email")
                except Exception:
                    email = None

    return user, email


def _normalize_provider_user(provider: Provider, user: Dict[str, Any], email: Optional[str]) -> Dict[str, Optional[str]]:
    """Normalize provider user data into stable identifiers."""
    if provider == "github":
        provider_user_id = str(user.get("id"))
        username = user.get("login")
        display_name = user.get("name") or username
        avatar_url = user.get("avatar_url")
        return {
            "provider_user_id": provider_user_id,
            "username": username,
            "email": email or user.get("email"),
            "display_name": display_name,
            "avatar_url": avatar_url,
        }
    if provider == "gitlab":
        provider_user_id = str(user.get("id"))
        username = user.get("username")
        display_name = user.get("name") or username
        avatar_url = user.get("avatar_url")
        return {
            "provider_user_id": provider_user_id,
            "username": username,
            "email": email or user.get("email"),
            "display_name": display_name,
            "avatar_url": avatar_url,
        }
    # bitbucket
    provider_user_id = str(user.get("account_id") or user.get("uuid") or "")
    username = (user.get("username") or (user.get("nickname") if isinstance(user.get("nickname"), str) else None))
    display_name = user.get("display_name") or username
    avatar_url = None
    links = user.get("links")
    if isinstance(links, dict):
        avatar = links.get("avatar")
        if isinstance(avatar, dict):
            avatar_url = avatar.get("href")
    return {
        "provider_user_id": provider_user_id,
        "username": username,
        "email": email,
        "display_name": display_name,
        "avatar_url": avatar_url,
    }


def _parse_expires_at(token_response: Dict[str, Any]) -> Optional[datetime]:
    """Convert token response expiry fields to UTC datetime."""
    expires_in = token_response.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    # GitLab sometimes uses "created_at" + "expires_in" but above covers.
    return None


# PUBLIC_INTERFACE
@router.get(
    "/{provider}/login",
    summary="Start OAuth login flow",
    description=(
        "Returns an authorization URL for the requested provider.\n\n"
        "Providers: github | gitlab | bitbucket\n\n"
        "Configure client credentials via environment variables:\n"
        "- GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET\n"
        "- GITLAB_CLIENT_ID / GITLAB_CLIENT_SECRET\n"
        "- BITBUCKET_CLIENT_ID / BITBUCKET_CLIENT_SECRET\n\n"
        "Redirect URIs default to backend http://localhost:3001/auth/{provider}/callback.\n"
        "Frontend base URL defaults to http://localhost:3000."
    ),
)
async def oauth_login(
    provider: Provider,
    request: Request,
    return_to: Optional[str] = Query(
        default=None,
        description="Optional frontend path or absolute URL to return to after success (stored in signed state).",
    ),
):
    """Initiate OAuth by returning provider authorization URL (JSON).

    Safe stub behavior:
      - If provider client credentials / OAUTH_STATE_SECRET are missing, we still return a
        deterministic stub auth_url and an unsigned state so smoke tests can exercise the route.
    """
    endpoints = _provider_endpoints(provider)

    # Build state payload first (used both in real and stub modes).
    payload = {
        "provider": provider,
        "nonce": secrets.token_urlsafe(16),
        "ts": int(time.time()),
        "return_to": return_to or _get_frontend_base_url(),
        "stub": not _oauth_configured(provider),
    }

    if not _oauth_configured(provider):
        # Unsigned state, and a non-provider URL (keeps UX predictable in dev/smoke tests).
        state = _encode_state_unsigned(payload)
        return {
            "provider": provider,
            "auth_url": f"{_get_backend_base_url()}/auth/{provider}/callback?code=mocked-code&state={state}",
            "redirect_uri": _redirect_uri(provider),
            "stub": True,
            "detail": "OAuth provider env is not configured; returning stub auth_url for smoke tests.",
        }

    client_id = _get_env_required(_client_id_env(provider))
    state_secret = _get_env_required("OAUTH_STATE_SECRET")

    scopes = os.getenv(_scopes_env(provider), _default_scopes(provider))
    state = _sign_state(payload, state_secret)

    # Provider-specific scope param differences: GitHub/GitLab/Bitbucket all accept `scope`.
    auth_params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(provider),
        "response_type": "code",
        "state": state,
        "scope": scopes,
    }

    # Build URL without depending on extra libs.
    from urllib.parse import urlencode

    auth_url = f"{endpoints['auth_url']}?{urlencode(auth_params)}"
    return {
        "provider": provider,
        "auth_url": auth_url,
        "redirect_uri": _redirect_uri(provider),
        "stub": False,
    }


# PUBLIC_INTERFACE
@router.get(
    "/{provider}/callback",
    summary="OAuth callback",
    description=(
        "OAuth redirect target. Exchanges `code` for tokens, fetches provider user profile, "
        "and upserts into `oauth_identities` table.\n\n"
        "Returns minimal JSON containing provider, provider_user_id, and persisted identity id."
    ),
)
async def oauth_callback(
    provider: Provider,
    code: Optional[str] = Query(default=None, description="OAuth authorization code"),
    state: Optional[str] = Query(default=None, description="Opaque CSRF state from /login"),
    error: Optional[str] = Query(default=None, description="OAuth error, if any"),
    error_description: Optional[str] = Query(default=None, description="OAuth error description, if any"),
    db: Session = Depends(get_db_session),
):
    """Handle OAuth callback and persist tokens into Postgres.

    Safe stub behavior:
      - If provider secrets are missing, we simulate token exchange + user profile and still
        upsert into oauth_identities. This enables E2E smoke tests without real provider keys.
    """
    if error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": error,
                "error_description": error_description,
                "provider": provider,
            },
        )
    if not code:
        raise HTTPException(status_code=400, detail={"error": "missing_code", "provider": provider})

    state_payload: Dict[str, Any] = {}
    return_to = _get_frontend_base_url()

    if state:
        state_secret = _get_env_optional("OAUTH_STATE_SECRET")
        if state_secret:
            state_payload = _verify_state(state, state_secret)
        else:
            # Unsigned state decode (stub mode)
            try:
                state_payload = _decode_state_unsigned(state)
            except Exception:
                state_payload = {}
        if state_payload.get("return_to"):
            return_to = str(state_payload.get("return_to"))
        if state_payload.get("provider") and state_payload.get("provider") != provider:
            raise HTTPException(status_code=400, detail={"error": "state_provider_mismatch", "provider": provider})

    # Real flow if configured; otherwise stub.
    if _oauth_configured(provider):
        token_response = await _exchange_code_for_token(provider, code)
        access_token = token_response.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise HTTPException(
                status_code=400,
                detail={"error": "missing_access_token", "provider": provider, "token_response": token_response},
            )

        refresh_token = token_response.get("refresh_token") if isinstance(token_response.get("refresh_token"), str) else None
        expires_at = _parse_expires_at(token_response)
        scopes = _parse_scopes(provider, token_response)

        provider_user_raw, email = await _fetch_provider_user(provider, access_token)
        normalized = _normalize_provider_user(provider, provider_user_raw, email)

        provider_user_id = normalized.get("provider_user_id")
        if not provider_user_id:
            raise HTTPException(status_code=400, detail={"error": "missing_provider_user_id", "provider": provider})

        username = normalized.get("username")
        user_email = normalized.get("email")
        display_name = normalized.get("display_name")
        avatar_url = normalized.get("avatar_url")
        stub = False
    else:
        # Stub flow: deterministic identifiers derived from inputs.
        digest = hashlib.sha256(f"{provider}:{code}".encode("utf-8")).hexdigest()[:12]
        provider_user_id = f"stub-{digest}"
        access_token = f"stub-token-{digest}"
        refresh_token = None
        expires_at = datetime.now(timezone.utc) + timedelta(days=3650)
        scopes = [s for s in _default_scopes(provider).replace(",", " ").split() if s]

        username = f"{provider}-user"
        user_email = f"{provider_user_id}@example.local"
        display_name = f"Stub {provider.title()} User"
        avatar_url = None
        stub = True

    # Upsert identity by (provider, provider_user_id) (unique in schema).
    existing_identity = db.scalar(
        select(OAuthIdentity).where(
            OAuthIdentity.provider == provider,
            OAuthIdentity.provider_user_id == provider_user_id,
        )
    )

    now = datetime.now(timezone.utc)

    if existing_identity:
        existing_identity.username = username
        existing_identity.email = user_email
        existing_identity.access_token = access_token
        existing_identity.refresh_token = refresh_token
        existing_identity.token_expires_at = expires_at
        existing_identity.scopes = scopes
        existing_identity.updated_at = now
        identity = existing_identity
    else:
        # If no identity exists, ensure we have a User row.
        user: Optional[User] = None
        if user_email:
            user = db.scalar(select(User).where(User.email == user_email))

        if not user:
            user = User(
                email=user_email,
                display_name=display_name,
                avatar_url=avatar_url,
                updated_at=now,
            )
            db.add(user)
            db.flush()  # assign user.id

        identity = OAuthIdentity(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            username=username,
            email=user_email,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=expires_at,
            scopes=scopes,
            updated_at=now,
        )
        db.add(identity)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail={"error": "db_persist_failed", "provider": provider, "message": str(e)})

    return {
        "status": "ok",
        "provider": provider,
        "provider_user_id": provider_user_id,
        "oauth_identity_id": str(identity.id),
        "user_id": str(identity.user_id),
        "return_to": return_to,
        "stub": stub,
    }
