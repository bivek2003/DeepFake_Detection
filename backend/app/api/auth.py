"""
JWT Authentication (optional - disabled by default).
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.settings import Settings, get_settings

security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """Token data payload."""

    sub: str
    exp: datetime


def create_access_token(
    data: dict,
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def verify_token(
    token: str,
    settings: Settings,
) -> TokenData:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        sub: str = payload.get("sub", "")
        exp: datetime = datetime.fromtimestamp(payload.get("exp", 0), tz=UTC)

        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        return TokenData(sub=sub, exp=exp)

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
        ) from e


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """
    Get current user from JWT token (optional).
    Returns None if auth is disabled or no token provided.
    """
    if not settings.auth_enabled:
        return None

    if credentials is None:
        return None

    token_data = verify_token(credentials.credentials, settings)
    return token_data.sub


async def get_current_user_required(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """
    Get current user from JWT token (required if auth is enabled).
    Raises 401 if auth is enabled but no valid token provided.
    """
    if not settings.auth_enabled:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = verify_token(credentials.credentials, settings)
    return token_data.sub
