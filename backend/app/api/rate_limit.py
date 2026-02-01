"""
Rate limiting middleware using token bucket algorithm.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging_config import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Simple token bucket rate limiter."""
    
    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        
        # Add tokens based on time elapsed
        time_passed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + time_passed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using in-memory token buckets.
    For production, consider using Redis for distributed rate limiting.
    """
    
    def __init__(
        self,
        app: Callable,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.rate = requests_per_window / window_seconds
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.rate, requests_per_window)
        )
        
        # Paths to exclude from rate limiting
        self.excluded_paths = {
            "/api/v1/healthz",
            "/api/v1/readyz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        bucket = self.buckets[client_id]
        
        if not bucket.consume():
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": self.window_seconds,
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.window_seconds),
                },
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(bucket.remaining)
        
        return response
