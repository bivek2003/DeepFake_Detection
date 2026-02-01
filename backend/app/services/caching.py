"""
Redis caching service.
"""

import json
from typing import Any

import redis.asyncio as redis

from app.logging_config import get_logger
from app.settings import get_settings

logger = get_logger(__name__)

_redis_client: redis.Redis | None = None


async def redis_client() -> redis.Redis | None:
    """Get or create Redis client."""
    global _redis_client
    
    if _redis_client is None:
        settings = get_settings()
        try:
            _redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            _redis_client = None
    
    return _redis_client


async def cache_get(key: str) -> Any | None:
    """Get value from cache."""
    client = await redis_client()
    if client is None:
        return None
    
    try:
        value = await client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None


async def cache_set(
    key: str,
    value: Any,
    expire_seconds: int = 3600,
) -> bool:
    """Set value in cache with expiration."""
    client = await redis_client()
    if client is None:
        return False
    
    try:
        await client.setex(key, expire_seconds, json.dumps(value))
        return True
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete value from cache."""
    client = await redis_client()
    if client is None:
        return False
    
    try:
        await client.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Cache delete failed: {e}")
        return False


def make_analysis_cache_key(sha256: str, model_version: str) -> str:
    """Create cache key for analysis result."""
    return f"analysis:{sha256}:{model_version}"
