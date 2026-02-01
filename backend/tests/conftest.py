"""
Pytest configuration and fixtures.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Set test environment
os.environ["DEMO_MODE"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = ""
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"

from app.main import app
from app.persistence.db import Base, get_async_session


# Create test database engine
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
)

TestSessionLocal = sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    import io
    from PIL import Image
    
    # Create a simple test image
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_video_path(tmp_path) -> str:
    """Create sample video file for testing."""
    import cv2
    import numpy as np
    
    video_path = tmp_path / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (224, 224))
    
    for _ in range(30):  # 1 second of video
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    return str(video_path)
