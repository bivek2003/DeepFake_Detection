"""
Storage service for file management.
Supports local filesystem with interface designed for S3 swap.
"""

import os
from pathlib import Path
from typing import Literal

import aiofiles

from app.logging_config import get_logger
from app.metrics import STORAGE_OPERATIONS

logger = get_logger(__name__)


class StorageService:
    """
    Storage service abstraction.
    Currently implements local filesystem storage.
    Interface designed to be easily swappable to S3.
    """

    def __init__(
        self,
        upload_dir: str = "/app/uploads",
        assets_dir: str = "/app/assets",
        backend: Literal["local", "s3"] = "local",
    ):
        self.upload_dir = Path(upload_dir)
        self.assets_dir = Path(assets_dir)
        self.backend = backend

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Storage service initialized",
            extra={
                "backend": backend,
                "upload_dir": str(self.upload_dir),
                "assets_dir": str(self.assets_dir),
            },
        )

    async def save_upload(
        self,
        content: bytes,
        filename: str,
        subfolder: str = "",
    ) -> str:
        """
        Save uploaded file.

        Args:
            content: File content as bytes
            filename: Filename to save as
            subfolder: Optional subfolder within uploads

        Returns:
            Relative path to saved file
        """
        try:
            # Build path
            if subfolder:
                dir_path = self.upload_dir / subfolder
            else:
                dir_path = self.upload_dir

            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / filename

            # Write file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Return relative path
            relative_path = str(file_path.relative_to(self.upload_dir))

            STORAGE_OPERATIONS.labels(operation="upload", status="success").inc()
            logger.debug(f"Saved upload: {relative_path}")

            return relative_path

        except Exception as e:
            STORAGE_OPERATIONS.labels(operation="upload", status="error").inc()
            logger.error(f"Failed to save upload: {e}")
            raise

    async def save_asset(
        self,
        content: bytes,
        filename: str,
        subfolder: str = "",
    ) -> str:
        """
        Save generated asset.

        Args:
            content: File content as bytes
            filename: Filename to save as
            subfolder: Optional subfolder within assets

        Returns:
            Relative path to saved file
        """
        try:
            # Build path
            if subfolder:
                dir_path = self.assets_dir / subfolder
            else:
                dir_path = self.assets_dir

            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / filename

            # Write file
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Return relative path
            relative_path = str(file_path.relative_to(self.assets_dir))

            STORAGE_OPERATIONS.labels(operation="asset", status="success").inc()
            logger.debug(f"Saved asset: {relative_path}")

            return relative_path

        except Exception as e:
            STORAGE_OPERATIONS.labels(operation="asset", status="error").inc()
            logger.error(f"Failed to save asset: {e}")
            raise

    def get_upload_path(self, relative_path: str) -> Path:
        """Get full path to uploaded file."""
        return self.upload_dir / relative_path

    def get_asset_path(self, relative_path: str) -> Path:
        """Get full path to asset file."""
        return self.assets_dir / relative_path

    async def read_upload(self, relative_path: str) -> bytes:
        """Read uploaded file content."""
        file_path = self.get_upload_path(relative_path)
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def read_asset(self, relative_path: str) -> bytes:
        """Read asset file content."""
        file_path = self.get_asset_path(relative_path)
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def delete_upload(self, relative_path: str) -> bool:
        """Delete uploaded file."""
        try:
            file_path = self.get_upload_path(relative_path)
            if file_path.exists():
                os.remove(file_path)
                STORAGE_OPERATIONS.labels(operation="delete", status="success").inc()
                return True
            return False
        except Exception as e:
            STORAGE_OPERATIONS.labels(operation="delete", status="error").inc()
            logger.error(f"Failed to delete upload: {e}")
            return False

    def exists_upload(self, relative_path: str) -> bool:
        """Check if uploaded file exists."""
        return self.get_upload_path(relative_path).exists()

    def exists_asset(self, relative_path: str) -> bool:
        """Check if asset file exists."""
        return self.get_asset_path(relative_path).exists()
