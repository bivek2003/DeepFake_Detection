"""
File hashing utilities.
"""

import hashlib
from pathlib import Path


def compute_sha256(content: bytes) -> str:
    """Compute SHA256 hash of bytes content."""
    return hashlib.sha256(content).hexdigest()


def compute_sha256_file(file_path: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def compute_md5(content: bytes) -> str:
    """Compute MD5 hash of bytes content."""
    return hashlib.md5(content).hexdigest()
