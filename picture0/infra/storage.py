"""
Storage backend implementations.

Provides LocalFileStorageBackend (default) and an S3StorageBackend skeleton.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from picture.providers.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalFileStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def save(self, local_path: str, key: str) -> str:
        """Copy a local file into the storage directory."""
        dest = self._base / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, str(dest))
        uri = f"local://{dest.resolve()}"
        logger.debug("Saved %s -> %s", local_path, uri)
        return uri

    def load(self, uri: str, local_path: str) -> str:
        """Load a file from a local:// URI to a local path."""
        source = uri.replace("local://", "")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_path)
        return local_path

    def exists(self, uri: str) -> bool:
        """Check if a URI exists."""
        path = uri.replace("local://", "")
        return Path(path).exists()

    def get_uri(self, key: str) -> str:
        """Get the full URI for a storage key."""
        return f"local://{(self._base / key).resolve()}"


class S3StorageBackend(StorageBackend):
    """
    S3/MinIO storage backend skeleton.

    Requires: boto3
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._endpoint_url = endpoint_url
        self._client = None

    def _get_client(self):  # type: ignore[no-untyped-def]
        """Lazy initialization of S3 client."""
        if self._client is None:
            try:
                import boto3  # type: ignore[import-untyped]
                kwargs = {}
                if self._endpoint_url:
                    kwargs["endpoint_url"] = self._endpoint_url
                self._client = boto3.client("s3", **kwargs)
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError
                raise ProviderNotAvailableError("S3 (boto3)")
        return self._client

    def save(self, local_path: str, key: str) -> str:
        """Upload a local file to S3."""
        client = self._get_client()
        full_key = f"{self._prefix}{key}"
        client.upload_file(local_path, self._bucket, full_key)
        return f"s3://{self._bucket}/{full_key}"

    def load(self, uri: str, local_path: str) -> str:
        """Download a file from S3."""
        client = self._get_client()
        # Parse s3://bucket/key
        parts = uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, local_path)
        return local_path

    def exists(self, uri: str) -> bool:
        """Check if an S3 URI exists."""
        client = self._get_client()
        parts = uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def get_uri(self, key: str) -> str:
        """Get the full S3 URI for a key."""
        return f"s3://{self._bucket}/{self._prefix}{key}"
