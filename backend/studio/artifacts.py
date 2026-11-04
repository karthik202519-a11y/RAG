"""Artifact store: original files and processing outputs, addressed by key.

Local folder by default; an S3-compatible store (MinIO, S3) when S3_ENDPOINT is set.
Keys look like  projects/<project_id>/versions/<version_id>/source.pdf
"""
import re
from pathlib import Path
from typing import Protocol

from .config import settings

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_\-./]+$")


def _check(key: str) -> str:
    if not _SAFE_KEY.match(key) or ".." in key.split("/") or key.startswith("/"):
        raise ValueError(f"unsafe artifact key: {key!r}")
    return key


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def local_path(self, key: str) -> Path: ...


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        p = (self.root / _check(key)).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("key escapes the store")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> None:
        import shutil
        p = self._path(prefix.rstrip("/"))
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def local_path(self, key: str) -> Path:
        return self._path(key)


class S3Store:
    """S3-compatible store (MinIO, AWS S3). Objects are cached to a local scratch folder for parsers."""

    def __init__(self) -> None:
        import boto3  # optional dependency, only needed for the Docker profile
        from botocore.config import Config

        self.bucket = settings.s3_bucket
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint,
                                   aws_access_key_id=settings.s3_access_key,
                                   aws_secret_access_key=settings.s3_secret_key,
                                   config=Config(signature_version="s3v4"), region_name="us-east-1")
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)
        self.scratch = LocalStore(settings.data_dir / "scratch")

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=_check(key), Body=data)

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=_check(key))["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=_check(key))
            return True
        except Exception:
            return False

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=_check(prefix.rstrip("/")) + "/"):
            for obj in page.get("Contents", []):
                self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def local_path(self, key: str) -> Path:
        self.scratch.put(key, self.get(key))
        return self.scratch.local_path(key)


_store: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    global _store
    if _store is None:
        _store = S3Store() if settings.s3_endpoint else LocalStore(settings.data_dir / "artifacts")
    return _store


def reset_store() -> None:
    global _store
    _store = None
