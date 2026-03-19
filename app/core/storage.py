import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3

from app.config import get_settings


class BaseStorage(ABC):
    @abstractmethod
    async def save(self, file_path: str, data: bytes) -> str: ...

    @abstractmethod
    async def load(self, file_path: str) -> bytes: ...

    @abstractmethod
    async def delete(self, file_path: str) -> bool: ...

    @abstractmethod
    async def exists(self, file_path: str) -> bool: ...

    @abstractmethod
    async def list_files(self, prefix: str) -> list[str]: ...


class LocalStorage(BaseStorage):
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(self, file_path: str, data: bytes) -> str:
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, full_path.write_bytes, data)
        return str(full_path)

    async def load(self, file_path: str) -> bytes:
        full_path = self.base_path / file_path
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, full_path.read_bytes)

    async def delete(self, file_path: str) -> bool:
        full_path = self.base_path / file_path
        if not full_path.exists():
            return False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, full_path.unlink)
        return True

    async def exists(self, file_path: str) -> bool:
        full_path = self.base_path / file_path
        return full_path.exists()

    async def list_files(self, prefix: str) -> list[str]:
        target = self.base_path / prefix
        if not target.exists():
            return []
        parent = target if target.is_dir() else target.parent
        pattern = "*" if target.is_dir() else target.name + "*"
        return [str(p.relative_to(self.base_path)) for p in parent.rglob(pattern)]


class S3Storage(BaseStorage):
    def __init__(
        self,
        bucket: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket
        kwargs: dict[str, Any] = {"region_name": region}
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self._client = boto3.client("s3", **kwargs)

    def _run(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def save(self, file_path: str, data: bytes) -> str:
        await self._run(self._client.put_object, Bucket=self.bucket, Key=file_path, Body=data)
        return f"s3://{self.bucket}/{file_path}"

    async def load(self, file_path: str) -> bytes:
        response = await self._run(self._client.get_object, Bucket=self.bucket, Key=file_path)
        body: bytes = response["Body"].read()
        return body

    async def delete(self, file_path: str) -> bool:
        if not await self.exists(file_path):
            return False
        await self._run(self._client.delete_object, Bucket=self.bucket, Key=file_path)
        return True

    async def exists(self, file_path: str) -> bool:
        try:
            await self._run(self._client.head_object, Bucket=self.bucket, Key=file_path)
        except self._client.exceptions.NoSuchKey:
            return False
        except Exception:
            return False
        return True

    async def list_files(self, prefix: str) -> list[str]:
        response = await self._run(
            self._client.list_objects_v2, Bucket=self.bucket, Prefix=prefix
        )
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents]


@lru_cache
def get_storage() -> BaseStorage:
    settings = get_settings()
    if settings.STORAGE_BACKEND == "s3":
        if not settings.S3_BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME is required when STORAGE_BACKEND=s3")
        return S3Storage(
            bucket=settings.S3_BUCKET_NAME,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region=settings.AWS_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL,
        )
    return LocalStorage(base_path=settings.STORAGE_LOCAL_PATH)
