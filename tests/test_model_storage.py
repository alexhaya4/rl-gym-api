
import pytest
from httpx import AsyncClient

from app.core.storage import LocalStorage


@pytest.fixture
def local_storage(tmp_path):
    return LocalStorage(base_path=str(tmp_path))


async def test_local_storage_save_and_load(local_storage):
    data = b"fake model weights"
    await local_storage.save("test/model.zip", data)
    loaded = await local_storage.load("test/model.zip")
    assert loaded == data


async def test_local_storage_exists(local_storage):
    assert await local_storage.exists("missing.zip") is False
    await local_storage.save("exists.zip", b"data")
    assert await local_storage.exists("exists.zip") is True


async def test_local_storage_delete(local_storage):
    await local_storage.save("deleteme.zip", b"data")
    assert await local_storage.exists("deleteme.zip") is True
    deleted = await local_storage.delete("deleteme.zip")
    assert deleted is True
    assert await local_storage.exists("deleteme.zip") is False
    deleted_again = await local_storage.delete("deleteme.zip")
    assert deleted_again is False


async def test_local_storage_list_files(local_storage):
    await local_storage.save("models/exp1/v1.zip", b"v1")
    await local_storage.save("models/exp1/v2.zip", b"v2")
    await local_storage.save("models/exp1/v3.zip", b"v3")
    files = await local_storage.list_files("models/exp1")
    assert len(files) == 3
    assert "models/exp1/v1.zip" in [f.replace("\\", "/") for f in files]
    assert "models/exp1/v2.zip" in [f.replace("\\", "/") for f in files]
    assert "models/exp1/v3.zip" in [f.replace("\\", "/") for f in files]


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "modeluser",
        "email": "modeluser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "modeluser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_list_model_versions_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/models/experiments/99999",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


async def test_get_nonexistent_model_version(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/models/99999",
        headers=auth_headers,
    )
    assert response.status_code == 404
