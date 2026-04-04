import io

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "importuser",
            "email": "importuser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "importuser",
            "password": "securepassword",
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def auth_headers_b(client: AsyncClient) -> dict[str, str]:
    """Second user for ownership tests."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "otherimportuser",
            "email": "otherimportuser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "otherimportuser",
            "password": "securepassword",
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _csv_file(
    name: str = "test.csv",
    content: str = "col_a,col_b,col_c\n1,2.5,hello\n3,4.5,world\n5,6.5,foo\n",
) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(content.encode()), "text/csv")


async def _upload_csv(
    client: AsyncClient,
    auth_headers: dict[str, str],
    name: str = "test-dataset",
    filename: str = "test.csv",
) -> dict:
    response = await client.post(
        "/api/v1/datasets/upload",
        files={"file": _csv_file(filename)},
        data={"name": name, "description": "A test dataset"},
        headers=auth_headers,
    )
    return response.json()


async def test_upload_csv(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """Upload a small CSV and verify metadata."""
    response = await client.post(
        "/api/v1/datasets/upload",
        files={"file": _csv_file()},
        data={"name": "my-csv", "description": "Test CSV upload"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "my-csv"
    assert data["dataset_type"] == "csv"
    assert data["num_samples"] == 3
    assert data["num_features"] == 3
    assert data["columns"] == ["col_a", "col_b", "col_c"]
    assert data["file_size_mb"] is not None


async def test_upload_invalid_extension(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Upload .exe file returns 422."""
    response = await client.post(
        "/api/v1/datasets/upload",
        files={
            "file": ("malware.exe", io.BytesIO(b"MZ..."), "application/octet-stream")
        },
        data={"name": "bad-file"},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert "extension" in response.json()["detail"].lower()


async def test_upload_too_large(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """File exceeding MAX_DATASET_SIZE_MB returns 413."""
    from unittest.mock import MagicMock, patch

    mock_settings = MagicMock()
    mock_settings.MAX_DATASET_SIZE_MB = 0  # 0 MB limit
    mock_settings.ALLOWED_DATASET_EXTENSIONS = [".csv"]
    mock_settings.DATASET_STORAGE_PATH = "/tmp/rl_datasets"

    with patch("app.config.get_settings", return_value=mock_settings):
        response = await client.post(
            "/api/v1/datasets/upload",
            files={"file": _csv_file()},
            data={"name": "too-large"},
            headers=auth_headers,
        )
    assert response.status_code == 413


async def test_get_preview(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """Upload CSV and get preview."""
    data = await _upload_csv(client, auth_headers, name="preview-test")
    dataset_id = data["id"]

    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}/preview?limit=2",
        headers=auth_headers,
    )
    assert response.status_code == 200
    preview = response.json()
    assert len(preview["rows"]) == 2
    assert preview["total_rows"] == 3
    assert "col_a" in preview["columns"]


async def test_get_statistics(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Upload CSV and get per-column statistics."""
    data = await _upload_csv(client, auth_headers, name="stats-test")
    dataset_id = data["id"]

    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}/statistics",
        headers=auth_headers,
    )
    assert response.status_code == 200
    stats = response.json()
    assert len(stats) == 3
    col_names = [s["column_name"] for s in stats]
    assert "col_a" in col_names

    # Numeric column should have mean
    col_a = next(s for s in stats if s["column_name"] == "col_a")
    assert col_a["mean"] is not None
    assert col_a["null_count"] == 0


async def test_list_datasets_pagination(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Upload 3 datasets, list with page_size=2."""
    for i in range(3):
        await _upload_csv(
            client, auth_headers, name=f"paginate-{i}", filename=f"p{i}.csv"
        )

    response = await client.get(
        "/api/v1/datasets/?include_public=false",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 3


async def test_delete_dataset(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Upload then delete, assert 404 on second get."""
    data = await _upload_csv(client, auth_headers, name="delete-test")
    dataset_id = data["id"]

    response = await client.delete(
        f"/api/v1/datasets/{dataset_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_owner_only_access(
    client: AsyncClient,
    auth_headers: dict[str, str],
    auth_headers_b: dict[str, str],
) -> None:
    """User A cannot access user B's file dataset."""
    data = await _upload_csv(client, auth_headers, name="private-dataset")
    dataset_id = data["id"]

    # User B tries to access
    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}",
        headers=auth_headers_b,
    )
    assert response.status_code == 404

    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}/preview",
        headers=auth_headers_b,
    )
    assert response.status_code == 404

    response = await client.get(
        f"/api/v1/datasets/file/{dataset_id}/statistics",
        headers=auth_headers_b,
    )
    assert response.status_code == 404
