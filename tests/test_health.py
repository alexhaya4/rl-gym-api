from httpx import AsyncClient


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_returns_expected_body(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.json() == {"status": "ok", "version": "0.1.0"}
