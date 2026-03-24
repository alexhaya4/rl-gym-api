from app.core.permissions import Permission, has_permission


async def _get_auth_headers(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "rbacuser", "email": "rbac@test.com", "password": "testpass123"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "rbacuser", "password": "testpass123"},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_get_my_permissions(client):
    headers = await _get_auth_headers(client)
    resp = await client.get("/api/v1/rbac/my-permissions", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "permissions" in data
    assert len(data["permissions"]) > 0
    assert data["role"] == "member"


async def test_check_permission_allowed(client):
    headers = await _get_auth_headers(client)
    resp = await client.post(
        "/api/v1/rbac/check",
        json={"permission": "experiment:create"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["allowed"] is True


async def test_check_permission_denied(client):
    headers = await _get_auth_headers(client)
    resp = await client.post(
        "/api/v1/rbac/check",
        json={"permission": "billing:manage"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["allowed"] is False


async def test_list_roles(client):
    resp = await client.get("/api/v1/rbac/roles")
    assert resp.status_code == 200
    data = resp.json()
    assert "owner" in data
    assert "admin" in data
    assert "member" in data
    assert "viewer" in data


def test_has_permission_owner():
    assert has_permission("owner", Permission.BILLING_MANAGE) is True


def test_has_permission_member():
    assert has_permission("member", Permission.BILLING_MANAGE) is False


def test_has_permission_viewer():
    assert has_permission("viewer", Permission.EXPERIMENT_CREATE) is False


async def test_assign_role_requires_auth(client):
    resp = await client.post(
        "/api/v1/rbac/assign",
        json={"user_id": 1, "role": "admin"},
    )
    assert resp.status_code == 401
