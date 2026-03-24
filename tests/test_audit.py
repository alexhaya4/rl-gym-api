from app.models.audit_log import AuditLog


async def _register_and_login(client, username="audituser", email="audit@test.com"):
    """Helper to register and login, returning auth headers."""
    await client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": "testpass123"},
    )
    login_resp = await client.post(
        "/api/v1/auth/login",
        data={"username": username, "password": "testpass123"},
    )
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_audit_log_created_on_register(client):
    headers = await _register_and_login(client, "reguser", "reg@test.com")
    resp = await client.get("/api/v1/audit/logs/me", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    event_types = [e["event_type"] for e in events]
    assert "register" in event_types


async def test_audit_log_created_on_login(client):
    headers = await _register_and_login(client, "loginuser", "login@test.com")
    resp = await client.get("/api/v1/audit/logs/me", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    event_types = [e["event_type"] for e in events]
    assert "login" in event_types


async def test_audit_log_me_endpoint(client):
    headers = await _register_and_login(client, "meuser", "me@test.com")
    resp = await client.get("/api/v1/audit/logs/me", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


async def test_audit_log_list(client):
    headers = await _register_and_login(client, "listuser", "list@test.com")
    resp = await client.get("/api/v1/audit/logs", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


async def test_audit_log_filter_by_event_type(client):
    headers = await _register_and_login(client, "filteruser", "filter@test.com")
    resp = await client.get(
        "/api/v1/audit/logs", params={"event_type": "login"}, headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    for item in data["items"]:
        assert item["event_type"] == "login"


async def test_audit_log_immutable():
    assert not hasattr(AuditLog, "updated_at")
