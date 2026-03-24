from app.services.oauth import generate_oauth_state, verify_oauth_state


async def _get_auth_headers(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "oauthuser", "email": "oauth@test.com", "password": "testpass123"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "oauthuser", "password": "testpass123"},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_google_login_redirect(client):
    resp = await client.get("/api/v1/oauth/google/login")
    assert resp.status_code == 200
    data = resp.json()
    assert "accounts.google.com" in data["authorization_url"]
    assert data["state"]


async def test_github_login_redirect(client):
    resp = await client.get("/api/v1/oauth/github/login")
    assert resp.status_code == 200
    data = resp.json()
    assert "github.com" in data["authorization_url"]
    assert data["state"]


def test_oauth_state_generation():
    state = generate_oauth_state("google")
    assert isinstance(state, str)
    assert len(state) > 0


def test_oauth_state_verification_valid():
    state = generate_oauth_state("google")
    assert verify_oauth_state(state, "google") is True


def test_oauth_state_verification_wrong_provider():
    state = generate_oauth_state("google")
    assert verify_oauth_state(state, "github") is False


async def test_list_oauth_accounts_empty(client):
    headers = await _get_auth_headers(client)
    resp = await client.get("/api/v1/oauth/accounts", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_google_callback_missing_code(client):
    resp = await client.get("/api/v1/oauth/google/callback")
    assert resp.status_code == 422


async def test_github_callback_missing_code(client):
    resp = await client.get("/api/v1/oauth/github/callback")
    assert resp.status_code == 422
