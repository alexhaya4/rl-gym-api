from datetime import timedelta

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_is_not_plaintext() -> None:
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert hashed != password


def test_verify_correct_password() -> None:
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True


def test_verify_wrong_password() -> None:
    hashed = hash_password("correctpassword")
    assert verify_password("wrongpassword", hashed) is False


def test_create_access_token() -> None:
    token = create_access_token(data={"sub": "testuser"})
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_valid_token() -> None:
    token = create_access_token(data={"sub": "testuser"})
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"


def test_decode_invalid_token() -> None:
    result = decode_access_token("garbage.invalid.token")
    assert result is None


def test_decode_expired_token() -> None:
    token = create_access_token(
        data={"sub": "testuser"},
        expires_delta=timedelta(seconds=-1),
    )
    result = decode_access_token(token)
    assert result is None
