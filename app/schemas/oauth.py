from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class OAuthProvider(StrEnum):
    google = "google"
    github = "github"


class OAuthLoginResponse(BaseModel):
    authorization_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str


class OAuthAccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    provider: str
    provider_email: str | None
    provider_username: str | None
    created_at: datetime


class OAuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]
    is_new_user: bool
