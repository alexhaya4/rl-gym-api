from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UserRole(Base):
    __tablename__ = "user_roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_user_org_role"),
    )

    def __repr__(self) -> str:
        return f"<UserRole(id={self.id}, user_id={self.user_id}, org_id={self.organization_id}, role={self.role!r})>"
