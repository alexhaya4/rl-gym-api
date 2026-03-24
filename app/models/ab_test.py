from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ABTest(Base):
    __tablename__ = "ab_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    environment_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=False
    )
    model_version_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=False
    )
    traffic_split_a: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    n_eval_episodes_per_model: Mapped[int] = mapped_column(Integer, default=100)
    significance_level: Mapped[float] = mapped_column(Float, default=0.05)
    winner: Mapped[str | None] = mapped_column(String(1), nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    statistical_test: Mapped[str] = mapped_column(String(20), default="ttest")
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<ABTest(id={self.id}, name={self.name!r}, status={self.status!r})>"


class ABTestResult(Base):
    __tablename__ = "ab_test_results"
    __table_args__ = (
        Index("ix_ab_test_results_ab_test_id", "ab_test_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ab_test_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ab_tests.id"), nullable=False
    )
    model_variant: Mapped[str] = mapped_column(String(1), nullable=False)
    episode_number: Mapped[int] = mapped_column(Integer, nullable=False)
    total_reward: Mapped[float] = mapped_column(Float, nullable=False)
    episode_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<ABTestResult(id={self.id}, ab_test_id={self.ab_test_id}, variant={self.model_variant!r})>"
