from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MultiAgentExperiment(Base):
    __tablename__ = "multi_agent_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(100), nullable=False)
    environment_type: Mapped[str] = mapped_column(String(20), nullable=False)
    n_agents: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    total_timesteps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_episodes: Mapped[int] = mapped_column(Integer, default=0)
    mean_team_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_team_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<MultiAgentExperiment(id={self.id}, name={self.name!r}, status={self.status!r})>"


class AgentPolicy(Base):
    __tablename__ = "agent_policies"
    __table_args__ = (
        UniqueConstraint("experiment_id", "agent_id", name="uq_experiment_agent"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("multi_agent_experiments.id"), nullable=False
    )
    agent_id: Mapped[str] = mapped_column(String(50), nullable=False)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<AgentPolicy(id={self.id}, agent_id={self.agent_id!r}, experiment_id={self.experiment_id})>"
