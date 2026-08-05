from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AttentionSignal:
    """A compact, model-independent description of attention inside a frame."""

    score: float
    focus_x: float
    focus_y: float
    entropy: float
    grid: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MediaRecord:
    id: str
    uri: str
    media_type: str
    source_uri: str
    width: int
    height: int
    sha256: str
    embedding_model: str
    timestamp: float | None = None
    frame_number: int | None = None
    attention: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    record: MediaRecord
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reason": self.reason, **self.record.to_dict()}


@dataclass(slots=True)
class EntityMatch:
    entity_id: str
    label: str | None
    kind: str
    confidence: float
    created: bool
    observation_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
