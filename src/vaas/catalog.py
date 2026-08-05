from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .models import EntityMatch, MediaRecord, SearchHit


def _pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def _unpack(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype=np.float32)


class Catalog:
    """Portable SQLite metadata store with an in-process vector scan."""

    def __init__(self, path: str | Path = "vaas.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    uri TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    timestamp REAL,
                    frame_number INTEGER,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    attention_json TEXT NOT NULL,
                    signals_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_assets_source ON assets(source_uri, timestamp);
                CREATE INDEX IF NOT EXISTS idx_assets_model ON assets(embedding_model);
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    kind TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    centroid BLOB NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS observations (
                    entity_id TEXT NOT NULL REFERENCES entities(id),
                    asset_id TEXT NOT NULL REFERENCES assets(id),
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(entity_id, asset_id)
                );
                """
            )

    def upsert(self, record: MediaRecord, embedding: np.ndarray) -> MediaRecord:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO assets (
                    id, uri, media_type, source_uri, timestamp, frame_number, width, height,
                    sha256, embedding_model, embedding, attention_json, signals_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    uri=excluded.uri, attention_json=excluded.attention_json,
                    signals_json=excluded.signals_json, metadata_json=excluded.metadata_json
                """,
                (
                    record.id,
                    record.uri,
                    record.media_type,
                    record.source_uri,
                    record.timestamp,
                    record.frame_number,
                    record.width,
                    record.height,
                    record.sha256,
                    record.embedding_model,
                    _pack(embedding),
                    json.dumps(record.attention),
                    json.dumps(record.signals),
                    json.dumps(record.metadata),
                ),
            )
        return self.get(record.id) or record

    def get(self, asset_id: str) -> MediaRecord | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return self._record(row) if row else None

    def get_embedding(self, asset_id: str) -> tuple[str, np.ndarray] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT embedding_model, embedding FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return (row["embedding_model"], _unpack(row["embedding"])) if row else None

    def list_source(self, source_uri: str) -> list[MediaRecord]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM assets WHERE source_uri = ? ORDER BY timestamp, frame_number",
                (source_uri,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0])

    def search_text(self, query: str, limit: int = 10) -> list[SearchHit]:
        needle = f"%{query.lower()}%"
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM assets
                WHERE lower(uri) LIKE ? OR lower(metadata_json) LIKE ? OR lower(signals_json) LIKE ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (needle, needle, needle, limit),
            ).fetchall()
        return [SearchHit(self._record(row), 1.0, "metadata") for row in rows]

    def search_vector(
        self, query: np.ndarray, embedding_model: str, limit: int = 10
    ) -> list[SearchHit]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM assets WHERE embedding_model = ?", (embedding_model,)
            ).fetchall()
        if not rows:
            return []
        vectors = [_unpack(row["embedding"]) for row in rows]
        same_size = [
            (row, vector) for row, vector in zip(rows, vectors) if vector.size == query.size
        ]
        if not same_size:
            return []
        matrix = np.vstack([vector for _, vector in same_size])
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = matrix @ query
        indices = np.argsort(scores)[::-1][:limit]
        return [
            SearchHit(self._record(same_size[index][0]), float(scores[index]), "embedding")
            for index in indices
        ]

    def resolve_entity(
        self,
        asset_id: str,
        kind: str = "visual-subject",
        label: str | None = None,
        threshold: float = 0.90,
    ) -> EntityMatch:
        found = self.get_embedding(asset_id)
        if not found:
            raise KeyError(f"Unknown asset: {asset_id}")
        model, vector = found
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM entities WHERE kind = ? AND embedding_model = ?", (kind, model)
            ).fetchall()
            best = None
            best_score = -1.0
            for row in rows:
                centroid = _unpack(row["centroid"])
                if centroid.size == vector.size:
                    score = float(centroid @ vector)
                    if score > best_score:
                        best, best_score = row, score

            created = best is None or best_score < threshold
            if created:
                entity_id = f"ent_{uuid.uuid4().hex[:16]}"
                db.execute(
                    """INSERT INTO entities
                    (id, label, kind, embedding_model, centroid, observation_count)
                    VALUES (?, ?, ?, ?, ?, 1)""",
                    (entity_id, label, kind, model, _pack(vector)),
                )
                confidence, count = 1.0, 1
            else:
                entity_id = best["id"]
                count = int(best["observation_count"]) + 1
                centroid = _unpack(best["centroid"])
                updated = centroid + (vector - centroid) / count
                norm = float(np.linalg.norm(updated))
                if norm > 1e-12:
                    updated /= norm
                db.execute(
                    """UPDATE entities SET centroid = ?, observation_count = ?,
                    label = COALESCE(?, label) WHERE id = ?""",
                    (_pack(updated), count, label, entity_id),
                )
                confidence = best_score
                label = label or best["label"]

            db.execute(
                """INSERT INTO observations(entity_id, asset_id, confidence)
                VALUES (?, ?, ?) ON CONFLICT(entity_id, asset_id)
                DO UPDATE SET confidence=excluded.confidence""",
                (entity_id, asset_id, confidence),
            )

        return EntityMatch(entity_id, label, kind, confidence, created, count)

    @staticmethod
    def _record(row: sqlite3.Row) -> MediaRecord:
        return MediaRecord(
            id=row["id"],
            uri=row["uri"],
            media_type=row["media_type"],
            source_uri=row["source_uri"],
            timestamp=row["timestamp"],
            frame_number=row["frame_number"],
            width=row["width"],
            height=row["height"],
            sha256=row["sha256"],
            embedding_model=row["embedding_model"],
            attention=json.loads(row["attention_json"]),
            signals=json.loads(row["signals_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )
