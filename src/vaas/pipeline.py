from __future__ import annotations

import hashlib
import io
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .catalog import Catalog
from .features import AttentionAnalyzer, Embedder, FaceSignalAnalyzer, make_embedder
from .models import EntityMatch, MediaRecord, SearchHit

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class VAAS:
    """High-level API shared by the Python, CLI, and MCP surfaces."""

    def __init__(
        self,
        db_path: str | Path = "vaas.db",
        embedder: str | Embedder = "visual",
        face_signals: bool = False,
    ) -> None:
        self.catalog = Catalog(db_path)
        self.embedder = make_embedder(embedder) if isinstance(embedder, str) else embedder
        self.attention = AttentionAnalyzer()
        self.face_analyzer = FaceSignalAnalyzer() if face_signals else None

    def index_image(
        self,
        path: str | Path,
        *,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaRecord:
        path = Path(path).expanduser().resolve()
        with Image.open(path) as image:
            image.load()
            return self._index_pil(
                image,
                uri=str(path),
                source_uri=str(path),
                media_type="image",
                tags=tags,
                metadata=metadata,
            )

    def index_directory(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        tags: Iterable[str] | None = None,
    ) -> list[MediaRecord]:
        root = Path(path).expanduser().resolve()
        pattern = "**/*" if recursive else "*"
        records = []
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(self.index_image(candidate, tags=tags))
        return records

    def index_path(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        sample_every: float = 2.0,
        tags: Iterable[str] | None = None,
    ) -> list[MediaRecord]:
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            return self.index_directory(path, recursive=recursive, tags=tags)
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return self.index_video(path, sample_every=sample_every, tags=tags)
        return [self.index_image(path, tags=tags)]

    def index_video(
        self,
        path: str | Path,
        *,
        sample_every: float = 2.0,
        max_frames: int | None = None,
        scene_threshold: float = 0.35,
        tags: Iterable[str] | None = None,
    ) -> list[MediaRecord]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install video support with: pip install '.[video]'") from exc

        path = Path(path).expanduser().resolve()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        stride = max(1, round(fps * sample_every))
        frame_number = 0
        sampled = 0
        shot = 0
        previous_hist: np.ndarray | None = None
        previous_gray: np.ndarray | None = None
        records: list[MediaRecord] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % stride:
                    frame_number += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([frame], [0, 1], None, [16, 16], [0, 256, 0, 256])
                cv2.normalize(hist, hist)
                scene_delta = (
                    float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                    if previous_hist is not None
                    else 0.0
                )
                if scene_delta >= scene_threshold:
                    shot += 1
                motion = (
                    float(np.mean(cv2.absdiff(previous_gray, gray))) / 255.0
                    if previous_gray is not None
                    else 0.0
                )
                timestamp = frame_number / fps
                record = self._index_pil(
                    image,
                    uri=f"{path}#t={timestamp:.3f}",
                    source_uri=str(path),
                    media_type="video-frame",
                    timestamp=timestamp,
                    frame_number=frame_number,
                    tags=tags,
                    metadata={"shot": shot, "scene_delta": scene_delta, "motion": motion},
                )
                records.append(record)
                previous_hist, previous_gray = hist, gray
                sampled += 1
                frame_number += 1
                if max_frames is not None and sampled >= max_frames:
                    break
        finally:
            capture.release()
        return records

    def search(
        self,
        query: str | None = None,
        *,
        image: str | Path | Image.Image | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        if image is not None:
            if isinstance(image, Image.Image):
                query_image = image
                vector = self.embedder.embed_image(query_image)
            else:
                with Image.open(Path(image).expanduser()) as query_image:
                    vector = self.embedder.embed_image(query_image)
            return self.catalog.search_vector(vector, self.embedder.name, limit)
        if not query:
            raise ValueError("Provide a text query or an example image")
        try:
            vector = self.embedder.embed_text(query)
        except NotImplementedError:
            return self.catalog.search_text(query, limit)
        return self.catalog.search_vector(vector, self.embedder.name, limit)

    def inspect(self, asset_id: str) -> MediaRecord:
        record = self.catalog.get(asset_id)
        if not record:
            raise KeyError(f"Unknown asset: {asset_id}")
        return record

    def attention_timeline(self, source: str | Path) -> list[dict[str, Any]]:
        source_uri = str(Path(source).expanduser().resolve())
        return [
            {
                "id": record.id,
                "timestamp": record.timestamp,
                "frame_number": record.frame_number,
                "attention": record.attention,
                "motion": record.metadata.get("motion"),
                "shot": record.metadata.get("shot"),
            }
            for record in self.catalog.list_source(source_uri)
        ]

    def resolve_entity(
        self,
        asset_id: str,
        *,
        kind: str = "visual-subject",
        label: str | None = None,
        threshold: float = 0.90,
    ) -> EntityMatch:
        return self.catalog.resolve_entity(asset_id, kind=kind, label=label, threshold=threshold)

    def export_frame(self, asset_id: str, output: str | Path) -> Path:
        record = self.inspect(asset_id)
        output = Path(output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        source = Path(record.source_uri)
        if record.media_type == "image":
            shutil.copy2(source, output)
            return output

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install video support with: pip install '.[video]'") from exc
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(record.timestamp or 0.0) * 1000.0)
            ok, frame = capture.read()
            if not ok or not cv2.imwrite(str(output), frame):
                raise ValueError(f"Could not export {asset_id} from {source}")
        finally:
            capture.release()
        return output

    def _index_pil(
        self,
        image: Image.Image,
        *,
        uri: str,
        source_uri: str,
        media_type: str,
        timestamp: float | None = None,
        frame_number: int | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaRecord:
        normalized = image.convert("RGB")
        buffer = io.BytesIO()
        normalized.save(buffer, format="JPEG", quality=90)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        identity = hashlib.sha256(f"{uri}:{digest}".encode()).hexdigest()[:24]
        embedding = self.embedder.embed_image(normalized)
        attention = self.attention.analyze(normalized).to_dict()
        signals: dict[str, Any] = {}
        if self.face_analyzer:
            signals["face"] = self.face_analyzer.analyze(normalized).to_dict()
        combined_metadata = dict(metadata or {})
        if tags:
            combined_metadata["tags"] = sorted(set(tags))
        record = MediaRecord(
            id=f"asset_{identity}",
            uri=uri,
            media_type=media_type,
            source_uri=source_uri,
            timestamp=timestamp,
            frame_number=frame_number,
            width=normalized.width,
            height=normalized.height,
            sha256=digest,
            embedding_model=self.embedder.name,
            attention=attention,
            signals=signals,
            metadata=combined_metadata,
        )
        return self.catalog.upsert(record, embedding)
