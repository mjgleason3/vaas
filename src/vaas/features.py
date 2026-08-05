from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from .models import AttentionSignal


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


class Embedder(Protocol):
    name: str

    def embed_image(self, image: Image.Image) -> np.ndarray: ...

    def embed_text(self, text: str) -> np.ndarray: ...


class VisualEmbedder:
    """Tiny deterministic image descriptor used when no learned model is installed.

    It intentionally supports example-image similarity, not semantic text/image matching.
    """

    name = "visual-v1"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        features: list[np.ndarray] = []

        for channel in range(3):
            hist, _ = np.histogram(rgb[..., channel], bins=16, range=(0.0, 1.0))
            features.append(hist.astype(np.float32))

        gray = rgb.mean(axis=2)
        gray_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
        features.append(gray_hist.astype(np.float32))

        # Spatial color layout keeps two similarly colored but differently composed images apart.
        blocks = []
        for row in np.array_split(rgb, 4, axis=0):
            for block in np.array_split(row, 4, axis=1):
                blocks.extend(block.mean(axis=(0, 1)).tolist())
        features.append(np.asarray(blocks, dtype=np.float32))

        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) + math.pi) % math.pi
        orientations = []
        for start in np.linspace(0, math.pi, 9)[:-1]:
            mask = (angle >= start) & (angle < start + math.pi / 8)
            orientations.append(float(magnitude[mask].sum()))
        features.append(np.asarray(orientations, dtype=np.float32))

        return _unit(np.concatenate(features))

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "The built-in visual embedder searches tags/metadata for text. "
            "Install VAAS with [semantic] and select openclip for semantic search."
        )


class OpenClipEmbedder:
    """Lazy OpenCLIP adapter; weights are downloaded only when explicitly selected."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install semantic support with: pip install '.[semantic]'") from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
        self.name = f"openclip:{model_name}:{pretrained}"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            vector = self._model.encode_image(tensor)
        return _unit(vector.cpu().numpy()[0])

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            vector = self._model.encode_text(tokens)
        return _unit(vector.cpu().numpy()[0])


def make_embedder(name: str = "visual") -> Embedder:
    if name == "visual":
        return VisualEmbedder()
    if name == "openclip":
        return OpenClipEmbedder()
    raise ValueError(f"Unknown embedder: {name!r}; choose 'visual' or 'openclip'")


class AttentionAnalyzer:
    """Fast bottom-up attention proxy based on contrast, edges, and saturation."""

    def __init__(self, grid_size: int = 8) -> None:
        self.grid_size = grid_size

    def analyze(self, image: Image.Image) -> AttentionSignal:
        rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        contrast = np.abs(gray - float(gray.mean()))
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        saliency = 0.45 * (gx + gy) + 0.35 * contrast + 0.20 * saturation
        saliency = np.maximum(saliency, 0.0)

        total = float(saliency.sum())
        if total <= 1e-12:
            return AttentionSignal(score=0.0, focus_x=0.5, focus_y=0.5, entropy=1.0, grid=[])

        weights = saliency / total
        height, width = weights.shape
        ys, xs = np.mgrid[0:height, 0:width]
        focus_x = float((weights * xs).sum() / max(width - 1, 1))
        focus_y = float((weights * ys).sum() / max(height - 1, 1))

        cells: list[float] = []
        for row in np.array_split(saliency, self.grid_size, axis=0):
            for cell in np.array_split(row, self.grid_size, axis=1):
                cells.append(float(cell.mean()))
        grid = _unit(np.asarray(cells, dtype=np.float32))
        probability = weights.reshape(-1)
        entropy = -float(np.sum(probability * np.log(probability + 1e-12))) / math.log(
            probability.size
        )

        return AttentionSignal(
            score=float(np.percentile(saliency, 95)),
            focus_x=focus_x,
            focus_y=focus_y,
            entropy=entropy,
            grid=[round(float(value), 6) for value in grid],
        )


@dataclass(slots=True)
class FaceSignals:
    available: bool
    face_count: int = 0
    smiling_faces: int = 0
    boxes: list[list[int]] | None = None
    backend: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "face_count": self.face_count,
            "smiling_faces": self.smiling_faces,
            "boxes": self.boxes or [],
            "backend": self.backend,
        }


class FaceSignalAnalyzer:
    """OpenCV Haar adapter for coarse face/smile signals, loaded only when requested."""

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install face/video support with: pip install '.[video]'") from exc
        self.cv2 = cv2
        root = cv2.data.haarcascades
        self.faces = cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml")
        self.smiles = cv2.CascadeClassifier(root + "haarcascade_smile.xml")

    def analyze(self, image: Image.Image) -> FaceSignals:
        frame = np.asarray(image.convert("RGB"))
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2GRAY)
        faces = self.faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        smiling = 0
        boxes: list[list[int]] = []
        for x, y, width, height in faces:
            boxes.append([int(x), int(y), int(width), int(height)])
            lower_face = gray[y + height // 2 : y + height, x : x + width]
            smiles = self.smiles.detectMultiScale(
                lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(12, 8)
            )
            smiling += int(len(smiles) > 0)
        return FaceSignals(
            available=True,
            face_count=len(faces),
            smiling_faces=smiling,
            boxes=boxes,
            backend="opencv-haar",
        )
