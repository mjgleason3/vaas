from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import VAAS


def create_server(
    db_path: str | Path = "vaas.db", embedder: str = "visual", face_signals: bool = False
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install MCP support with: pip install '.[mcp]'") from exc

    service = VAAS(db_path, embedder=embedder, face_signals=face_signals)
    mcp = FastMCP("VAAS")

    @mcp.tool()
    def visual_status() -> dict[str, Any]:
        """Check the visual catalog and selected feature backend."""
        return {
            "ok": True,
            "assets": service.catalog.count(),
            "embedder": service.embedder.name,
            "database": str(Path(db_path).resolve()),
        }

    @mcp.tool()
    def index_visual_path(
        path: str,
        recursive: bool = True,
        sample_every_seconds: float = 2.0,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index one image/video or a directory; videos are sampled into searchable frames."""
        records = service.index_path(
            path, recursive=recursive, sample_every=sample_every_seconds, tags=tags
        )
        return {"indexed": len(records), "asset_ids": [record.id for record in records]}

    @mcp.tool()
    def search_visual_memory(
        query: str | None = None, example_image: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search indexed imagery by text/tags or by an example-image path."""
        return [
            hit.to_dict()
            for hit in service.search(query, image=example_image, limit=min(limit, 100))
        ]

    @mcp.tool()
    def inspect_visual_asset(asset_id: str) -> dict[str, Any]:
        """Return provenance, attention, detected signals, and metadata for an asset."""
        return service.inspect(asset_id).to_dict()

    @mcp.tool()
    def read_attention_timeline(source_path: str) -> list[dict[str, Any]]:
        """Read focus, saliency, motion, and scene changes across an indexed video."""
        return service.attention_timeline(source_path)

    @mcp.tool()
    def resolve_visual_entity(
        asset_id: str,
        kind: str = "visual-subject",
        label: str | None = None,
        similarity_threshold: float = 0.90,
    ) -> dict[str, Any]:
        """Associate a visual observation with a stable prototype entity."""
        return service.resolve_entity(
            asset_id, kind=kind, label=label, threshold=similarity_threshold
        ).to_dict()

    @mcp.tool()
    def export_visual_frame(asset_id: str, output_path: str) -> dict[str, str]:
        """Materialize an indexed still or video frame so the agent can inspect/use it."""
        return {"path": str(service.export_frame(asset_id, output_path))}

    return mcp


def run(
    db_path: str | Path = "vaas.db", embedder: str = "visual", face_signals: bool = False
) -> None:
    create_server(db_path, embedder, face_signals).run(transport="stdio")


if __name__ == "__main__":
    run()
