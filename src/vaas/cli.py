from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .pipeline import VAAS


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaas", description="Give agents a searchable visual memory."
    )
    parser.add_argument("--db", default=os.getenv("VAAS_DB", "vaas.db"))
    parser.add_argument(
        "--embedder", choices=("visual", "openclip"), default=os.getenv("VAAS_EMBEDDER", "visual")
    )
    parser.add_argument("--face-signals", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Index an image, video, or directory")
    index.add_argument("path")
    index.add_argument("--no-recursive", action="store_true")
    index.add_argument("--sample-every", type=float, default=2.0)
    index.add_argument("--tag", action="append", default=[])

    search = commands.add_parser("search", help="Search by text or example image")
    search.add_argument("query", nargs="?")
    search.add_argument("--image")
    search.add_argument("--limit", type=int, default=10)

    inspect = commands.add_parser("inspect", help="Inspect an indexed asset")
    inspect.add_argument("asset_id")

    timeline = commands.add_parser("timeline", help="Read attention across a video")
    timeline.add_argument("source")

    entity = commands.add_parser("resolve-entity", help="Map an asset to a stable entity")
    entity.add_argument("asset_id")
    entity.add_argument("--kind", default="visual-subject")
    entity.add_argument("--label")
    entity.add_argument("--threshold", type=float, default=0.90)

    export = commands.add_parser("export", help="Materialize an image or indexed video frame")
    export.add_argument("asset_id")
    export.add_argument("output")

    commands.add_parser("serve", help="Run the MCP server over stdio")
    commands.add_parser("status", help="Show catalog status")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from .mcp_server import run

        run(db_path=args.db, embedder=args.embedder, face_signals=args.face_signals)
        return

    vaas = VAAS(args.db, embedder=args.embedder, face_signals=args.face_signals)
    if args.command == "index":
        records = vaas.index_path(
            args.path,
            recursive=not args.no_recursive,
            sample_every=args.sample_every,
            tags=args.tag,
        )
        _emit({"indexed": len(records), "assets": [record.to_dict() for record in records]})
    elif args.command == "search":
        _emit(
            [hit.to_dict() for hit in vaas.search(args.query, image=args.image, limit=args.limit)]
        )
    elif args.command == "inspect":
        _emit(vaas.inspect(args.asset_id).to_dict())
    elif args.command == "timeline":
        _emit(vaas.attention_timeline(args.source))
    elif args.command == "resolve-entity":
        _emit(
            vaas.resolve_entity(
                args.asset_id, kind=args.kind, label=args.label, threshold=args.threshold
            ).to_dict()
        )
    elif args.command == "export":
        _emit({"path": str(vaas.export_frame(args.asset_id, args.output))})
    elif args.command == "status":
        _emit(
            {
                "ok": True,
                "assets": vaas.catalog.count(),
                "database": str(Path(args.db).resolve()),
                "embedder": vaas.embedder.name,
            }
        )


if __name__ == "__main__":
    main()
