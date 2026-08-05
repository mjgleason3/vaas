"""Index media, retrieve a moment, resolve it, and export evidence for an agent."""

import argparse
from pathlib import Path

from vaas import VAAS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("media", help="An image, video, or directory")
    parser.add_argument("--query-image", required=True, help="Example image to search for")
    parser.add_argument("--db", default="vaas.db")
    parser.add_argument("--sample-every", type=float, default=2.0)
    args = parser.parse_args()

    vision = VAAS(args.db)
    indexed = vision.index_path(args.media, sample_every=args.sample_every, tags=["agent-demo"])
    hits = vision.search(image=args.query_image, limit=5)
    if not hits:
        raise SystemExit("No matching assets. Ensure index and query use the same embedder.")

    best = hits[0]
    entity = vision.resolve_entity(best.record.id, kind="visual-subject")
    extension = Path(best.record.source_uri).suffix if best.record.media_type == "image" else ".jpg"
    output = Path("exports") / f"{best.record.id}{extension}"
    vision.export_frame(best.record.id, output)

    print(f"Indexed: {len(indexed)}")
    print(f"Best match: {best.record.uri} ({best.score:.3f})")
    print(f"Entity: {entity.entity_id} ({entity.confidence:.3f})")
    print(f"Evidence: {output.resolve()}")


if __name__ == "__main__":
    main()
