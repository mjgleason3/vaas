"""Self-contained demo: create three images, index them, and find a visual neighbor."""

from pathlib import Path

from PIL import Image, ImageDraw

from vaas import VAAS


def make_card(path: Path, color: str, shape: str) -> None:
    image = Image.new("RGB", (320, 200), "#101626")
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((90, 30, 230, 170), fill=color)
    else:
        draw.rounded_rectangle((80, 40, 240, 160), radius=20, fill=color)
    image.save(path)


def main() -> None:
    workspace = Path(".vaas-demo").resolve()
    workspace.mkdir(exist_ok=True)
    make_card(workspace / "red-circle.png", "#ff4d6d", "circle")
    make_card(workspace / "red-square.png", "#ff4d6d", "square")
    make_card(workspace / "blue-circle.png", "#4d9dff", "circle")

    vision = VAAS(workspace / "catalog.db")
    records = vision.index_directory(workspace, tags=["synthetic", "demo"])
    hits = vision.search(image=workspace / "red-circle.png", limit=3)

    print(f"Indexed {len(records)} assets into {workspace / 'catalog.db'}")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. {Path(hit.record.uri).name:18} similarity={hit.score:.3f}")
    print("\nTry: vaas --db .vaas-demo/catalog.db search demo")


if __name__ == "__main__":
    main()
