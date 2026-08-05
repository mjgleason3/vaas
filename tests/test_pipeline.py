from pathlib import Path

from PIL import Image, ImageDraw

from vaas import VAAS


def make_image(path: Path, color: str, offset: int = 0) -> None:
    image = Image.new("RGB", (120, 80), "#101626")
    draw = ImageDraw.Draw(image)
    draw.ellipse((25 + offset, 10, 85 + offset, 70), fill=color)
    image.save(path)


def test_index_search_inspect_and_export(tmp_path: Path) -> None:
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    make_image(red, "#ff3344")
    make_image(blue, "#3388ff")
    vision = VAAS(tmp_path / "catalog.db")

    red_record = vision.index_image(red, tags=["warm", "portfolio"])
    vision.index_image(blue, tags=["cool", "portfolio"])

    hits = vision.search(image=red, limit=2)
    assert hits[0].record.id == red_record.id
    assert hits[0].score > hits[1].score
    assert vision.search("warm")[0].record.id == red_record.id
    assert vision.inspect(red_record.id).metadata["tags"] == ["portfolio", "warm"]

    exported = vision.export_frame(red_record.id, tmp_path / "out" / "red.png")
    assert exported.exists()


def test_directory_index_and_entity_resolution(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    make_image(first, "#55dd88", offset=0)
    make_image(second, "#55dd88", offset=2)
    vision = VAAS(tmp_path / "catalog.db")
    records = vision.index_directory(tmp_path, tags=["green subject"])
    assert len(records) == 2

    one = vision.resolve_entity(records[0].id, label="green subject", threshold=0.8)
    two = vision.resolve_entity(records[1].id, threshold=0.8)
    assert one.created is True
    assert two.created is False
    assert one.entity_id == two.entity_id
    assert two.observation_count == 2


def test_flat_image_attention_is_well_formed(tmp_path: Path) -> None:
    path = tmp_path / "flat.png"
    Image.new("RGB", (32, 32), "black").save(path)
    record = VAAS(tmp_path / "catalog.db").index_image(path)
    assert record.attention["focus_x"] == 0.5
    assert record.attention["focus_y"] == 0.5
