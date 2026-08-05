from PIL import Image, ImageDraw

from vaas.features import AttentionAnalyzer, VisualEmbedder


def test_visual_embedding_is_normalized_and_stable() -> None:
    image = Image.new("RGB", (80, 60), "red")
    embedder = VisualEmbedder()
    first = embedder.embed_image(image)
    second = embedder.embed_image(image)
    assert first.shape == (120,)
    assert abs(float(first @ first) - 1.0) < 1e-5
    assert (first == second).all()


def test_attention_focus_moves_toward_salient_region() -> None:
    image = Image.new("RGB", (200, 100), "#777777")
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 20, 195, 80), fill="white")
    signal = AttentionAnalyzer().analyze(image)
    assert signal.focus_x > 0.5
    assert 0.0 <= signal.focus_y <= 1.0
    assert 0.0 <= signal.entropy <= 1.0
    assert len(signal.grid) == 64
