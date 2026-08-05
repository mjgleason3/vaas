"""Render the README demo as an MP4 plus a GitHub-friendly animated GIF."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 960, 540
FPS = 30
DURATION = 19.5

BG = "#07111f"
PANEL = "#101b30"
PANEL_2 = "#17243c"
TEXT = "#e8f2ff"
MUTED = "#8fa8c8"
MINT = "#62e6c6"
BLUE = "#5ca8ff"
PURPLE = "#a977ff"
PINK = "#ff6f91"

FONT_PATHS = (
    Path("/System/Library/Fonts/SFNS.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
MONO_PATHS = (
    Path("/System/Library/Fonts/SFNSMono.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
)


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    for path in MONO_PATHS if mono else FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    label: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = TEXT,
) -> None:
    box = draw.textbbox((0, 0), label, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), label, font=text_font, fill=fill)


def base_frame() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for x in range(0, WIDTH, 80):
        draw.line((x, 0, x, HEIGHT), fill="#0d1c31", width=1)
    for y in range(0, HEIGHT, 60):
        draw.line((0, y, WIDTH, y), fill="#0d1c31", width=1)
    return image


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: str = MINT) -> None:
    text_font = font(14, mono=True)
    width = int(draw.textlength(label, font=text_font)) + 28
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 29), radius=14, fill="#13273b", outline=color)
    draw.text((x + 14, y + 6), label, font=text_font, fill=color)


def hero_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    pulse = 1.0 + 0.04 * math.sin(progress * math.pi * 4)
    reveal = ease(progress / 0.35)
    cx, cy = 480, 245
    eye_w, eye_h = int(350 * pulse * reveal), int(130 * pulse * reveal)
    if eye_w > 4:
        draw.ellipse((cx - eye_w, cy - eye_h, cx + eye_w, cy + eye_h), outline=BLUE, width=5)
        draw.ellipse((cx - 88, cy - 88, cx + 88, cy + 88), outline=MINT, width=5)
        draw.ellipse((cx - 31, cy - 31, cx + 31, cy + 31), fill=PURPLE)
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=TEXT)
    center_text(draw, (480, 62), "VAAS", font(42), TEXT)
    center_text(draw, (480, 394), "VISION AS A SERVICE", font(23), TEXT)
    center_text(draw, (480, 432), "Visual memory for any AI agent", font(18), MUTED)
    return image


def media_card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], kind: str, color: str, phase: float
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#263a58", width=2)
    draw.rounded_rectangle((x1 + 12, y1 + 12, x2 - 12, y2 - 43), radius=12, fill="#0a1424")
    if kind == "IMAGES":
        draw.ellipse((x1 + 35, y1 + 30, x1 + 89, y1 + 84), fill=color)
        draw.polygon([(x1 + 25, y2 - 55), (x1 + 72, y1 + 62), (x1 + 112, y2 - 55)], fill="#284966")
    elif kind == "VIDEO":
        offset = int(14 * math.sin(phase * math.pi * 2))
        draw.rounded_rectangle(
            (x1 + 39 + offset, y1 + 30, x1 + 98 + offset, y1 + 92), 12, fill=color
        )
        draw.polygon([(x1 + 62, y1 + 46), (x1 + 62, y1 + 78), (x1 + 86, y1 + 62)], fill=BG)
    else:
        draw.ellipse((x1 + 45, y1 + 36, x1 + 105, y1 + 96), outline=color, width=5)
        draw.ellipse((x1 + 66, y1 + 57, x1 + 84, y1 + 75), fill=color)
    center_text(draw, ((x1 + x2) / 2, y2 - 33), kind, font(13, mono=True), MUTED)


def index_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    center_text(draw, (480, 38), "Index once. Remember everything.", font(30), TEXT)
    center_text(draw, (480, 80), "Images, long video, or a live camera feed", font(17), MUTED)

    cards = [(55, 165, 205, 315), (225, 165, 375, 315), (395, 165, 545, 315)]
    for box, kind, color in zip(cards, ("IMAGES", "VIDEO", "CAMERA"), (PINK, BLUE, MINT)):
        media_card(draw, box, kind, color, progress)

    flow = ease((progress - 0.12) / 0.48)
    for row in range(3):
        start_x = 570 + row * 22
        end_x = 684
        x = start_x + (end_x - start_x) * flow
        draw.line((545, 205 + row * 36, 690, 205 + row * 36), fill="#28405f", width=2)
        draw.ellipse((x - 5, 200 + row * 36, x + 5, 210 + row * 36), fill=MINT)

    draw.ellipse((690, 153, 890, 205), fill="#1a3652", outline=BLUE, width=3)
    draw.rectangle((690, 178, 890, 315), fill="#142b43", outline=BLUE, width=3)
    draw.ellipse((690, 288, 890, 340), fill="#142b43", outline=BLUE, width=3)
    center_text(draw, (790, 220), "VISUAL", font(19, mono=True), TEXT)
    center_text(draw, (790, 249), "MEMORY", font(19, mono=True), TEXT)
    count = int(12840 * ease((progress - 0.25) / 0.55))
    center_text(draw, (790, 281), f"{count:,} frames", font(14, mono=True), MINT)

    for x, label in (
        (116, "provenance"),
        (300, "embeddings"),
        (474, "attention"),
        (700, "entities"),
    ):
        badge(draw, (x, 405), label)
    return image


def result_thumbnail(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], variant: int
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill="#d6d3c8")
    draw.rectangle((x1 + 12, y1 + 12, x2 - 12, y2 - 30), fill="#f0ede3")
    draw.rectangle((x1 + 35, y1 + 28, x2 - 35, y2 - 47), fill="#f7faf8", outline="#9ba7a3", width=2)
    draw.line((x1 + 48, y1 + 45 + variant * 4, x2 - 48, y1 + 45), fill="#4d78a0", width=3)
    draw.line((x1 + 48, y1 + 61, x2 - 65, y1 + 67 + variant * 2), fill="#cf765a", width=3)
    draw.line((x1 + 48, y1 + 78, x2 - 85, y1 + 76), fill="#648f6b", width=3)
    draw.rectangle((x1 + 13, y2 - 28, x2 - 13, y2 - 13), fill="#2d3850")


def search_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    center_text(draw, (480, 32), "Search visual memory", font(30), TEXT)
    draw.rounded_rectangle(
        (60, 92, 900, 452), radius=18, fill="#091321", outline="#263b58", width=2
    )
    draw.ellipse((82, 111, 94, 123), fill="#ff6b6b")
    draw.ellipse((102, 111, 114, 123), fill="#ffd166")
    draw.ellipse((122, 111, 134, 123), fill="#62e6c6")

    full_query = '> search_visual_memory("whiteboard", limit=3)'
    typed = full_query[: int(len(full_query) * ease(progress / 0.34))]
    draw.text((87, 151), typed, font=font(18, mono=True), fill=MINT)
    if progress < 0.34 and int(progress * 16) % 2 == 0:
        x = 87 + draw.textlength(typed, font=font(18, mono=True))
        draw.rectangle((x + 3, 151, x + 6, 174), fill=MINT)

    reveal = ease((progress - 0.33) / 0.45)
    for index in range(3):
        local = ease((reveal - index * 0.16) / 0.65)
        y = int(218 + (1.0 - local) * 35)
        x = 86 + index * 272
        result_thumbnail(draw, (x, y, x + 236, y + 135), index)
        if local > 0:
            score = (0.97, 0.93, 0.89)[index]
            draw.text(
                (x + 12, y + 146), f"{score:.2f}  meeting.mp4", font=font(13, mono=True), fill=TEXT
            )
            draw.text(
                (x + 12, y + 166),
                f"t={(74, 312, 988)[index]}.0s",
                font=font(13, mono=True),
                fill=BLUE,
            )
    badge(draw, (699, 398), "3 matches · ranked", BLUE)
    return image


def attention_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    center_text(draw, (480, 30), "Attention that persists over time", font(30), TEXT)
    center_text(draw, (480, 70), "Focus · motion · shot changes · facial signals", font(16), MUTED)

    draw.rounded_rectangle((48, 118, 570, 417), radius=18, fill=PANEL, outline="#2b4262", width=2)
    draw.rectangle((69, 139, 549, 385), fill="#c7d5dd")
    draw.rectangle((69, 300, 549, 385), fill="#5d6e74")
    draw.rectangle((95, 166, 294, 286), fill="#f1eee2", outline="#86989e", width=3)
    for row in range(3):
        draw.line((116, 191 + row * 27, 263, 188 + row * 28), fill=(BLUE, PINK, MINT)[row], width=4)
    person_x = int(365 + 70 * math.sin(progress * math.pi * 2))
    draw.ellipse((person_x - 27, 178, person_x + 27, 232), fill="#30465e")
    draw.rounded_rectangle((person_x - 42, 226, person_x + 42, 348), 24, fill="#253a51")

    focus_x = int(210 + 185 * ease(progress))
    focus_y = int(230 + 32 * math.sin(progress * math.pi * 3))
    radius = int(24 + 7 * math.sin(progress * math.pi * 6))
    draw.ellipse(
        (focus_x - radius, focus_y - radius, focus_x + radius, focus_y + radius),
        outline=MINT,
        width=4,
    )
    draw.ellipse((focus_x - 4, focus_y - 4, focus_x + 4, focus_y + 4), fill=MINT)
    draw.text((73, 392), f"t={progress * 42:05.1f}s", font=font(13, mono=True), fill=MUTED)

    chart = (610, 140, 911, 385)
    draw.rounded_rectangle(chart, radius=16, fill=PANEL, outline="#2b4262", width=2)
    draw.text((635, 163), "ATTENTION TIMELINE", font=font(14, mono=True), fill=MUTED)
    for y in (226, 282, 338):
        draw.line((635, y, 885, y), fill="#263b58", width=1)
    points = []
    visible = max(2, int(34 * progress))
    for index in range(visible):
        x = 635 + index * 7.3
        y = 292 - 42 * math.sin(index * 0.45) - 23 * math.sin(index * 1.12)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=BLUE, width=4, joint="curve")
        x, y = points[-1]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=MINT)
    badge(draw, (633, 343), "focus: whiteboard", MINT)
    return image


def mcp_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    center_text(draw, (480, 30), "Native vision tools for any agent", font(30), TEXT)
    center_text(
        draw, (480, 69), "A small CLI + MCP surface over a modular sensing library", font(16), MUTED
    )

    tools = [
        ("index_visual_path", MINT),
        ("search_visual_memory", BLUE),
        ("read_attention_timeline", PURPLE),
        ("export_visual_frame", PINK),
    ]
    for index, (label, color) in enumerate(tools):
        x, y = 72, 130 + index * 73
        slide = int((1 - ease((progress - index * 0.08) / 0.45)) * -55)
        draw.rounded_rectangle(
            (x + slide, y, 420 + slide, y + 51), radius=13, fill=PANEL, outline=color, width=2
        )
        draw.ellipse((91 + slide, y + 17, 107 + slide, y + 33), fill=color)
        draw.text((124 + slide, y + 14), label, font=font(16, mono=True), fill=TEXT)

    for index in range(4):
        y = 155 + index * 73
        draw.line((425, y, 625, 270), fill="#28415f", width=2)
        dot_t = (progress * 1.8 + index * 0.19) % 1.0
        x = 425 + (625 - 425) * dot_t
        dot_y = y + (270 - y) * dot_t
        draw.ellipse((x - 4, dot_y - 4, x + 4, dot_y + 4), fill=tools[index][1])

    draw.rounded_rectangle((625, 145, 887, 395), radius=28, fill="#111f34", outline=BLUE, width=3)
    draw.ellipse((700, 185, 812, 297), outline=MINT, width=5)
    draw.ellipse((737, 222, 775, 260), fill=PURPLE)
    center_text(draw, (756, 322), "MULTIMODAL AGENT", font(16, mono=True), TEXT)
    center_text(draw, (756, 352), "SEE · FIND · ACT", font(14, mono=True), MUTED)
    return image


def end_scene(progress: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    reveal = ease(progress / 0.35)
    center_text(draw, (480, 120), "VAAS", font(int(62 * reveal + 1)), TEXT)
    center_text(draw, (480, 229), "SEE  ·  INDEX  ·  FIND  ·  ACT", font(25), MINT)
    center_text(draw, (480, 289), "Vision as a queryable data source", font(20), MUTED)
    draw.rounded_rectangle((287, 362, 673, 413), radius=25, fill="#14273c", outline=BLUE, width=2)
    center_text(draw, (480, 375), "CLI + MCP  /  LIGHTWEIGHT  /  MIT", font(15, mono=True), TEXT)
    return image


SCENES = [
    (0.0, 3.2, hero_scene),
    (2.7, 7.0, index_scene),
    (6.5, 11.0, search_scene),
    (10.5, 15.0, attention_scene),
    (14.5, 18.0, mcp_scene),
    (17.5, DURATION, end_scene),
]


def render_frame(time_seconds: float) -> Image.Image:
    output = base_frame().convert("RGBA")
    for start, end, renderer in SCENES:
        if start <= time_seconds <= end:
            progress = (time_seconds - start) / (end - start)
            opacity = min(ease((time_seconds - start) / 0.45), ease((end - time_seconds) / 0.45))
            scene = renderer(progress).convert("RGBA")
            scene.putalpha(int(255 * opacity))
            output = Image.alpha_composite(output, scene)
    return output.convert("RGB")


def render(output: Path, gif_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f"{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open the MP4 writer")
    for frame_number in range(round(FPS * DURATION)):
        frame = np.asarray(render_frame(frame_number / FPS))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    raw.unlink()

    gif_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-filter_complex",
            (
                "[0:v]fps=10,scale=720:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=4"
            ),
            "-loop",
            "0",
            str(gif_output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/vaas-demo.mp4"))
    parser.add_argument("--gif", type=Path, default=Path("assets/vaas-demo.gif"))
    args = parser.parse_args()
    render(args.output, args.gif)
    print(f"Rendered {args.output} and {args.gif}")


if __name__ == "__main__":
    main()
