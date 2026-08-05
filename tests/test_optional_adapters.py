from pathlib import Path

import numpy as np
import pytest

from vaas import VAAS


def test_mcp_server_exposes_the_agent_surface(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from vaas.mcp_server import create_server

    server = create_server(tmp_path / "mcp.db")
    assert set(server._tool_manager._tools) == {
        "export_visual_frame",
        "index_visual_path",
        "inspect_visual_asset",
        "read_attention_timeline",
        "resolve_visual_entity",
        "search_visual_memory",
        "visual_status",
    }


def test_video_index_attention_and_export(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "motion.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for index in range(12):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.circle(frame, (10 + index * 6, 32), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    vision = VAAS(tmp_path / "video.db", face_signals=True)
    records = vision.index_video(video, sample_every=0.5)

    assert len(records) == 6
    assert len(vision.attention_timeline(video)) == 6
    assert records[0].signals["face"]["backend"] == "opencv-haar"
    assert vision.export_frame(records[-1].id, tmp_path / "frame.jpg").exists()
