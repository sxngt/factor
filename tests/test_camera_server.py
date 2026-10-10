"""Camera server shared files: the readers in the package accept what the server writes."""

import json
import time

import numpy as np
import pytest

from factor.real.camera_server import load_intrinsics, read_meta, repeats_for_gap, server_alive


def test_repeats_for_gap():
    assert repeats_for_gap(0.0, 10.0) == 1
    assert repeats_for_gap(0.1, 10.0) == 1
    assert repeats_for_gap(0.37, 10.0) == 4
    assert repeats_for_gap(5.0, 10.0) == 5


def test_load_intrinsics(tmp_path):
    path = tmp_path / "cam.json"
    path.write_text(json.dumps({"width": 640, "height": 480, "K": [[500, 0, 320], [0, 501, 240], [0, 0, 1]]}))
    assert load_intrinsics(str(path), 640, 480) == [500.0, 501.0, 320.0, 240.0]
    assert load_intrinsics(str(path), 1280, 720) is None
    path.write_text(json.dumps({"width": 8, "height": 6, "fx": 1, "fy": 2, "cx": 3, "cy": 4}))
    assert load_intrinsics(str(path), 8, 6) == [1.0, 2.0, 3.0, 4.0]
    assert load_intrinsics("", 8, 6) is None


def test_alive_without_server(tmp_path):
    assert read_meta(str(tmp_path)) is None
    assert not server_alive(str(tmp_path))


def test_publish_round_trip_with_package_readers(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from factor.calib.run import read_frame
    from factor.real.camera_server import publish_frame
    from factor.real.cli import shared_frame_source

    rgb = np.zeros((24, 32, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[5:10, 3:8] = (10, 220, 30)
    t_before = time.time() - 1.0
    meta = publish_frame(
        rgb, 7, str(tmp_path), intrinsics=[1.0, 2.0, 3.0, 4.0], backend="webcam", mode="test"
    )

    on_disk = read_meta(str(tmp_path))
    assert on_disk == meta
    for key in ("t", "n", "w", "h", "color_sha256", "intrinsics", "backend", "mode"):
        assert key in on_disk
    assert (on_disk["w"], on_disk["h"], on_disk["n"]) == (32, 24, 7)
    assert server_alive(str(tmp_path))

    assert np.array_equal(shared_frame_source(str(tmp_path))(), rgb)
    frame, t, n = read_frame(str(tmp_path), newer_than=t_before)
    assert np.array_equal(frame, rgb) and n == 7 and t == pytest.approx(meta["t"])
