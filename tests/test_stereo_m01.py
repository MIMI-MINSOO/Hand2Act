"""Deterministic M0/M1 tests; no physical camera is required."""

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from handdata.apps.realsense_probe import choose_common_profile, summarize_samples
from handdata.calib.intrinsics import require_verified_board, validate_factory_intrinsics
from handdata.config import Cfg, load_config
from handdata.core.board import CharucoTarget
from handdata.core.camera import Intrinsics, intrinsics_from_realsense


def _rs_intr(model="Brown Conrady"):
    return SimpleNamespace(width=1280, height=720, fx=900.0, fy=901.0,
                           ppx=640.0, ppy=360.0, coeffs=[.1, -.02, .001, .002, .003],
                           model=model)


def test_factory_intrinsics_conversion_and_roundtrip(tmp_path):
    intr = intrinsics_from_realsense(_rs_intr(), serial="ABC", fps=30)
    assert intr.rms != intr.rms  # factory RMS is deliberately unknown
    assert intr.meta["serial"] == "ABC"
    path = tmp_path / "factory.json"
    intr.save(path)
    loaded = Intrinsics.load(path)
    np.testing.assert_allclose(loaded.K, intr.K)
    np.testing.assert_allclose(loaded.dist, intr.dist)
    assert loaded.meta == intr.meta


@pytest.mark.parametrize("model", ["Modified Brown Conrady", "Inverse Brown Conrady",
                                    "F-Theta", "Kannala Brandt4"])
def test_factory_intrinsics_rejects_non_opencv_models(model):
    with pytest.raises(ValueError, match="unsupported RealSense distortion"):
        intrinsics_from_realsense(_rs_intr(model), serial="ABC", fps=30)


def test_zero_coefficient_inverse_brown_is_no_distortion_equivalent():
    raw = _rs_intr("distortion.inverse_brown_conrady")
    raw.coeffs = [0.0] * 5
    intr = intrinsics_from_realsense(raw, serial="ABC", fps=30)
    np.testing.assert_array_equal(intr.dist, np.zeros(5))
    assert intr.meta["realsense_distortion_model"] == "distortion.inverse_brown_conrady"
    assert intr.meta["opencv_distortion_model"] == "none"
    assert intr.meta["distortion_conversion"] == "zero_coefficients_equivalent_to_none"


def test_stereo_config_blocks_unverified_physical_board():
    cfg = load_config("configs/stereo.example.yaml")
    cfg.board["physical_verified"] = False
    with pytest.raises(ValueError, match="physical ChArUco board is not verified"):
        require_verified_board(cfg.board)


def test_stereo_config_matches_authoritative_board_txt():
    values = {}
    for line in Path("configs/charuco_A4_5x7_40mm.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    board = load_config("configs/stereo.example.yaml").board
    assert (board.squares_x, board.squares_y) == (int(values["squaresX"]),
                                                  int(values["squaresY"]))
    assert board.dictionary == values["arucoDictionary"]
    assert board.square_length == float(values["squareLength"])
    assert board.marker_length == float(values["markerLength"])
    assert board.reference.board_width_m == float(values["boardWidth_mm"]) / 1000
    assert board.reference.board_height_m == float(values["boardHeight_mm"]) / 1000
    assert board.physical_verified is False


def test_verified_board_uses_measured_lengths_over_reference():
    board = Cfg(dictionary="DICT_6X6_250", squares_x=5, squares_y=7,
                square_length=.040, marker_length=.024, legacy_pattern=False,
                physical_verified=True,
                reference=Cfg(board_width_m=.200, board_height_m=.280),
                verification=Cfg(dictionary_verified=True, layout_verified=True,
                                 legacy_pattern_verified=True,
                                 print_scale_verified=True, flatness_verified=True,
                                 measured_board_width_m=.195,
                                 measured_board_height_m=.273,
                                 measured_square_length_m=.039,
                                 measured_marker_length_m=.0235))
    spec = require_verified_board(board)
    assert spec["square_length_m"] == .039
    assert spec["reference_match"]["square_length"] is False
    assert board.square_length == .039 and board.marker_length == .0235


def test_verified_board_rejects_internally_inconsistent_measurements():
    cfg = load_config("configs/stereo.example.yaml")
    cfg.board["physical_verified"] = True
    cfg.board.verification.update(
        dictionary_verified=True, layout_verified=True, legacy_pattern_verified=True,
        print_scale_verified=True, flatness_verified=True,
        measured_board_width_m=.200, measured_board_height_m=.270,
        measured_square_length_m=.040, measured_marker_length_m=.024)
    with pytest.raises(ValueError, match="height conflicts"):
        require_verified_board(cfg.board)


def test_factory_validation_reports_corner_statistics():
    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    intr = Intrinsics(K, np.zeros(5), 1280, 720)
    obj = np.array([[x, y, 0.] for y in np.linspace(0, .16, 5)
                    for x in np.linspace(0, .12, 4)], dtype=float)
    views, images = [], []
    rng = np.random.default_rng(7)
    for i in range(8):
        rvec = np.array([.05 * i, -.03 * i, .01 * i])
        tvec = np.array([-.06, -.08, .6 + .02 * i])
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
        views.append(obj.copy())
        images.append(uv.reshape(-1, 2) + rng.normal(0, .15, (len(obj), 2)))
    report = validate_factory_intrinsics(views, images, intr)
    assert report["n_views"] == 8 and report["n_corners"] == 160
    assert report["corner_median_px"] < .3 and report["corner_p95_px"] < .5


def test_charuco_detection_keeps_old_api_and_exposes_ids():
    target = CharucoTarget(load_config("configs/stereo.example.yaml").board)
    image = target.generate_image((500, 700), 20)
    old = target.detect(image)
    with_ids = target.detect_with_ids(image)
    assert old is not None and with_ids is not None
    np.testing.assert_allclose(old[0], with_ids[0])
    np.testing.assert_allclose(old[1], with_ids[1])
    assert len(with_ids[2]) == len(with_ids[0])


def test_common_profile_and_timestamp_statistics():
    devices = [
        {"color_profiles": [(640, 480, 30, "RGB8"), (1280, 720, 30, "RGB8")]},
        {"color_profiles": [(1280, 720, 30, "YUYV"), (1920, 1080, 30, "RGB8")]},
    ]
    assert choose_common_profile(devices) == (1280, 720, 30)
    rows = [{"host_received_ns": i * 33_000_000, "frame_number": n,
             "timestamp_domain": "hardware_clock", "sensor_timestamp_ms": i * 33.0}
            for i, n in enumerate([10, 11, 13])]
    stats = summarize_samples(rows)
    assert stats["frame_number_drops"] == 1
    assert stats["timestamp_domains"] == ["hardware_clock"]
