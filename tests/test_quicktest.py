"""quicktest 경로: 근사 K + 자 방식 즉석 손 모델 + 보드 없이 카메라 기준 EE."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from handdata.apps.quicktest import approx_intrinsics
from handdata.calib.hand_model import model_from_observations
from handdata.config import load_config
from handdata.core.landmarks import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST
from handdata.core.lifting import hand_points_in_palm_frame
from handdata.pipeline import NO_BOARD, FrameProcessor

from .synthetic import camera, detection, hand_model, random_pose

ROOT = Path(__file__).resolve().parents[1]


def test_approx_intrinsics():
    K = approx_intrinsics(1280, 720, 90.0).K
    assert np.isclose(K[0, 0], 640.0) and np.isclose(K[0, 2], 640) and np.isclose(K[1, 2], 360)


def test_ruler_model_then_camera_frame_ee():
    intr, true_model, rng = camera(1280, 720, 1000.0), hand_model(), np.random.default_rng(0)
    L = float(np.linalg.norm(true_model.all21[MIDDLE_MCP] - true_model.all21[WRIST]))
    # 즉석 손 모델: MediaPipe world landmark(크기 틀림) 30프레임 + 자로 잰 길이
    obs = [hand_points_in_palm_frame(detection(true_model, intr, random_pose(rng)).world, "right")
           for _ in range(30)]
    m = model_from_observations(obs, "right", "ruler", 0.0, L)
    assert abs(m.wrist_middle - L) < 1e-9

    cfg = load_config(ROOT / "configs/default.yaml")
    proc = FrameProcessor(cfg, intr, m)
    T = random_pose(rng)
    r = proc.process(0.0, detection(true_model, intr, T), None)
    assert r.flags == NO_BOARD                                # 보드 없어도 카메라 기준 결과는 나옴
    T_cam_ee = proc.retarget.ee_pose(r.T_cam_hand, r.kp3d_cam)
    P_true = true_model.all21 @ T[:3, :3].T + T[:3, 3]
    pinch_true = 0.5 * (P_true[THUMB_TIP] + P_true[INDEX_TIP])
    assert np.linalg.norm(T_cam_ee[:3, 3] - pinch_true) < 0.01


def test_quicktest_runs_headless(tmp_path):
    """MediaPipe 실제 호출 포함, 손 없는 영상으로 크래시 없이 도는지."""
    model = ROOT / "assets/hand_landmarker.task"
    if not model.exists():
        import pytest
        pytest.skip("MediaPipe 모델 없음")
    vid = tmp_path / "blank.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(15):
        vw.write(np.full((480, 640, 3), 120, np.uint8))
    vw.release()
    r = subprocess.run([sys.executable, "-m", "handdata.apps.quicktest", "--video", str(vid),
                        "--no-window", "--max-frames", "15"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "근사 K" in r.stdout
