"""보드 렌더링 → 검출/pose/캘리브레이션, 파이프라인·저장·후처리·IK 통합 테스트 (합성 데이터)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handdata.calib.hand_model import model_from_observations, points_on_table
from handdata.calib.intrinsics import calibrate
from handdata.config import Cfg, load_config
from handdata.core.board import BoardPose, CharucoTarget
from handdata.core.geometry import T_from_xyz_rpy, apply_T, inv_T, make_T, rot_angle
from handdata.core.landmarks import INDEX_TIP, THUMB_TIP
from handdata.dataio.episode import EpisodeBuffer, load_episode, save_detections, save_episode
from handdata.pipeline import FrameProcessor

from .synthetic import _RIGHT_HAND, camera, detection, hand_model, random_pose

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/default.yaml")


def render_board(target: CharucoTarget, cfgb, intr, T_cam_board, px_per_m=4000):
    W = int(cfgb.squares_x * cfgb.square_length * px_per_m)
    H = int(cfgb.squares_y * cfgb.square_length * px_per_m)
    img = target.generate_image((W, H), 0)
    S = np.diag([1 / px_per_m, 1 / px_per_m, 1.0])
    Hm = intr.K @ np.c_[T_cam_board[:3, 0], T_cam_board[:3, 1], T_cam_board[:3, 3]] @ S
    out = cv2.warpPerspective(img, Hm, (intr.width, intr.height), flags=cv2.INTER_LINEAR, borderValue=255)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def board_pose_true(rng):
    R = Rotation.from_euler("xyz", [rng.uniform(-30, 30), rng.uniform(-25, 25), rng.uniform(-20, 20)],
                            degrees=True).as_matrix()
    return make_T(R, [rng.uniform(-0.15, -0.05), rng.uniform(-0.15, -0.05), rng.uniform(0.4, 0.6)])


def test_board_pose_from_rendered_image():
    intr, target, rng = camera(), CharucoTarget(CFG.board), np.random.default_rng(0)
    for _ in range(5):
        T = board_pose_true(rng)
        bp = target.pose(render_board(target, CFG.board, intr, T), intr)
        assert bp is not None, "보드 검출 실패"
        assert np.linalg.norm(bp.T_cam_board[:3, 3] - T[:3, 3]) < 0.003    # 좌표계 원점·축 규약 확인
        assert rot_angle(bp.T_cam_board[:3, :3], T[:3, :3]) < np.deg2rad(0.5)


def test_intrinsic_calibration_recovers_K():
    intr, target, rng = camera(), CharucoTarget(CFG.board), np.random.default_rng(1)
    objs, imgs = [], []
    for _ in range(15):
        found = target.detect(render_board(target, CFG.board, intr, board_pose_true(rng)))
        if found is not None:
            objs.append(found[0]); imgs.append(found[1])
    rms, K, dist, _ = calibrate(objs, imgs, (intr.width, intr.height), cv2.CALIB_FIX_K3)
    assert rms < 0.5 and abs(K[0, 0] - intr.K[0, 0]) / intr.K[0, 0] < 0.02


def test_hand_model_board_method_end_to_end():
    """여러 프레임(카메라 위치 다름)에서 책상 위 손 → median 모델이 실제 손과 일치."""
    intr, rng = camera(), np.random.default_rng(2)
    from handdata.core.lifting import hand_points_in_palm_frame
    truth = hand_points_in_palm_frame(_RIGHT_HAND * np.array([1, 1, 0]), "right")
    obs = []
    for _ in range(20):
        T = board_pose_true(rng)
        side = np.sign(inv_T(T)[:3, 3][2])
        P_board = _RIGHT_HAND * np.array([1, 1, 0]) + np.array([0.22, 0.05, side * 0.01])
        uv = intr.project(apply_T(T, P_board)) + rng.normal(0, 1.0, (21, 2))
        obs.append(points_on_table(uv, intr, T, 0.01, "right"))
    m = model_from_observations(obs, "right", "board", 0.01)
    assert np.abs(m.all21 - truth).max() < 0.002
    assert abs(m.wrist_middle - np.linalg.norm(_RIGHT_HAND[9] - _RIGHT_HAND[0])) < 0.001


def _processor():
    intr, model = camera(), hand_model()
    return intr, model, FrameProcessor(CFG, intr, model)


def test_pipeline_base_frame_composition():
    """머리(카메라)가 움직여도 base 기준 손 pose는 같아야 함."""
    intr, model, proc = _processor()
    rng = np.random.default_rng(3)
    T_base_board = T_from_xyz_rpy(CFG.board.board_in_base.xyz, CFG.board.board_in_base.rpy_deg)
    T_base_hand_true = make_T(Rotation.from_euler("xyz", [170, 10, 30], degrees=True).as_matrix(), [0.3, 0.1, 0.25])
    outs = []
    for k in range(10):
        T_cam_board = board_pose_true(rng)                   # 카메라(머리)가 매번 다른 곳
        T_cam_hand = T_cam_board @ inv_T(T_base_board) @ T_base_hand_true
        if T_cam_hand[2, 3] < 0.1:
            continue
        r = proc.process(k * 10.0, detection(model, intr, T_cam_hand), BoardPose(T_cam_board, 20, 0.2))
        assert r.valid
        outs.append(r)
        assert np.linalg.norm(r.T_base_hand[:3, 3] - T_base_hand_true[:3, 3]) < 1e-3
        assert rot_angle(r.T_base_hand[:3, :3], T_base_hand_true[:3, :3]) < 1e-2
        pinch = 0.5 * (r.kp3d_base[THUMB_TIP] + r.kp3d_base[INDEX_TIP])
        assert np.linalg.norm(r.ee_pos - pinch) < 1e-9       # EE 원점 = pinch point
    assert len(outs) >= 5


def test_flags_and_save_load(tmp_path):
    intr, model, proc = _processor()
    buf = EpisodeBuffer()
    T_cam_hand = random_pose(np.random.default_rng(4))
    buf.add(proc.process(0.0, None, None))
    buf.add(proc.process(0.1, detection(model, intr, T_cam_hand), None))
    arr = buf.to_arrays()
    assert not arr["valid"].any() and arr["flags"][0] == 3 and arr["flags"][1] == 2
    assert not np.isnan(arr["kp3d_cam"][1]).any()          # 보드 없어도 카메라 기준 결과는 저장
    save_episode(tmp_path, arr, {"x": 1})
    back = load_episode(tmp_path)
    assert np.array_equal(back["flags"], arr["flags"])
    assert json.loads((tmp_path / "meta.json").read_text())["episode"]["n_frames"] == 2


def test_ik_reaches_fk_targets():
    from handdata.robot.ik import DLSSolver
    from handdata.robot.kinematics import MjKinematics
    kin = MjKinematics.from_config(CFG)
    solver = DLSSolver(kin, CFG.robot.ik)
    rng = np.random.default_rng(5)
    home = np.array(CFG.robot.home_q)
    ok = 0
    for _ in range(20):
        q_true = kin.clamp(home + rng.normal(0, 0.3, kin.ndof))
        pos, R = kin.fk(q_true)
        res = solver.solve(home, pos, R, restarts=4)
        ok += res.converged
    assert ok >= 17


def test_offline_stage2_and_postprocess_cli(tmp_path):
    """검출 캐시(합성) → offline --reuse-detections → postprocess CLI 끝까지."""
    intr, model = camera(), hand_model()
    calib = tmp_path / "calib"
    intr.save(calib / "intrinsics.json")
    model.save(calib / "hand_model.json")
    (calib / "gripper_range.json").write_text(json.dumps({"d_closed": 0.01, "d_open": 0.1}))
    cfg_text = (ROOT / "configs/default.yaml").read_text().replace(
        "assets/demo_arm.xml", str(ROOT / "assets/demo_arm.xml"))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text(cfg_text)

    rng = np.random.default_rng(6)
    T_cam_board = board_pose_true(rng)
    T_base_board = T_from_xyz_rpy(CFG.board.board_in_base.xyz, CFG.board.board_in_base.rpy_deg)
    ts, dets, boards = [], [], []
    for k in range(90):                                     # 3초, 30fps, 손이 원을 그림
        th = 2 * np.pi * k / 90
        T_base_hand = make_T(Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix(),
                             [0.35 + 0.05 * np.cos(th), 0.05 * np.sin(th), 0.25])
        T_cam_hand = T_cam_board @ inv_T(T_base_board) @ T_base_hand
        ts.append(k / 30)
        dets.append(None if k in (40, 41) else detection(model, intr, T_cam_hand, rng, noise_px=0.5))
        boards.append(BoardPose(T_cam_board, 20, 0.2))
    ep = tmp_path / "ep"
    save_detections(ep / "detections.npz", ts, dets, boards, {"video": "synthetic", "stride": 1, "fps": 30})

    run = lambda *args: subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    r1 = run("handdata.apps.offline", "--config", str(tmp_path / "configs/default.yaml"),
             "--video", "unused.mp4", "--out", str(ep), "--reuse-detections")
    assert r1.returncode == 0, r1.stderr
    r2 = run("handdata.apps.postprocess", "--config", str(tmp_path / "configs/default.yaml"), str(ep))
    assert r2.returncode == 0, r2.stderr
    print("\n" + r1.stdout + r2.stdout)

    raw, proc = load_episode(ep), load_episode(ep, "episode_proc.npz")
    assert raw["valid"].sum() == 88 and proc["valid"].sum() == 90      # 2프레임 결측 보간
    assert np.nanmax(np.abs(np.diff(proc["ee_pos"], axis=0))) < 0.02
