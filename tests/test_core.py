import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handdata.core.board import backproject_to_plane
from handdata.core.geometry import (apply_T, inv_T, make_T, palm_frame, parse_axes, rot_angle,
                                    umeyama)
from handdata.core.landmarks import PALM
from handdata.core.lifting import (PalmLifter, hand_points_in_palm_frame, lift_size_ratio,
                                   select_solution)

from .synthetic import _RIGHT_HAND, camera, detection, hand_model, random_pose


# ---------------------------------------------------------------- geometry
def test_palm_frame_orthonormal_and_side_consistent():
    o, R = palm_frame(_RIGHT_HAND[PALM], "right")
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert R[:, 2] @ np.array([0, 0, 1.0]) > 0.99          # 합성 손의 z(손바닥 쪽)
    left = _RIGHT_HAND * np.array([1, -1, 1])               # 거울상 = 왼손
    _, RL = palm_frame(left[PALM], "left")
    assert np.isclose(np.linalg.det(RL), 1)
    assert RL[:, 2] @ np.array([0, 0, 1.0]) > 0.99          # 거울상에서도 손바닥 쪽 = +z


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(10, 3))
    R = Rotation.random(random_state=1).as_matrix()
    dst = 1.7 * src @ R.T + np.array([0.1, -0.2, 0.3])
    s, R2, t = umeyama(src, dst)
    assert np.isclose(s, 1.7) and np.allclose(R2, R) and np.allclose(t, [0.1, -0.2, 0.3])


def test_parse_axes():
    R = parse_axes(["y", "z", "x"])
    assert np.allclose(R, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        parse_axes(["x", "y", "-z"])


# ---------------------------------------------------------------- 광선-평면 교차 (체커보드 손 크기 측정)
@pytest.mark.parametrize("offset", [0.0, 0.01])
def test_backproject_to_plane_exact(offset):
    intr = camera()
    # 보드가 카메라 앞 0.5m, 비스듬히
    T_cam_board = make_T(Rotation.from_euler("xyz", [160, 10, 5], degrees=True).as_matrix(), [-0.1, -0.05, 0.5])
    C = inv_T(T_cam_board)[:3, 3]
    side = np.sign(C[2])
    P_board = np.c_[np.random.default_rng(0).uniform(0, 0.2, (21, 2)), np.full(21, side * offset)]
    uv = intr.project(apply_T(T_cam_board, P_board))
    P_rec = backproject_to_plane(uv, intr, T_cam_board, offset)
    assert np.abs(P_rec - P_board).max() < 1e-6


def test_hand_model_from_table_is_metric():
    """손을 보드 옆 책상에 펴 놓은 상황 → 역투영으로 만든 모델 = 실제 손."""
    intr = camera()
    T_cam_board = make_T(Rotation.from_euler("xyz", [180 - 25, 0, 0], degrees=True).as_matrix(), [-0.05, -0.1, 0.45])
    side = np.sign(inv_T(T_cam_board)[:3, 3][2])
    hand_on_table = _RIGHT_HAND * np.array([1, 1, 0]) + np.array([0.2, 0.05, side * 0.01])
    uv = intr.project(apply_T(T_cam_board, hand_on_table))
    P = backproject_to_plane(uv, intr, T_cam_board, 0.01)
    M = hand_points_in_palm_frame(P, "right")
    ref = hand_points_in_palm_frame(hand_on_table, "right")
    assert np.abs(M - ref).max() < 1e-6


# ---------------------------------------------------------------- PnP lifting
def test_lifting_exact_without_noise():
    intr, model, rng = camera(), hand_model(), np.random.default_rng(1)
    lifter = PalmLifter(model, intr)
    for k in range(30):
        T_true = random_pose(rng)
        det = detection(model, intr, T_true)
        lifter.reset()
        res = lifter.lift(det, t=float(k))
        assert res is not None
        assert np.linalg.norm(res.T_cam_hand[:3, 3] - T_true[:3, 3]) < 1e-4
        assert rot_angle(res.T_cam_hand[:3, :3], T_true[:3, :3]) < 1e-3


def test_lifting_with_pixel_noise_is_reasonable():
    """2px 노이즈, 0.35~0.6m 거리: 위치 오차가 cm 이하 수준인지."""
    intr, model, rng = camera(), hand_model(), np.random.default_rng(2)
    lifter = PalmLifter(model, intr)
    pos_err, wrong_flip = [], 0
    for k in range(200):
        T_true = random_pose(rng)
        res = lifter.lift(detection(model, intr, T_true, rng, noise_px=2.0), t=k * 10.0)  # prev 안 씀
        pos_err.append(np.linalg.norm(res.T_cam_hand[:3, 3] - T_true[:3, 3]))
        wrong_flip += rot_angle(res.T_cam_hand[:3, :3], T_true[:3, :3]) > np.deg2rad(20)
    print(f"\n[2px] median pos err {np.median(pos_err) * 1000:.1f} mm, p90 {np.percentile(pos_err, 90) * 1000:.1f} mm, flip 오선택 {wrong_flip}/200")
    assert np.median(pos_err) < 0.01
    assert wrong_flip < 20


def test_fingers_umeyama_and_ray_bone():
    intr, model, rng = camera(), hand_model(), np.random.default_rng(3)
    errs = {}
    for method in ["umeyama", "ray_bone"]:
        lifter = PalmLifter(model, intr, finger_method=method)
        e = []
        for k in range(100):
            T_true = random_pose(rng)
            det = detection(model, intr, T_true, rng, noise_px=1.0, world_noise=0.004)
            res = lifter.lift(det, t=k * 10.0)
            P_true = model.all21 @ T_true[:3, :3].T + T_true[:3, 3]
            e.append(np.linalg.norm(res.kp3d_cam[[4, 8]] - P_true[[4, 8]], axis=1).mean())
        errs[method] = np.median(e)
    print(f"\nfingertip median err: umeyama {errs['umeyama'] * 1000:.1f} mm, ray_bone {errs['ray_bone'] * 1000:.1f} mm")
    assert errs["umeyama"] < 0.02 and errs["ray_bone"] < 0.02


def test_select_solution_temporal_and_hint():
    R1 = np.eye(3)
    R2 = Rotation.from_rotvec([0, np.pi * 0.8, 0]).as_matrix()
    assert select_solution([R1, R2], [0.5, 0.6], R_prev=R2)[0] == 1
    assert select_solution([R1, R2], [0.5, 0.6], normal_hint=R2[:, 2])[0] == 1
    assert select_solution([R1, R2], [0.2, 5.0], R_prev=R2)[0] == 0      # 오차 차이가 확실하면 reproj 우선


def test_size_ratio_baseline_runs():
    intr, model, rng = camera(), hand_model(), np.random.default_rng(4)
    T_true = random_pose(rng)
    res = lift_size_ratio(detection(model, intr, T_true), intr, model)
    assert abs(res.T_cam_hand[2, 3] - T_true[2, 3]) < 0.1
