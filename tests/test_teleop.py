"""WeGo NERO 양팔 MJCF + 상대 매핑 teleop (뷰어/카메라 없이)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handdata.config import load_config
from handdata.core.geometry import make_T, parse_axes, rot_angle

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/nero_dual.yaml"
pytestmark = pytest.mark.skipif(not (ROOT / "assets/nero/nero_dual.xml").exists(),
                                reason="scripts/build_wego_dual.py 로 모델 생성 필요")


def test_config_extends_default():
    cfg, base = load_config(CFG_PATH), load_config(ROOT / "configs/default.yaml")
    assert cfg.robot.ee_site == "right_tcp" and cfg.lifting == base.lifting       # 나머지는 상속
    assert cfg.robot.ik.max_iters == 40 and cfg.robot.ik.ori_weight == base.robot.ik.ori_weight


def test_home_pose_reaches_design_target():
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    table = 0.75
    for side, y in [("right", -0.20), ("left", 0.20)]:
        arm = ArmTeleop(cfg, model, side, cfg.arms[side])
        p, R = arm.kin.fk(arm.q)
        assert np.allclose(p, [0.40, y, table + 0.15], atol=0.003)
        assert np.degrees(np.arccos(-R[2, 2])) < 1.5           # 접근축(z) 수직 아래
        assert abs(R[1, 1]) > 0.99                             # 손가락 열림축 = 좌우(y)


def test_gripper_opens_both_fingers_symmetrically():
    """공식 모델은 두 손가락 축이 반대 (0~0.05 / -0.05~0) → g 하나로 둘 다 바깥으로 열려야 함."""
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    data = mujoco.MjData(model)
    arm = ArmTeleop(cfg, model, "right", cfg.arms["right"])
    assert np.allclose(arm.g_open, [0.05, -0.05])
    gap = []
    for g in (0.0, 1.0):
        arm.g = g
        arm.write(data, physics=False)
        mujoco.mj_kinematics(model, data)
        a, b = (data.xpos[model.body(f"right_gripper_link{i}").id] for i in (1, 2))
        gap.append(np.linalg.norm(a - b))
    assert gap[1] - gap[0] == pytest.approx(0.10, abs=0.002)   # 손가락 하나당 5cm → 총 10cm


def test_relative_mapping_moves_ee_by_hand_delta():
    """손이 카메라 쪽(-z_cam)으로 5cm, 위(-y_cam)로 3cm → 오른팔 EE가 +x 5cm, +z 3cm. 자세도 따라 회전."""
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH, ["filter_live.pos_min_cutoff=1000", "filter_live.rot_min_cutoff=1000"])
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    arm = ArmTeleop(cfg, model, "right", cfg.arms["right"])
    R_bc = parse_axes(cfg.teleop.cam_axes_in_base)
    p0, R0 = arm.kin.fk(arm.q)

    T_h0 = make_T(Rotation.from_euler("xyz", [170, 20, 5], degrees=True).as_matrix(), [0.05, 0.0, 0.5])
    arm.update(0.0, T_h0, 1.0, R_bc, cfg.teleop)                     # anchor
    dR = Rotation.from_euler("y", 10, degrees=True).as_matrix()        # 카메라 y축(=base -z) 기준 10°
    for k in range(1, 30):                                             # 천천히 이동 (IK warm start)
        a = k / 29
        T = make_T(Rotation.from_rotvec(a * Rotation.from_matrix(dR).as_rotvec()).as_matrix() @ T_h0[:3, :3],
                   T_h0[:3, 3] + a * np.array([0, -0.03, -0.05]))
        arm.update(k / 30, T, 0.2, R_bc, cfg.teleop)
    p, R = arm.kin.fk(arm.q)
    assert np.allclose(p - p0, [0.05, 0, 0.03], atol=0.004), p - p0
    R_expect = R_bc @ dR @ R_bc.T @ R0
    assert rot_angle(R, R_expect) < np.radians(2.5)
    assert abs(arm.g - 0.2) < 1e-6


def test_hold_gap_reanchors_without_jump():
    """손이 hold_sec 이상 사라졌다가 다른 곳에서 나타나도 로봇은 튀지 않음 (clutch)."""
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    arm = ArmTeleop(cfg, model, "right", cfg.arms["right"])
    R_bc = parse_axes(cfg.teleop.cam_axes_in_base)
    T = make_T(np.diag([1.0, -1, -1]), [0, 0, 0.5])
    arm.update(0.0, T, 1.0, R_bc, cfg.teleop)
    q_before = arm.q.copy()
    T2 = make_T(np.diag([1.0, -1, -1]), [0.2, 0.1, 0.7])                 # 손을 내렸다가 먼 곳에서 다시
    arm.update(1.0, T2, 1.0, R_bc, cfg.teleop)
    assert np.abs(arm.q - q_before).max() < 1e-3


def test_physics_hold_home():
    """물리 모드: 액추에이터 + 중력보상으로 홈 자세 유지."""
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    data = mujoco.MjData(model)
    arms = [ArmTeleop(cfg, model, s, cfg.arms[s]) for s in ("right", "left")]
    for arm in arms:
        data.qpos[arm.kin.qpos_adr] = arm.q
        arm.write(data, physics=True)
    for _ in range(1000):                                                # 2초
        mujoco.mj_step(model, data)
    for arm in arms:
        assert np.abs(data.qpos[arm.kin.qpos_adr] - arm.q).max() < 0.02


def test_flip_jump_rejected_then_accepted_if_persistent():
    """한 프레임 180° 뒤집힘(PnP flip) → 버림. 계속 유지되면 진짜 자세로 보고 재-anchor."""
    from handdata.apps.teleop import ArmTeleop
    cfg = load_config(CFG_PATH)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    arm = ArmTeleop(cfg, model, "right", cfg.arms["right"])
    R_bc = parse_axes(cfg.teleop.cam_axes_in_base)
    T = make_T(np.diag([1.0, -1, -1]), [0, 0, 0.5])
    assert arm.update(0.0, T, 1.0, R_bc, cfg.teleop) == "anchor"
    q0 = arm.q.copy()
    T_flip = make_T(np.diag([-1.0, 1, -1]) @ T[:3, :3], T[:3, 3])
    assert arm.update(0.033, T_flip, 1.0, R_bc, cfg.teleop) == "jump"
    assert np.allclose(arm.q, q0)
    assert arm.update(0.066, T, 1.0, R_bc, cfg.teleop) == "ok"          # 원래 자세로 복귀하면 정상
    sts = [arm.update(0.1 + 0.033 * k, T_flip, 1.0, R_bc, cfg.teleop) for k in range(8)]
    assert "anchor" in sts and sts[-1] == "ok"
    assert np.abs(arm.q - q0).max() < 1e-2                              # 재-anchor라 로봇은 안 튐


def test_assign_hands_follows_position_not_label():
    from handdata.apps.teleop import assign_hands
    from handdata.core.tracker import HandDetection
    mk = lambda x, side: HandDetection(np.tile([x, 300.0], (21, 1)), None, None, side, 0.9, True)
    sides = ["right", "left"]
    first = assign_hands([mk(300, "right"), mk(900, "left")], sides, {}, 0.0, 0.4, False)
    assert first["right"].uv[0, 0] == 300                                # 미러 아님: 오른손 = 화면 왼쪽
    prev = {s: (d.uv[0], 0.0) for s, d in first.items()}
    swapped = assign_hands([mk(310, "left"), mk(890, "right")], sides, prev, 0.03, 0.4, False)
    assert swapped["right"].uv[0, 0] == 310 and swapped["left"].uv[0, 0] == 890   # 라벨 뒤바뀜 무시
    one = assign_hands([mk(905, "right")], sides, prev, 0.03, 0.4, False)
    assert list(one) == ["left"]
