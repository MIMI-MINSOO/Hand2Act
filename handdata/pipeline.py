"""프레임 단위 처리: (손 검출, 보드 pose) → 로봇 base 기준 EE pose + 그리퍼.

이미지/MediaPipe와 분리돼 있어 live·offline·테스트(합성 데이터)에서 똑같이 사용.

    T_base_hand = T_base_board · T_cam_board⁻¹ · T_cam_hand
    T_base_ee   = retarget(T_base_hand, kp3d_base)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.board import BoardPose
from .core.camera import Intrinsics
from .core.geometry import T_from_xyz_rpy, apply_T, inv_T, quat_from_R
from .core.lifting import HandModel, PalmLifter, lift_size_ratio
from .core.tracker import HandDetection
from .robot.retarget import Retargeter, load_gripper_range

# 프레임 상태 bitmask
NO_HAND, NO_BOARD, PNP_FAIL, HIGH_REPROJ, INTERPOLATED, IK_FAIL, SIDE_MISMATCH = 1, 2, 4, 8, 16, 32, 64
FLAG_NAMES = {NO_HAND: "no_hand", NO_BOARD: "no_board", PNP_FAIL: "pnp_fail",
              HIGH_REPROJ: "high_reproj", INTERPOLATED: "interpolated", IK_FAIL: "ik_fail",
              SIDE_MISMATCH: "side_mismatch"}
INVALID_MASK = NO_HAND | NO_BOARD | PNP_FAIL | HIGH_REPROJ   # 이 중 하나라도 있으면 valid=False

_NAN44 = np.full((4, 4), np.nan)
_NAN21 = np.full((21, 3), np.nan)


@dataclass
class FrameResult:
    t: float
    flags: int = 0
    reproj_px: float = np.nan
    choice: str = ""
    gripper: float = np.nan
    pinch_dist: float = np.nan
    uv: np.ndarray = field(default_factory=lambda: np.full((21, 2), np.nan))
    T_cam_hand: np.ndarray = field(default_factory=lambda: _NAN44.copy())
    kp3d_cam: np.ndarray = field(default_factory=lambda: _NAN21.copy())
    T_cam_board: np.ndarray = field(default_factory=lambda: _NAN44.copy())
    T_base_hand: np.ndarray = field(default_factory=lambda: _NAN44.copy())
    kp3d_base: np.ndarray = field(default_factory=lambda: _NAN21.copy())
    T_base_ee: np.ndarray = field(default_factory=lambda: _NAN44.copy())

    @property
    def valid(self) -> bool:
        return not (self.flags & INVALID_MASK)

    @property
    def ee_pos(self):
        return self.T_base_ee[:3, 3]

    @property
    def ee_quat(self):
        return quat_from_R(self.T_base_ee[:3, :3]) if self.valid else np.full(4, np.nan)


class FrameProcessor:
    def __init__(self, cfg, intr: Intrinsics, model: HandModel, lift_method: str | None = None):
        self.cfg, self.intr, self.model = cfg, intr, model
        self.lifter = PalmLifter.from_config(cfg, model, intr)
        self.lift_method = lift_method or cfg.lifting.get("method", "pnp")   # pnp | size_ratio
        bib = cfg.board.board_in_base
        self.T_base_board = T_from_xyz_rpy(bib.xyz, bib.rpy_deg)
        d_closed, d_open, self.gripper_calibrated = load_gripper_range(
            cfg.resolve(cfg.paths.gripper_range), *cfg.gripper.default_range)
        self.retarget = Retargeter(cfg.retarget, d_closed, d_open)
        self.max_reproj = float(cfg.lifting.max_reproj_px)

    def reset(self):
        self.lifter.reset()

    def lift(self, det: HandDetection, t: float):
        if self.lift_method == "size_ratio":
            return lift_size_ratio(det, self.intr, self.model)
        return self.lifter.lift(det, t)

    def process(self, t: float, det: HandDetection | None, board: BoardPose | None) -> FrameResult:
        r = FrameResult(t)
        if board is None:
            r.flags |= NO_BOARD
        else:
            r.T_cam_board = board.T_cam_board
        if det is None:
            r.flags |= NO_HAND
            return r
        r.uv = det.uv
        if not det.side_matched:
            r.flags |= SIDE_MISMATCH

        lift = self.lift(det, t)
        if lift is None:
            r.flags |= PNP_FAIL
            return r
        r.T_cam_hand, r.kp3d_cam, r.reproj_px, r.choice = lift.T_cam_hand, lift.kp3d_cam, lift.reproj_px, lift.choice
        if lift.reproj_px > self.max_reproj:
            r.flags |= HIGH_REPROJ
        r.gripper, r.pinch_dist = self.retarget.gripper(lift.kp3d_cam)   # 거리라 좌표계 무관

        if board is not None:
            T_base_cam = self.T_base_board @ inv_T(board.T_cam_board)
            r.T_base_hand = T_base_cam @ lift.T_cam_hand
            r.kp3d_base = apply_T(T_base_cam, lift.kp3d_cam)
            r.T_base_ee = self.retarget.ee_pose(r.T_base_hand, r.kp3d_base)
        return r
