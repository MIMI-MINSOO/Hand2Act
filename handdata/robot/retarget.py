"""손 3D → 로봇 EE pose(위치·자세) + 그리퍼 개폐.

레퍼런스와 달리 상대 매핑/clutch 없이 로봇 base 기준 절대 pose를 그대로 만든다.
로봇이 바뀌면 configs의 retarget 항목(축 대응, TCP offset)만 바꾸면 됨.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.geometry import make_T, normalize, parse_axes
from ..core.landmarks import INDEX_TIP, MIDDLE_MCP, PALM, THUMB_TIP, WRIST


def load_gripper_range(path, default_closed=0.015, default_open=0.09):
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text())
        return float(d["d_closed"]), float(d["d_open"]), True
    return default_closed, default_open, False


class Retargeter:
    def __init__(self, cfg, d_closed: float, d_open: float):
        self.origin = cfg.origin                     # pinch | wrist | palm_center
        self.orientation = cfg.orientation           # palm | grasp
        self.R_palm_tool = parse_axes(cfg.tool_axes_in_palm)
        self.R_adjust = Rotation.from_euler("xyz", cfg.extra_rpy_deg, degrees=True).as_matrix()
        self.tcp_offset = np.asarray(cfg.tcp_offset, float)   # tool frame 기준 추가 offset
        self.grasp_min_dist = float(cfg.get("grasp_min_dist", 0.025))
        self.d_closed, self.d_open = d_closed, d_open

    # ------------------------------------------------------------ 위치
    def _origin(self, kp: np.ndarray) -> np.ndarray:
        if self.origin == "pinch":
            return 0.5 * (kp[THUMB_TIP] + kp[INDEX_TIP])       # 엄지 끝–검지 끝 중점 = 그리퍼 TCP 대응
        if self.origin == "palm_center":
            return kp[PALM].mean(0)
        return kp[WRIST].copy()

    # ------------------------------------------------------------ 자세
    def _rotation(self, R_base_hand: np.ndarray, kp: np.ndarray) -> np.ndarray:
        R_tool = R_base_hand @ self.R_palm_tool
        if self.orientation == "grasp":
            # z = 접근 방향(wrist→middle MCP), y = 닫히는 방향(엄지→검지). 집게가 거의 닫히면 palm 기반 y 사용
            z = normalize(kp[MIDDLE_MCP] - kp[WRIST])
            d = kp[INDEX_TIP] - kp[THUMB_TIP]
            y_raw = d if np.linalg.norm(d) >= self.grasp_min_dist else R_tool[:, 1]
            y = normalize(y_raw - (y_raw @ z) * z)
            R_tool = np.stack([np.cross(y, z), y, z], axis=1)
        return R_tool @ self.R_adjust

    def ee_pose(self, T_base_hand: np.ndarray, kp_base: np.ndarray) -> np.ndarray:
        R = self._rotation(T_base_hand[:3, :3], kp_base)
        return make_T(R, self._origin(kp_base) + R @ self.tcp_offset)

    # ------------------------------------------------------------ 그리퍼
    def gripper(self, kp: np.ndarray):
        """→ (g ∈ [0,1], 1=열림, d[m] 엄지–검지 거리)."""
        d = float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]))
        g = (d - self.d_closed) / max(self.d_open - self.d_closed, 1e-6)
        return float(np.clip(g, 0.0, 1.0)), d
