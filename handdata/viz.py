"""디버그 오버레이: 손 skeleton, palm/보드 좌표축, HUD."""

from __future__ import annotations

import cv2
import numpy as np

from .core.camera import Intrinsics
from .core.geometry import rvec_tvec_from_T
from .core.landmarks import CONNECTIONS, PALM
from .pipeline import FLAG_NAMES, FrameResult


def draw_axes(img, intr: Intrinsics, T: np.ndarray, length: float = 0.05, thickness: int = 2):
    if T is None or np.any(np.isnan(T)):
        return
    rvec, tvec = rvec_tvec_from_T(T)
    cv2.drawFrameAxes(img, intr.K, intr.dist, rvec, tvec, length, thickness)


def draw_frame(img, intr: Intrinsics, r: FrameResult, model_palm: np.ndarray | None = None,
               hud: bool = True):
    out = img
    if not np.any(np.isnan(r.uv)):
        pts = r.uv.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(out, tuple(p), 4 if i in PALM else 3, (0, 0, 255) if i in PALM else (255, 255, 255), -1)
    draw_axes(out, intr, r.T_cam_board, 0.08, 3)
    draw_axes(out, intr, r.T_cam_hand, 0.05, 2)
    if model_palm is not None and not np.any(np.isnan(r.T_cam_hand)):
        # 손 모델을 재투영 → 검출점(빨강)과 겹쳐야 정상
        P = model_palm @ r.T_cam_hand[:3, :3].T + r.T_cam_hand[:3, 3]
        for p in intr.project(P).astype(int):
            cv2.circle(out, tuple(p), 7, (0, 255, 0), 1)
    if not hud:
        return out

    flags = [n for b, n in FLAG_NAMES.items() if r.flags & b]
    lines = [f"t={r.t:6.2f}s  {'VALID' if r.valid else 'INVALID'}  {' '.join(flags)}",
             f"reproj={r.reproj_px:5.2f}px  flip={r.choice}  grip={r.gripper:4.2f}  d={r.pinch_dist * 100:4.1f}cm"]
    if r.valid:
        p = r.ee_pos
        lines.append(f"EE[base] x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f} m")
    put_lines(out, lines, (80, 255, 80) if r.valid else (80, 80, 255))
    return out


def put_lines(img, lines, color=(80, 255, 80), y0=28, scale=0.7):
    for k, s in enumerate(lines):
        y = y0 + int(36 * scale) * k
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
