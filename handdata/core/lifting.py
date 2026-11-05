"""2D keypoint → metric 3D 손 pose (PnP lifting).

1) 손바닥 5점(강체)을 '마커'처럼 사용: 미리 잰 metric 손 모델 + 2D 검출 → PnP → T_cam_hand
2) 손가락은 강체가 아니므로 PnP에서 제외하고 따로 복원
   - umeyama : MediaPipe world landmark 상대 형상을 PnP 손바닥에 similarity 정렬
   - ray_bone: umeyama 결과를 초기값으로, 각 관절을 자기 2D 광선 위에 올리고 뼈 길이(캘리브레이션 실측)를 강제
3) baseline: 레퍼런스(hand-control-robot) 방식의 손 크기 비율 깊이 (lift_size_ratio)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import (T_from_rvec_tvec, apply_T, make_T, palm_frame, rot_angle,
                       rvec_tvec_from_T, umeyama)
from .landmarks import BONES, CHAINS, MIDDLE_MCP, PALM, WRIST
from .tracker import HandDetection

# ======================================================================== 손 모델


@dataclass
class HandModel:
    """palm frame 기준 metric 손 모델."""
    side: str
    palm: np.ndarray            # (5,3) PnP용 손바닥 점, z=0 평면
    all21: np.ndarray           # (21,3) 전체 keypoint (뼈 길이 계산용)
    method: str = "board"       # board | ruler
    h_offset: float = 0.0
    n_frames: int = 0
    std_mm: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def bone_lengths(self) -> dict:
        return {(a, b): float(np.linalg.norm(self.all21[b] - self.all21[a])) for a, b in BONES}

    @property
    def wrist_middle(self) -> float:
        return float(np.linalg.norm(self.all21[MIDDLE_MCP] - self.all21[WRIST]))

    @classmethod
    def from_points(cls, P21: np.ndarray, side: str, **kw) -> "HandModel":
        """palm frame 좌표 21점 → 모델. 손바닥 5점은 IPPE를 위해 z=0 평면으로 평탄화."""
        palm = P21[PALM].copy()
        palm[:, 2] = 0.0
        return cls(side=side, palm=palm, all21=P21.copy(), **kw)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "side": self.side, "method": self.method, "h_offset": self.h_offset,
            "n_frames": self.n_frames, "std_mm": list(map(float, self.std_mm)),
            "palm": self.palm.tolist(), "all21": self.all21.tolist(),
            "wrist_middle_m": self.wrist_middle, "meta": self.meta,
        }, indent=2))

    @classmethod
    def load(cls, path) -> "HandModel":
        d = json.loads(Path(path).read_text())
        return cls(d["side"], np.array(d["palm"]), np.array(d["all21"]), d.get("method", "board"),
                   d.get("h_offset", 0.0), d.get("n_frames", 0), d.get("std_mm", []), d.get("meta", {}))


def hand_points_in_palm_frame(P21: np.ndarray, side: str) -> np.ndarray:
    o, R = palm_frame(P21[PALM], side)
    return (P21 - o) @ R


# ======================================================================== PnP


@dataclass
class LiftResult:
    T_cam_hand: np.ndarray       # palm frame (원점 wrist) pose, 카메라 기준
    kp3d_cam: np.ndarray         # (21,3) 카메라 기준 3D keypoint
    reproj_px: float             # 손바닥 5점 RMS 재투영 오차
    choice: str                  # flip 해 선택 근거: single/reproj/temporal/mp_normal
    n_solutions: int


def select_solution(Rs, errs, R_prev=None, normal_hint=None,
                    err_ratio: float = 3.0, err_margin_px: float = 1.0):
    """IPPE의 두 해 중 하나 고르기 (평면 물체 flip ambiguity).

    1) 재투영 오차가 확실히 차이나면 작은 쪽
    2) 아니면 이전 프레임 회전과 가까운 쪽 (temporal)
    3) 이전 프레임이 없으면 MediaPipe 상대 깊이로 만든 손바닥 법선과 맞는 쪽
    """
    errs = np.asarray(errs, float).reshape(-1)
    order = np.argsort(errs)
    if len(Rs) == 1:
        return int(order[0]), "single"
    b, s = int(order[0]), int(order[1])
    if errs[s] > err_ratio * max(errs[b], 1e-6) and errs[s] - errs[b] > err_margin_px:
        return b, "reproj"
    if R_prev is not None:
        return int(np.argmin([rot_angle(R_prev, R) for R in Rs])), "temporal"
    if normal_hint is not None:
        return int(np.argmax([R[:, 2] @ normal_hint for R in Rs])), "mp_normal"
    return b, "reproj"


class PalmLifter:
    def __init__(self, model: HandModel, intr: Intrinsics, pnp_method: str = "ippe",
                 refine_lm: bool = True, finger_method: str = "umeyama",
                 max_reproj_px: float = 6.0, prev_max_gap_sec: float = 0.2,
                 flip_err_ratio: float = 3.0):
        self.model, self.intr = model, intr
        self.pnp_method, self.refine_lm = pnp_method, refine_lm
        self.finger_method = finger_method
        self.max_reproj_px = max_reproj_px
        self.prev_max_gap = prev_max_gap_sec
        self.flip_err_ratio = flip_err_ratio
        self._bones = model.bone_lengths
        self.reset()

    @classmethod
    def from_config(cls, cfg, model, intr):
        L = cfg.lifting
        return cls(model, intr, L.pnp_method, L.refine_lm, L.finger_method, L.max_reproj_px,
                   L.prev_max_gap_sec, L.get("flip_err_ratio", 3.0))

    def reset(self) -> None:
        self._prev_T, self._prev_t = None, -np.inf

    # ------------------------------------------------------------ 손바닥
    def solve_palm(self, uv5: np.ndarray, t: float, normal_hint=None):
        obj = self.model.palm.astype(np.float64)
        img = np.ascontiguousarray(uv5, np.float64)
        K, dist = self.intr.K, self.intr.dist

        flags = {"ippe": cv2.SOLVEPNP_IPPE, "sqpnp": cv2.SOLVEPNP_SQPNP,
                 "iterative": cv2.SOLVEPNP_ITERATIVE}[self.pnp_method]
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=flags)
        if n == 0:
            return None
        Ts = [T_from_rvec_tvec(r, t_) for r, t_ in zip(rvecs, tvecs)]
        Ts_front = [T for T in Ts if T[2, 3] > 0]       # 카메라 앞에 있는 해만
        if not Ts_front:
            return None
        idx_front = [i for i, T in enumerate(Ts) if T[2, 3] > 0]
        errs_front = np.asarray(errs).reshape(-1)[idx_front]

        has_prev = self._prev_T is not None and (t - self._prev_t) <= self.prev_max_gap
        i, choice = select_solution([T[:3, :3] for T in Ts_front], errs_front,
                                    self._prev_T[:3, :3] if has_prev else None,
                                    normal_hint, self.flip_err_ratio)
        T = Ts_front[i]
        if self.refine_lm:
            rvec, tvec = rvec_tvec_from_T(T)
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            T = T_from_rvec_tvec(rvec, tvec)

        proj = self.intr.project(apply_T(T, obj))
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        return T, err, choice, len(Ts_front)

    # ------------------------------------------------------------ 전체
    def lift(self, det: HandDetection, t: float) -> LiftResult | None:
        hint = None
        if det.uvz is not None:
            _, R_hint = palm_frame(det.uvz[PALM], self.model.side)
            hint = R_hint[:, 2]
        out = self.solve_palm(det.uv[PALM], t, hint)
        if out is None:
            return None
        T, err, choice, n = out

        palm_cam = apply_T(T, self.model.palm)
        kp = lift_fingers_umeyama(det.world, palm_cam)
        if self.finger_method == "ray_bone":
            kp = refine_fingers_ray_bone(kp, self.intr.rays(det.uv), self._bones)

        if err <= self.max_reproj_px:          # 신뢰할 만한 프레임만 다음 flip 선택 기준으로
            self._prev_T, self._prev_t = T, t
        return LiftResult(T, kp, err, choice, n)


# ======================================================================== 손가락


def lift_fingers_umeyama(world21: np.ndarray, palm_cam: np.ndarray) -> np.ndarray:
    """MediaPipe world landmark(손 상대 형상)를 PnP 손바닥 5점에 similarity 정렬 → 21점 카메라 좌표."""
    s, R, t = umeyama(world21[PALM], palm_cam, with_scale=True)
    kp = s * world21 @ R.T + t
    kp[PALM] = palm_cam
    return kp


def refine_fingers_ray_bone(kp_init: np.ndarray, rays: np.ndarray, bones: dict) -> np.ndarray:
    """각 관절을 자기 2D 광선 위로 올리고 부모와의 거리를 실측 뼈 길이로 맞춤.

    부모 P, 자식 광선 r(단위), 뼈 길이 L:  ||λ r − P|| = L  →  λ = r·P ± sqrt((r·P)² − |P|² + L²)
    두 근 중 초기값(umeyama) 깊이에 가까운 쪽 선택. 해가 없으면 광선 위 최근접점.
    """
    kp = kp_init.copy()
    for chain in CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if b in PALM:
                continue
            P, r, L = kp[a], rays[b], bones[(a, b)]
            rp = float(r @ P)
            disc = rp * rp - float(P @ P) + L * L
            if disc < 0:
                lam = rp
            else:
                lam0 = float(r @ kp_init[b])
                sq = np.sqrt(disc)
                lam = min((rp - sq, rp + sq), key=lambda v: abs(v - lam0))
            kp[b] = lam * r
    return kp


# ======================================================================== baseline


def lift_size_ratio(det: HandDetection, intr: Intrinsics, model: HandModel) -> LiftResult:
    """레퍼런스 방식 baseline: 깊이 = f · (실제 손바닥 길이의 이미지면 투영) / (이미지상 길이).

    - 방향: MediaPipe world landmark의 palm frame (카메라 축과 정렬돼 있다고 가정)
    - 크기: world landmark를 손 모델 길이(wrist–middle MCP)로 scale
    - 위치: wrist 광선 위 깊이 Z
    """
    W = det.world * (model.wrist_middle / max(np.linalg.norm(det.world[MIDDLE_MCP] - det.world[WRIST]), 1e-6))
    segs = [(WRIST, MIDDLE_MCP), (5, 17), (WRIST, 5), (WRIST, 17)]
    xy = intr.normalized(det.uv)                     # 정규화 좌표 = 픽셀/f
    num = sum(np.linalg.norm(W[b, :2] - W[a, :2]) for a, b in segs)   # 이미지면 투영 metric 길이
    den = sum(np.linalg.norm(xy[b] - xy[a]) for a, b in segs)
    Z = num / max(den, 1e-9)
    p_wrist = np.array([xy[WRIST, 0] * Z, xy[WRIST, 1] * Z, Z])
    kp = W - W[WRIST] + p_wrist
    o, R = palm_frame(kp[PALM], model.side)
    T = make_T(R, o)
    proj = intr.project(kp[PALM])
    err = float(np.sqrt(np.mean(np.sum((proj - det.uv[PALM]) ** 2, axis=1))))
    return LiftResult(T, kp, err, "size_ratio", 1)
