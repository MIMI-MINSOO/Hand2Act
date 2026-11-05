"""SE(3)·회전 유틸, palm frame 정의, Umeyama 정렬.

규약
- 4x4 동차변환 T_a_b: b 좌표계의 점을 a 좌표계로 옮김 (p_a = T_a_b · p_b)
- 회전행렬의 열 = 해당 좌표계 축을 부모 좌표계에서 표현한 것
- quaternion은 [x, y, z, w] (scipy 규약)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------- SE(3)


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(N,3) 점들에 T 적용."""
    P = np.asarray(P, float)
    return P @ T[:3, :3].T + T[:3, 3]


def T_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return make_T(R, tvec)


def rvec_tvec_from_T(T: np.ndarray):
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))
    return rvec.reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def T_from_xyz_rpy(xyz, rpy_deg) -> np.ndarray:
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return make_T(R, xyz)


# ---------------------------------------------------------------- 회전


def quat_from_R(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def R_from_quat(q) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """두 회전 사이 각도 (rad)."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def orientation_error(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """R_cur → R_target으로 가는 회전벡터 (부모 좌표계 기준)."""
    return Rotation.from_matrix(R_target @ R_cur.T).as_rotvec()


def quat_make_continuous(Q: np.ndarray) -> np.ndarray:
    """q와 -q 부호 뒤집힘 제거 (NaN 행은 건너뜀)."""
    Q = np.array(Q, float)
    last = None
    for i in range(len(Q)):
        if np.any(np.isnan(Q[i])):
            continue
        if last is not None and np.dot(Q[i], last) < 0:
            Q[i] = -Q[i]
        last = Q[i]
    return Q


def quat_slerp(q0, q1, a: float) -> np.ndarray:
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(Rs)).mean().as_matrix()


def parse_axes(spec) -> np.ndarray:
    """['y','z','-x'] → 회전행렬. i번째 원소 = 새 좌표계 i축을 기존 좌표계 축으로 표현."""
    basis = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    cols = []
    for s in spec:
        s = s.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        cols.append(sign * basis[s.lstrip("+-")])
    R = np.stack(cols, axis=1)
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError(f"axes {spec}는 오른손 좌표계가 아님 (det={np.linalg.det(R):.0f})")
    return R


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


# ---------------------------------------------------------------- 손 기하


def palm_frame(P5: np.ndarray, side: str = "right"):
    """손바닥 5점 [wrist, index MCP, middle MCP, ring MCP, pinky MCP] → (원점, R).

    x: wrist → middle MCP (손가락 방향)
    z: 손바닥 면 법선, 손바닥 쪽(바닥면)을 향함. 왼손/오른손 모두 같은 의미가 되도록 부호 결정
    y: z × x
    원점: wrist
    """
    P5 = np.asarray(P5, float)
    o = P5[0]
    _, _, Vt = np.linalg.svd(P5 - P5.mean(0))
    n = Vt[2]
    across = (P5[4] - P5[1]) if side == "right" else (P5[1] - P5[4])
    if np.dot(n, np.cross(P5[2] - o, across)) < 0:
        n = -n
    x = P5[2] - o
    x = normalize(x - np.dot(x, n) * n)
    y = np.cross(n, x)
    return o, np.stack([x, y, n], axis=1)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """dst ≈ s·R·src + t 최소자승 해 (Umeyama 1991). returns (s, R, t)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float((sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
