"""필터링.

실시간(causal): One Euro filter — advaithdasu/hand-control-robot (MIT)의 filters.py를 가져와 정리
오프라인(non-causal): 결측 보간 + Savitzky–Golay smoothing (미래 프레임까지 사용 → lag 없음)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .geometry import quat_make_continuous, quat_slerp, rot_angle, R_from_quat

# ======================================================================== 실시간


def _alpha(dt: float, cutoff: float) -> float:
    r = 2.0 * np.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """벡터 One Euro filter (Casiez et al., 2012). 정지 시 강하게, 빠를 때 약하게 smoothing."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = None

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self._x is None:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return x.copy()
        if dt <= 0:
            return self._x.copy()
        self._dx = _alpha(dt, self.d_cutoff) * (x - self._x) / dt + (1 - _alpha(dt, self.d_cutoff)) * self._dx
        a = _alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(self._dx)))
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class QuatOneEuro:
    """회전용 One Euro: 각속도에 따라 cutoff가 열리는 slerp low-pass."""

    def __init__(self, min_cutoff=2.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._q, self._rate = None, 0.0

    def __call__(self, q, dt):
        q = np.asarray(q, float) / np.linalg.norm(q)
        if self._q is None:
            self._q = q.copy()
            return q.copy()
        if dt <= 0:
            return self._q.copy()
        ang = rot_angle(R_from_quat(self._q), R_from_quat(q))
        a_d = _alpha(dt, self.d_cutoff)
        self._rate = a_d * ang / dt + (1 - a_d) * self._rate
        self._q = quat_slerp(self._q, q, _alpha(dt, self.min_cutoff + self.beta * self._rate))
        return self._q.copy()


class PoseFilter:
    """EE 위치·자세·그리퍼를 한 번에 거르는 래퍼 (live teleop용)."""

    def __init__(self, cfg):
        self.pos = OneEuroFilter(cfg.pos_min_cutoff, cfg.pos_beta, cfg.d_cutoff)
        self.rot = QuatOneEuro(cfg.rot_min_cutoff, cfg.rot_beta, cfg.d_cutoff)
        self.grip = OneEuroFilter(cfg.grip_min_cutoff, 0.0, cfg.d_cutoff)
        self._t = None

    def reset(self):
        self.pos.reset(); self.rot.reset(); self.grip.reset(); self._t = None

    def __call__(self, t, pos, quat, grip):
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        return self.pos(pos, dt), self.rot(quat, dt), float(self.grip(np.array([grip]), dt)[0])


# ======================================================================== 오프라인


def segments(mask: np.ndarray):
    """True 연속 구간 [(start, end_exclusive), ...]."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_gaps(t, pos, quat, grip, valid, max_gap_sec: float):
    """valid 사이의 짧은 결측(≤ max_gap_sec)을 보간. → (pos, quat, grip, filled_mask)."""
    pos, quat, grip = pos.copy(), quat.copy(), grip.copy()
    filled = np.zeros(len(t), bool)
    idx = np.where(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 1 or t[b] - t[a] > max_gap_sec:
            continue
        for k in range(a + 1, b):
            w = (t[k] - t[a]) / (t[b] - t[a])
            pos[k] = (1 - w) * pos[a] + w * pos[b]
            grip[k] = (1 - w) * grip[a] + w * grip[b]
            quat[k] = quat_slerp(quat[a], quat[b], w)
            filled[k] = True
    return pos, quat, grip, filled


def smooth_offline(pos, quat, grip, valid, window: int, polyorder: int):
    """valid 연속 구간마다 Savitzky–Golay. 구간이 window보다 짧으면 그대로 둠."""
    pos, quat, grip = pos.copy(), quat_make_continuous(quat), grip.copy()
    window = window if window % 2 == 1 else window + 1
    for a, b in segments(valid):
        if b - a < window:
            continue
        pos[a:b] = savgol_filter(pos[a:b], window, polyorder, axis=0)
        grip[a:b] = savgol_filter(grip[a:b], window, polyorder)
        q = savgol_filter(quat[a:b], window, polyorder, axis=0)  # 작은 window에서는 성분별 smoothing 후 정규화로 충분
        quat[a:b] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return pos, quat, np.clip(grip, 0.0, 1.0)
