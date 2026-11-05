"""Damped least squares (Levenberg–Marquardt) IK.

advaithdasu/hand-control-robot (MIT)의 ik.py를 회전행렬 기반으로 정리.
- 특이점 근처(manipulability 작음)에서 damping 증가 → 관절 속도 폭주 방지
- 매 iteration joint limit clamp
- 이전 해에서 warm start + step/error clamp → 프레임 간 IK branch 튐 방지
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.geometry import orientation_error


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    ori_err: float
    iters: int
    converged: bool
    manipulability: float


class DLSSolver:
    def __init__(self, kin, cfg):
        self.kin = kin
        self.c = cfg

    def solve(self, q_init, target_pos, target_R, restarts: int = 0) -> IKResult:
        best = self._descend(q_init, target_pos, target_R, self.c.max_iters)
        if best.converged or restarts <= 0:
            return best
        rng = np.random.default_rng(0)
        for _ in range(restarts):
            seed = self.kin.lower + rng.random(self.kin.ndof) * (self.kin.upper - self.kin.lower)
            res = self._descend(seed, target_pos, target_R, self.c.max_iters * 4)
            if res.converged:
                return res
            if res.pos_err + res.ori_err < best.pos_err + best.ori_err:
                best = res
        return best

    def _descend(self, q_init, target_pos, target_R, max_iters) -> IKResult:
        c = self.c
        q = self.kin.clamp(np.asarray(q_init, float).copy())
        manip, it = 0.0, 0
        for it in range(1, max_iters + 1):
            pos, R, J = self.kin.fk_and_jacobian(q)
            manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0))) if J.shape[1] >= 6 else 0.0
            e_pos = target_pos - pos
            e_ori = orientation_error(target_R, R)
            pe, oe = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_ori))
            if pe < c.pos_tol and oe < c.ori_tol:
                return IKResult(q, pe, oe, it, True, manip)
            if pe > c.max_pos_err:
                e_pos *= c.max_pos_err / pe
            e = np.concatenate([e_pos, c.ori_weight * e_ori])
            Jw = np.vstack([J[:3], c.ori_weight * J[3:]])
            lam = c.base_damping
            if manip < c.manip_threshold:
                lam += c.singular_damping * (1.0 - manip / c.manip_threshold)
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(6), e)
            q = self.kin.clamp(q + np.clip(dq, -c.max_step, c.max_step))
        pos, R = self.kin.fk(q)
        pe = float(np.linalg.norm(target_pos - pos))
        oe = float(np.linalg.norm(orientation_error(target_R, R)))
        return IKResult(q, pe, oe, it, pe < c.pos_tol and oe < c.ori_tol, manip)
