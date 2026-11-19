"""MuJoCo 기반 범용 FK/Jacobian. MJCF 경로 + joint 이름 + EE site 이름만 있으면 어떤 로봇이든 동작.

advaithdasu/hand-control-robot (MIT)의 kinematics.py를 config 기반으로 일반화.
IK 반복이 실제 시뮬 상태를 건드리지 않도록 별도 scratch MjData 사용.
"""

from __future__ import annotations

import mujoco
import numpy as np


def gripper_open_values(model: mujoco.MjModel, joint_names) -> np.ndarray:
    """손가락 관절별 '완전히 열림' qpos. 닫힘 = 0, 열림 = 관절 범위에서 0과 먼 쪽 끝.

    AgileX 공식 모델처럼 두 손가락 축이 반대(0~0.05 / -0.05~0)여도 g ∈ [0,1] 하나로 제어:
        qpos = g · open
    """
    out = []
    for n in joint_names:
        lo, hi = model.joint(n).range
        out.append(hi if abs(hi) >= abs(lo) else lo)
    return np.asarray(out, float)


def actuator_open_value(model: mujoco.MjModel, name: str) -> float:
    lo, hi = model.actuator(name).ctrlrange
    return float(hi if abs(hi) >= abs(lo) else lo)


class MjKinematics:
    def __init__(self, mjcf_path, joint_names, site_name, model: mujoco.MjModel | None = None):
        self.model = model or mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(site_name).id
        self.joint_names = list(joint_names)
        self.qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in joint_names])
        self.dof_adr = np.array([self.model.joint(n).dofadr[0] for n in joint_names])
        rng = np.array([self.model.joint(n).range for n in joint_names])
        self.lower, self.upper = rng[:, 0], rng[:, 1]
        self.ndof = len(joint_names)

    @classmethod
    def from_config(cls, cfg):
        r = cfg.robot
        return cls(cfg.resolve(r.mjcf), r.joints, r.ee_site)

    def clamp(self, q):
        return np.clip(q, self.lower, self.upper)

    def _sync(self, q):
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _pose(self):
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def fk(self, q):
        """q → (EE 위치, EE 회전행렬), MuJoCo world(=로봇 base로 가정) 기준."""
        self._sync(q)
        return self._pose()

    def fk_and_jacobian(self, q):
        self._sync(q)
        pos, R = self._pose()
        mujoco.mj_comPos(self.model, self.data)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return pos, R, np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
