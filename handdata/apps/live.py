"""실시간: 카메라/스트림 → EE pose → (MuJoCo 시뮬 teleop) + 에피소드 기록.

    python -m handdata.apps.live --camera 0 --sim
    python -m handdata.apps.live --camera "rtsp://<iphone-ip>:8554/live" --sim

키: r 기록 시작/정지 · c 필터·flip 상태 리셋 · q 종료
- 기록은 필터 전 원본 결과를 저장 (smoothing은 postprocess에서 non-causal로)
- 시뮬 표시는 One Euro 필터 + DLS IK 결과
- macOS에서 --sim은 `mjpython -m handdata.apps.live ...`로 실행해야 함 (MuJoCo viewer 제약)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget
from ..core.camera import Intrinsics
from ..core.filters import PoseFilter
from ..core.geometry import R_from_quat
from ..core.lifting import HandModel
from ..core.tracker import HandTracker
from ..core.video import FrameSource
from ..dataio.episode import EpisodeBuffer, save_episode
from ..pipeline import FrameProcessor
from ..viz import draw_frame


class SimView:
    """MuJoCo passive viewer에 IK 결과를 kinematic하게 표시 (물리 스텝 없음 → 어떤 MJCF든 동작)."""

    def __init__(self, cfg):
        import mujoco
        import mujoco.viewer
        from ..robot.ik import DLSSolver
        from ..robot.kinematics import MjKinematics
        self.mj = mujoco
        self.kin = MjKinematics.from_config(cfg)
        self.solver = DLSSolver(self.kin, cfg.robot.ik)
        self.data = mujoco.MjData(self.kin.model)
        self.q = np.asarray(cfg.robot.home_q, float)
        gj = cfg.robot.get("gripper_joints", []) or []
        self.g_adr = [self.kin.model.joint(n).qposadr[0] for n in gj]
        self.g_lo, self.g_hi = cfg.robot.get("gripper_joint_range", [0.0, 0.0])
        self.viewer = mujoco.viewer.launch_passive(self.kin.model, self.data)
        self.last = None
        self._apply(0.5)

    def _apply(self, g):
        self.data.qpos[self.kin.qpos_adr] = self.q
        for adr in self.g_adr:
            self.data.qpos[adr] = self.g_lo + g * (self.g_hi - self.g_lo)
        self.mj.mj_forward(self.kin.model, self.data)
        self.viewer.sync()

    def update(self, pos, quat, g):
        self.last = self.solver.solve(self.q, pos, R_from_quat(quat))
        self.q = self.last.q
        self._apply(g)

    @property
    def running(self):
        return self.viewer.is_running()

    def close(self):
        self.viewer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--camera", default="0", help="카메라 인덱스, 스트림 URL, 또는 영상 파일(재생 테스트)")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--out", default="data/live")
    p.add_argument("--hold-sec", type=float, default=0.5, help="이 시간 이상 검출이 끊기면 필터 리셋")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    model = HandModel.load(cfg.resolve(cfg.paths.hand_model))
    proc = FrameProcessor(cfg, intr, model)
    board = CharucoTarget(cfg.board)
    filt = PoseFilter(cfg.filter_live)
    sim = SimView(cfg) if a.sim else None

    buf, rec_t0, last_valid_t = None, 0.0, -np.inf
    checked = False
    with HandTracker.from_config(cfg) as tracker, \
            FrameSource(a.camera, width=intr.width, height=intr.height) as src:
        for frame, t, _ in src:
            if not checked:
                intr.check_frame(frame)
                checked = True
            tic = time.perf_counter()
            det = tracker.detect(frame, t)
            r = proc.process(t, det, board.pose(frame, intr))
            dt_ms = (time.perf_counter() - tic) * 1000

            if r.valid:
                if t - last_valid_t > a.hold_sec:
                    filt.reset()
                last_valid_t = t
                pos, quat, g = filt(t, r.ee_pos, r.ee_quat, r.gripper)
                if sim is not None:
                    sim.update(pos, quat, g)
            if buf is not None:
                r.t = t - rec_t0
                buf.add(r)

            vis = draw_frame(frame, intr, r, model.palm)
            hud = f"{dt_ms:5.1f} ms/frame"
            if sim is not None and sim.last is not None:
                hud += f"  IK {'ok' if sim.last.converged else 'FAIL'} ({sim.last.pos_err * 1000:.0f}mm)"
            if buf is not None:
                hud += f"  ● REC {len(buf)}"
            cv2.putText(vis, hud, (12, vis.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("hand-pose-data", cv2.resize(vis, None, fx=0.6, fy=0.6) if vis.shape[1] > 1600 else vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or (sim is not None and not sim.running):
                break
            if k == ord("c"):
                filt.reset(); proc.reset()
            if k == ord("r"):
                if buf is None:
                    buf, rec_t0 = EpisodeBuffer(), t
                    print("● 기록 시작")
                else:
                    _save(buf, a.out, cfg)
                    buf = None
    if buf is not None and len(buf):
        _save(buf, a.out, cfg)
    cv2.destroyAllWindows()
    if sim is not None:
        sim.close()


def _save(buf, out_root, cfg):
    out = Path(out_root) / time.strftime("ep_%Y%m%d_%H%M%S")
    arr = buf.to_arrays()
    save_episode(out, arr, {"source": {"live": True}, "config": cfg.to_plain()})
    print(f"■ 저장: {out} ({arr['valid'].sum()}/{len(arr['t'])} valid)")


if __name__ == "__main__":
    main()
