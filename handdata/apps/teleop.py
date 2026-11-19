"""웹캠 손 → MuJoCo 로봇 실시간 teleop (상대 매핑 + clutch). 캘리브레이션 없이 바로 동작.

    python -m handdata.apps.teleop --config configs/nero_dual.yaml                 # 양손 → 양팔
    python -m handdata.apps.teleop --config configs/nero_dual.yaml --arms right    # 오른손 → 오른팔
    python -m handdata.apps.teleop --config configs/nero_dual.yaml --replay data/teleop/tp_xxx.npz

절대 좌표 대신 '손이 움직인 만큼' 로봇을 움직임 (VR teleop의 clutch 방식):
    손이 처음 보인 순간(또는 space)에  손 pose H0, 로봇 EE pose E0 를 anchor
    p_target = p_E0 + s · R_bc (p_H − p_H0)
    R_target = (R_bc R_H R_H0ᵀ R_bcᵀ) · R_E0           (R_bc = 카메라 → 로봇 base 축 변환)
→ 근사 K(깊이 스케일 오차), 즉석 손 모델, 카메라 설치 위치 오차가 있어도 방향·자세는 직관적으로 맞음.
   (보드로 base 기준 절대 pose를 쓰는 기록용 경로는 live.py)

손 3D 복원은 quicktest와 동일 (MediaPipe 21점 → 손바닥 PnP → EE), 손 모델은 시작할 때 손별로 즉석 생성.
키 (카메라 창에서): space 재-anchor · h 홈 자세 · r 기록 시작/정지 · k 손 모델 다시 · m 미러 · q 종료
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..calib.hand_model import model_from_observations
from ..config import add_config_args, load_config
from ..core.filters import PoseFilter
from ..core.geometry import R_from_quat, parse_axes, quat_from_R, rot_angle
from ..core.lifting import HandModel, hand_points_in_palm_frame
from ..core.tracker import HandTracker
from ..core.video import FrameSource
from ..pipeline import NO_HAND, PNP_FAIL, FrameProcessor
from ..viz import draw_frame, put_lines
from .quicktest import get_intrinsics


def _scaled_rotation(R: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return R
    rv, _ = cv2.Rodrigues(R)
    return cv2.Rodrigues(rv * s)[0]


class ArmTeleop:
    """팔 하나: anchor(clutch) + One Euro + DLS IK."""

    def __init__(self, cfg, model, side: str, acfg):
        from ..robot.ik import DLSSolver
        from ..robot.kinematics import MjKinematics, actuator_open_value, gripper_open_values
        self.side = side
        self.kin = MjKinematics(None, acfg.joints, acfg.ee_site, model=model)
        self.solver = DLSSolver(self.kin, cfg.robot.ik)
        self.home = np.asarray(acfg.home_q, float)
        self.q = self.home.copy()
        self.g = 1.0
        self.g_adr = [model.joint(n).qposadr[0] for n in acfg.gripper_joints]
        self.g_open = gripper_open_values(model, acfg.gripper_joints)     # qpos = g · open
        # 물리 모드용 액추에이터 (없으면 kinematic 표시만)
        self.act_q = [model.actuator(n).id for n in acfg.get("actuators", [])]
        ga = acfg.get("gripper_actuator")
        self.act_g = [model.actuator(ga).id] if ga else []
        self.g_ctrl_open = actuator_open_value(model, ga) if ga else 0.0
        self.filt = PoseFilter(cfg.filter_live)
        self.anchor = None
        self.last_seen = -np.inf
        self.last_T = None          # 마지막으로 받아들인 손 pose (튐 검사용)
        self.n_reject = 0
        self.ik = None
        self.target = None

    def reanchor(self):
        self.anchor = None

    def go_home(self):
        self.q, self.anchor = self.home.copy(), None

    def update(self, t, T_cam_ee, g, R_bc, tcfg) -> str:
        """→ 'ok' | 'anchor' | 'jump'(이 프레임 버림)."""
        status = "ok"
        if self.anchor is None or t - self.last_seen > tcfg.hold_sec:
            p_e, R_e = self.kin.fk(self.q)
            self.anchor = (T_cam_ee[:3, 3].copy(), T_cam_ee[:3, :3].copy(), p_e, R_e)
            self.filt.reset()
            status = "anchor"
        elif self.last_T is not None:
            # 손바닥 PnP flip(앞/뒷면 해 뒤바뀜)·양손 뒤바뀜 등으로 한 프레임에 크게 튀면 버림.
            # 연속으로 계속 튀면 진짜 움직임으로 보고 그 자리에서 재-anchor
            jump_deg = np.degrees(rot_angle(self.last_T[:3, :3], T_cam_ee[:3, :3]))
            jump_m = np.linalg.norm(self.last_T[:3, 3] - T_cam_ee[:3, 3])
            if jump_deg > tcfg.get("max_jump_deg", 45) or jump_m > tcfg.get("max_jump_m", 0.08):
                self.n_reject += 1
                if self.n_reject < tcfg.get("max_reject", 6):
                    return "jump"
                self.anchor, self.n_reject, self.last_T = None, 0, None
                return self.update(t, T_cam_ee, g, R_bc, tcfg)
        self.n_reject = 0
        self.last_T = T_cam_ee.copy()
        self.last_seen = t
        pH0, RH0, pE0, RE0 = self.anchor
        p = pE0 + tcfg.pos_scale * R_bc @ (T_cam_ee[:3, 3] - pH0)
        dR = _scaled_rotation(R_bc @ (T_cam_ee[:3, :3] @ RH0.T) @ R_bc.T, tcfg.rot_scale)
        p, quat, self.g = self.filt(t, p, quat_from_R(dR @ RE0), g)
        self.target = (p, quat)
        self.ik = self.solver.solve(self.q, p, R_from_quat(quat))
        self.q = self.ik.q
        return status

    def write(self, data, physics: bool):
        if physics and self.act_q:
            data.ctrl[self.act_q] = self.q
            data.ctrl[self.act_g] = self.g * self.g_ctrl_open
        else:
            data.qpos[self.kin.qpos_adr] = self.q
            data.qpos[self.g_adr] = self.g * self.g_open


class SimScene:
    def __init__(self, cfg, sides, physics: bool):
        import mujoco
        import mujoco.viewer
        self.mj, self.physics = mujoco, physics
        self.model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
        self.data = mujoco.MjData(self.model)
        self.arms = {s: ArmTeleop(cfg, self.model, s, cfg.arms[s]) for s in cfg.arms}
        self.active = [s for s in sides if s in self.arms]
        for arm in self.arms.values():                   # 시작 자세 = 홈 (물리 모드도 qpos 초기화)
            self.data.qpos[arm.kin.qpos_adr] = arm.q
            self.data.qpos[arm.g_adr] = arm.g * arm.g_open
            arm.write(self.data, physics)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False,
                                                   show_right_ui=False)
        v = cfg.get("teleop", {}).get("viewer")
        if v:
            with self.viewer.lock():
                self.viewer.cam.azimuth, self.viewer.cam.elevation = v["azimuth"], v["elevation"]
                self.viewer.cam.distance = v["distance"]
                self.viewer.cam.lookat[:] = v["lookat"]
        self._t = time.perf_counter()

    def step(self):
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        for arm in self.arms.values():
            arm.write(self.data, self.physics)
        if self.physics:
            n = int(np.clip(round(dt / self.model.opt.timestep), 1, 50))
            for _ in range(n):
                self.mj.mj_step(self.model, self.data)
        else:
            self.mj.mj_forward(self.model, self.data)
        self.viewer.sync()

    @property
    def running(self):
        return self.viewer.is_running()

    def close(self):
        self.viewer.close()


# ======================================================================== 기록 / 재생


class Recorder:
    def __init__(self, sim: SimScene):
        self.sim, self.rows, self.t0 = sim, [], time.perf_counter()

    def add(self):
        row = {"t": time.perf_counter() - self.t0, "qpos": self.sim.data.qpos.copy()}
        for s, arm in self.sim.arms.items():
            row[f"{s}_q_cmd"], row[f"{s}_gripper"] = arm.q.copy(), arm.g
            p, R = arm.kin.fk(arm.q)
            row[f"{s}_ee_pos"], row[f"{s}_ee_quat"] = p, quat_from_R(R)
        self.rows.append(row)

    def save(self, out_root, cfg) -> Path:
        out = Path(out_root) / time.strftime("tp_%Y%m%d_%H%M%S.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(out, **arr, mjcf=str(cfg.robot.mjcf), arms=np.array(list(self.sim.arms)))
        print(f"■ 저장: {out} ({len(self.rows)} frames, {arr['t'][-1]:.1f}s)  "
              f"키: t, qpos, <arm>_q_cmd / _gripper / _ee_pos / _ee_quat")
        return out


def replay(cfg, path, speed: float):
    import mujoco
    import mujoco.viewer
    d = np.load(path)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve(cfg.robot.mjcf)))
    data = mujoco.MjData(model)
    print(f"재생: {path} ({len(d['t'])} frames) — 창을 닫으면 종료")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            t0 = time.perf_counter()
            for k in range(len(d["t"])):
                data.qpos[:] = d["qpos"][k]
                mujoco.mj_forward(model, data)
                v.sync()
                while time.perf_counter() - t0 < d["t"][k] / speed:
                    time.sleep(0.002)
                if not v.is_running():
                    return


# ======================================================================== 손 ↔ 팔 매칭


def assign_hands(cands, sides, prev: dict, t: float, hold: float, mirrored: bool, max_px: float = 250.0):
    """검출된 손들 → {팔: det}.

    직전 프레임 손목 픽셀이 있으면 가까운 쪽으로 이어 붙임 (MediaPipe handedness 라벨이 프레임마다
    뒤바뀌어도 팔이 바뀌지 않게). 없으면 화면 위치(두 손) 또는 handedness(한 손).
    """
    if not cands:
        return {}
    d = lambda c, s: float(np.linalg.norm(c.uv[0] - prev[s][0]))
    recent = [s for s in sides if s in prev and t - prev[s][1] <= hold]
    if len(sides) == 1:
        s = sides[0]
        if s in recent:
            c = min(cands, key=lambda c: d(c, s))
            if d(c, s) <= max_px:
                return {s: c}
        same = [c for c in cands if c.side == s]
        return {s: max(same or cands, key=lambda c: c.score)}
    if len(cands) >= 2:
        a, b = sorted(cands, key=lambda c: -c.score)[:2]
        if len(recent) == 2:
            keep = d(a, "right") + d(b, "left") <= d(a, "left") + d(b, "right")
            return {"right": a, "left": b} if keep else {"right": b, "left": a}
        a, b = sorted((a, b), key=lambda c: c.uv[0, 0])              # 화면 왼쪽부터
        return {"left": a, "right": b} if mirrored else {"right": a, "left": b}   # 미러 아니면 오른손이 화면 왼쪽
    c = cands[0]
    near = [s for s in recent if d(c, s) <= max_px]
    if near:
        return {min(near, key=lambda s: d(c, s)): c}
    return {c.side: c} if c.side in sides else {}


# ======================================================================== main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--camera", default="0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hfov", type=float, default=65.0, help="근사 K용 가로 화각 [deg]")
    p.add_argument("--arms", default="both", choices=["both", "right", "left"])
    p.add_argument("--wrist-middle", type=float, default=0.09, help="자로 잰 wrist–middle MCP 길이 [m]")
    p.add_argument("--calib-frames", type=int, default=30)
    p.add_argument("--kinematic", action="store_true", help="물리 스텝 없이 관절값만 표시 (물체 상호작용 X)")
    p.add_argument("--out", default="data/teleop")
    p.add_argument("--replay", default=None, help="기록한 npz 재생")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--save-video", action="store_true",
                   help="원본 카메라 영상 + 프레임별 로그 저장 (같은 영상을 --camera 로 넣어 오프라인 재현/디버깅)")
    p.add_argument("--no-window", action="store_true", help="카메라 창 없이 (영상 파일 재현용)")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if "arms" not in cfg:
        raise SystemExit("config에 arms 항목이 없음 → configs/nero_dual.yaml 참고")
    if a.replay:
        return replay(cfg, a.replay, a.speed)
    tcfg = cfg.teleop
    R_bc = parse_axes(tcfg.cam_axes_in_base)
    sides = ["right", "left"] if a.arms == "both" else [a.arms]

    sim = SimScene(cfg, sides, physics=not a.kinematic)
    models: dict = {}
    if cfg.resolve(cfg.paths.hand_model).exists():
        m = HandModel.load(cfg.resolve(cfg.paths.hand_model))
        models[m.side] = m
        print(f"손 모델 로드: {m.side} ({m.method})")
    obs = {s: [] for s in sides}
    procs: dict = {}
    intr, mirror, rec = None, True, None
    prev_uv: dict = {}
    stats = {s: {} for s in sides}
    log, vw, n_frames, wall0 = [], None, 0, time.perf_counter()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    def count(s, k):
        stats[s][k] = stats[s].get(k, 0) + 1

    with HandTracker.from_config(cfg) as tracker, \
            FrameSource(a.camera, width=a.width, height=a.height) as src:
        for frame, t, _ in src:
            n_frames += 1
            if intr is None:
                intr, how = get_intrinsics(cfg, frame, a.hfov)
                print(f"영상 {frame.shape[1]}x{frame.shape[0]}, {how}")
                if a.save_video:
                    Path(a.out).mkdir(parents=True, exist_ok=True)
                    vpath = f"{a.out}/session_{stamp}.mp4"        # 상대경로 (OpenCV는 한글 절대경로 못 씀)
                    vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (frame.shape[1], frame.shape[0]))
                    print(f"영상 저장: {vpath}")
            if vw is not None:
                vw.write(frame)
            dets = assign_hands(tracker.detect_all(frame, t), sides, prev_uv, t, tcfg.hold_sec,
                                tracker.input_mirrored)
            for s, det in dets.items():
                prev_uv[s] = (det.uv[0].copy(), t)
            vis = frame.copy()
            lines = []

            for s in sides:
                det = dets.get(s)
                if s not in models:                                   # 즉석 손 모델 (자 방식)
                    if det is not None:
                        obs[s].append(hand_points_in_palm_frame(det.world, s))
                        for uv in det.uv.astype(int):
                            cv2.circle(vis, tuple(uv), 4, (0, 220, 255), -1)
                    lines.append(f"[{s.upper()} HAND MODEL] open {s} hand toward camera, hold  "
                                 f"{len(obs[s])}/{a.calib_frames}")
                    if len(obs[s]) >= a.calib_frames:
                        models[s] = model_from_observations(obs[s], s, "ruler", 0.0, a.wrist_middle,
                                                            {"source": "teleop"})
                        print(f"{s} 손 모델: 손바닥 폭 {models[s].meta['palm_width_m'] * 100:.1f}cm, "
                              f"흔들림 {np.mean(models[s].std_mm):.1f}mm")
                    continue
                if s not in procs:
                    procs[s] = FrameProcessor(cfg, intr, models[s])
                arm = sim.arms[s]
                r = procs[s].process(t, det, None)
                ok = not r.flags & (NO_HAND | PNP_FAIL) and r.reproj_px <= tcfg.max_reproj_px
                row = {"t": t, "side": s, "status": "", "reproj": r.reproj_px, "choice": r.choice,
                       "T_cam_ee": np.full((4, 4), np.nan), "ik_err": np.nan}
                if ok:
                    T_cam_ee = procs[s].retarget.ee_pose(r.T_cam_hand, r.kp3d_cam)
                    st = arm.update(t, T_cam_ee, r.gripper, R_bc, tcfg)
                    row.update(status=st, T_cam_ee=T_cam_ee)
                    vis = draw_frame(vis, intr, r, models[s].palm, hud=False)
                    if st == "jump":
                        lines.append(f"{s:5s} skip (pose jump {arm.n_reject})")
                    else:
                        ik = arm.ik
                        row["ik_err"] = ik.pos_err
                        if not ik.converged:
                            count(s, "ik_not_converged")
                        lines.append(f"{s:5s} grip {arm.g:4.2f}  reproj {r.reproj_px:4.1f}px  "
                                     f"IK {'ok' if ik.converged else 'x'} {ik.pos_err * 1000:3.0f}mm")
                else:
                    st = "no_hand" if r.flags & NO_HAND else ("pnp_fail" if r.flags & PNP_FAIL else "high_reproj")
                    row["status"] = st
                    lines.append(f"{s:5s} hold ({st} {'' if st != 'high_reproj' else f'{r.reproj_px:.0f}px'})")
                count(s, row["status"])
                row["q"] = arm.q.copy()
                log.append(row)

            sim.step()
            if rec is not None:
                rec.add()
                lines.append(f"REC {len(rec.rows)}")

            if a.no_window:
                if not sim.running:
                    break
                continue
            if mirror:
                vis = cv2.flip(vis, 1)
            put_lines(vis, lines + ["space: re-anchor  h: home  r: rec  k: hand model  m: mirror  q: quit"])
            cv2.imshow("Hand2Act teleop", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or not sim.running:
                break
            if k == ord(" "):
                for arm in sim.arms.values():
                    arm.reanchor()
            if k == ord("h"):
                for arm in sim.arms.values():
                    arm.go_home()
            if k == ord("m"):
                mirror = not mirror
            if k == ord("k"):
                models, obs, procs = {}, {s: [] for s in sides}, {}
            if k == ord("r"):
                if rec is None:
                    rec = Recorder(sim)
                    print("● 기록 시작")
                else:
                    rec.save(a.out, cfg) if rec.rows else None
                    rec = None
    if rec is not None and rec.rows:
        rec.save(a.out, cfg)
    if vw is not None:
        vw.release()
    if a.save_video and log:
        lp = Path(a.out) / f"session_{stamp}_log.npz"
        np.savez_compressed(lp, **{k: np.array([r[k] for r in log]) for k in log[0]})
        print(f"로그 저장: {lp}")
    wall = time.perf_counter() - wall0
    print(f"\n=== 세션 요약: {n_frames} frames, {wall:.1f}s, {n_frames / max(wall, 1e-6):.1f} fps")
    for s, c in stats.items():
        n = sum(v for k, v in c.items() if k != "ik_not_converged")
        print(f"  {s:5s}: " + ", ".join(f"{k} {v} ({100 * v / max(n, 1):.0f}%)" for k, v in sorted(c.items())))
    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
