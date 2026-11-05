"""episode.npz → episode_proc.npz : 짧은 결측 보간 → Savitzky–Golay smoothing → IK feasibility.

    python -m handdata.apps.postprocess data/ep001

오프라인이라 미래 프레임까지 쓰는 양방향 smoothing (실시간 One Euro보다 lag 없음).
IK는 이전 해에서 warm start하며 순차로 풀고, 수렴 못한 프레임에 IK_FAIL 플래그.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import add_config_args, load_config
from ..core.filters import fill_gaps, segments, smooth_offline
from ..core.geometry import R_from_quat
from ..dataio.episode import flag_summary, load_episode, save_episode
from ..pipeline import IK_FAIL, INTERPOLATED


def run_ik(cfg, pos, quat, valid):
    from ..robot.ik import DLSSolver
    from ..robot.kinematics import MjKinematics
    kin = MjKinematics.from_config(cfg)
    solver = DLSSolver(kin, cfg.robot.ik)
    T = len(pos)
    q_all = np.full((T, kin.ndof), np.nan)
    ok = np.zeros(T, bool)
    pos_err = np.full(T, np.nan)
    for a, b in segments(valid):
        q = np.asarray(cfg.robot.home_q, float)
        for i in range(a, b):
            res = solver.solve(q, pos[i], R_from_quat(quat[i]), restarts=8 if i == a else 0)
            q_all[i], ok[i], pos_err[i] = res.q, res.converged, res.pos_err
            q = res.q
    return q_all, ok, pos_err


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("episode_dir")
    p.add_argument("--name", default="episode.npz")
    p.add_argument("--out-name", default="episode_proc.npz")
    p.add_argument("--no-ik", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    pp = cfg.postprocess
    e = load_episode(a.episode_dir, a.name)
    t, valid, flags = e["t"], e["valid"].astype(bool), e["flags"].copy()

    pos, quat, grip, filled = fill_gaps(t, e["ee_pos"], e["ee_quat"], e["gripper"], valid, pp.max_gap_sec)
    valid2 = valid | filled
    flags[filled] |= INTERPOLATED
    pos, quat, grip = smooth_offline(pos, quat, grip, valid2, pp.savgol_window, pp.savgol_polyorder)
    for arr in (pos, quat, grip):
        arr[~valid2] = np.nan

    out = {"t": t, "valid": valid2, "flags": flags, "ee_pos": pos, "ee_quat": quat, "gripper": grip}
    if pp.run_ik and not a.no_ik:
        q, ok, pe = run_ik(cfg, pos, quat, valid2)
        flags[valid2 & ~ok] |= IK_FAIL
        out.update(q=q, ik_ok=ok, ik_pos_err=pe, flags=flags)
        print(f"IK 수렴 {ok[valid2].mean() * 100:.1f}% (median pos err {np.nanmedian(pe) * 1000:.1f}mm)")

    save_episode(a.episode_dir, out, {"from": a.name, "postprocess": pp.to_plain(),
                                      "robot": cfg.robot.to_plain()},
                 a.out_name)
    segs = segments(valid2)
    print(f"유효 {valid2.sum()}/{len(t)} (보간 {filled.sum()}), 연속 구간 {len(segs)}개, "
          f"최장 {max((b - a for a, b in segs), default=0)} frames")
    print("플래그:", {k: n for k, n in flag_summary(flags).items() if n})


if __name__ == "__main__":
    main()
