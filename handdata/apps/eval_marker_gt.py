"""[평가 2] 손등 ArUco 마커로 '공중' lifting 일관성 측정.

마커는 손등에 붙어 있으므로 palm frame과 마커 사이 변환 X = T_palm_marker는 (손바닥이 강체라면) 일정해야 함.
    X_i = T_cam_palm_i⁻¹ · T_cam_marker_i
평균 X̄를 구한 뒤 각 프레임 잔차(위치 mm, 회전 deg)를 lifting 오차 지표로 사용.
마커 pose 자체 오차도 섞이므로 '마커+lifting 합산 오차'의 상한 성격.

    python -m handdata.calib.make_board --hand-marker-id 40 --hand-marker-mm 30   # 마커 출력
    python -m handdata.apps.eval_marker_gt --video eval/marker_hand.mov --marker-id 40 --marker-size 0.030
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.camera import Intrinsics
from ..core.geometry import T_from_rvec_tvec, inv_T, mean_rotation, rot_angle
from ..core.lifting import HandModel
from ..core.tracker import HandTracker
from ..core.video import add_source_args, open_source
from ..pipeline import FrameProcessor
from .eval_board_gt import stats


class MarkerTracker:
    def __init__(self, dictionary: str, marker_id: int, size: float):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self.det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        self.id, s = marker_id, size / 2
        # IPPE_SQUARE 규약 순서: 좌상, 우상, 우하, 좌하 (detectMarkers 코너 순서와 동일)
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        self.prev_R = None

    def pose(self, img, intr: Intrinsics):
        corners, ids, _ = self.det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None or self.id not in ids.reshape(-1):
            return None
        c = corners[list(ids.reshape(-1)).index(self.id)].reshape(4, 2).astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, c, intr.K, intr.dist,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        Ts = [T_from_rvec_tvec(r, t) for r, t in zip(rvecs, tvecs)]
        errs = np.asarray(errs).reshape(-1)
        i = int(np.argmin(errs))
        if self.prev_R is not None and len(Ts) == 2 and errs.max() < 3 * max(errs.min(), 1e-6):
            i = int(np.argmin([rot_angle(self.prev_R, T[:3, :3]) for T in Ts]))   # 작은 마커 flip 대응
        self.prev_R = Ts[i][:3, :3]
        return Ts[i]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    add_source_args(p, stride=1)
    p.add_argument("--marker-id", type=int, required=True)
    p.add_argument("--marker-size", type=float, required=True, help="검은 정사각형 한 변 [m]")
    p.add_argument("--out", default="results/marker_gt.json")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    proc = FrameProcessor(cfg, intr, HandModel.load(cfg.resolve(cfg.paths.hand_model)))
    mk = MarkerTracker(cfg.board.dictionary, a.marker_id, a.marker_size)

    Xs, dist = [], []
    with HandTracker.from_config(cfg) as tracker, open_source(a) as src:
        for frame, t, _ in src:
            det = tracker.detect(frame, t)
            T_cam_marker = mk.pose(frame, intr) if det is not None else None
            if T_cam_marker is None:
                continue
            lift = proc.lift(det, t)
            if lift is None or lift.reproj_px > cfg.lifting.max_reproj_px:
                continue
            Xs.append(inv_T(lift.T_cam_hand) @ T_cam_marker)
            dist.append(lift.T_cam_hand[2, 3])
    if len(Xs) < 20:
        raise SystemExit(f"유효 프레임 {len(Xs)}개: 너무 적음")

    Xs = np.stack(Xs)
    t_mean = np.median(Xs[:, :3, 3], axis=0)
    R_mean = mean_rotation(Xs[:, :3, :3])
    pos_mm = np.linalg.norm(Xs[:, :3, 3] - t_mean, axis=1) * 1000
    rot_deg = np.degrees([rot_angle(R_mean, X[:3, :3]) for X in Xs])
    res = {"config": {"lift_method": proc.lift_method, "finger": cfg.lifting.finger_method},
           "n": len(Xs), "distance_m": stats(dist), "pos_residual_mm": stats(pos_mm),
           "rot_residual_deg": stats(rot_deg),
           "T_palm_marker_mean": {"t": t_mean.tolist(), "R": R_mean.tolist()}}
    out = Path(cfg.resolve(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"프레임 {len(Xs)}개, 거리 median {np.median(dist):.2f}m")
    print(f"  위치 잔차 median {np.median(pos_mm):.1f}mm  p90 {np.percentile(pos_mm, 90):.1f}mm")
    print(f"  회전 잔차 median {np.median(rot_deg):.1f}°   p90 {np.percentile(rot_deg, 90):.1f}°")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
