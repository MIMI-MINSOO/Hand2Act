"""[평가 1] 책상 위 손 GT로 2D→3D lifting 정확도 측정.

GT  : 2D keypoint 광선 ∩ 책상 평면 (보드 PnP로 평면을 앎) — 손 모델 캘리브레이션과 같은 원리
Pred: PnP lifting (공중에서 쓰는 것과 같은 방법, 평면 정보 사용 안 함)
같은 2D 검출을 공유하므로 "2D→3D 복원 오차"만 분리해서 측정됨.

    python -m handdata.apps.eval_board_gt --video eval/hand_table.mov --out results/board_gt_pnp.json
    python -m handdata.apps.eval_board_gt --video eval/hand_table.mov --set lifting.method=size_ratio \
        --out results/board_gt_sizeratio.json

주의: 캘리브레이션에 쓴 영상과 다른 영상(손 위치·카메라 거리·각도 다르게)으로 평가할 것.
책상 위 손은 평평해서 PnP에 유리한 조건 → 공중 정확도는 eval_marker_gt로 따로 확인.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget, backproject_to_plane
from ..core.camera import Intrinsics
from ..core.geometry import apply_T, inv_T, palm_frame
from ..core.landmarks import INDEX_TIP, PALM, THUMB_TIP, WRIST
from ..core.lifting import HandModel
from ..core.tracker import HandTracker
from ..core.video import add_source_args, open_source
from ..pipeline import FrameProcessor


def stats(x) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "n": int(len(x))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    add_source_args(p, stride=2)
    p.add_argument("--out", default="results/board_gt.json")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    model = HandModel.load(cfg.resolve(cfg.paths.hand_model))
    proc = FrameProcessor(cfg, intr, model)
    board = CharucoTarget(cfg.board)
    side, offset = cfg.hand.side, cfg.hand_model.plane_offset

    E = {k: [] for k in ["palm_mm", "wrist_mm", "fingertip_mm", "depth_mm", "normal_deg", "dist_m"]}
    with HandTracker.from_config(cfg) as tracker, open_source(a) as src:
        for frame, t, _ in src:
            det = tracker.detect(frame, t)
            bp = board.pose(frame, intr) if det is not None else None
            if bp is None:
                continue
            lift = proc.lift(det, t)
            if lift is None or lift.reproj_px > cfg.lifting.max_reproj_px:
                continue
            gt_cam = apply_T(bp.T_cam_board, backproject_to_plane(det.uv, intr, bp.T_cam_board, offset))
            pr_cam = lift.kp3d_cam
            err = np.linalg.norm(pr_cam - gt_cam, axis=1) * 1000
            E["palm_mm"].append(err[PALM].mean())
            E["wrist_mm"].append(err[WRIST])
            E["fingertip_mm"].append(err[[THUMB_TIP, INDEX_TIP]].mean())
            E["depth_mm"].append(abs(pr_cam[WRIST, 2] - gt_cam[WRIST, 2]) * 1000)
            n_gt = palm_frame(gt_cam[PALM], side)[1][:, 2]
            n_pr = lift.T_cam_hand[:3, 2]
            E["normal_deg"].append(np.degrees(np.arccos(np.clip(n_gt @ n_pr, -1, 1))))
            E["dist_m"].append(gt_cam[WRIST, 2])

    if not E["palm_mm"]:
        raise SystemExit("평가 가능한 프레임 없음")
    dist = np.array(E.pop("dist_m"))
    res = {"config": {"lift_method": proc.lift_method, "pnp": cfg.lifting.pnp_method,
                      "finger": cfg.lifting.finger_method, "hand_model": model.method,
                      "offset": offset},
           "distance_m": stats(dist), **{k: stats(v) for k, v in E.items()}}
    # 거리 구간별 손바닥 오차 (깊이 오차 ∝ Z² 확인용)
    bins = np.arange(0.2, 1.01, 0.1)
    res["palm_mm_by_distance"] = {
        f"{lo:.1f}-{hi:.1f}m": stats(np.array(E["palm_mm"])[(dist >= lo) & (dist < hi)])
        for lo, hi in zip(bins[:-1], bins[1:]) if np.any((dist >= lo) & (dist < hi))}

    out = Path(cfg.resolve(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"프레임 {res['palm_mm']['n']}개, 거리 median {res['distance_m']['median']:.2f}m")
    for k in ["palm_mm", "wrist_mm", "fingertip_mm", "depth_mm", "normal_deg"]:
        s = res[k]
        print(f"  {k:13s} median {s['median']:6.1f}  mean {s['mean']:6.1f}  p90 {s['p90']:6.1f}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
