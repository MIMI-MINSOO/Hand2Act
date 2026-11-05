"""그리퍼 개폐 범위 캘리브레이션: 엄지–검지 끝 3D 거리의 최소/최대.

    python -m handdata.calib.gripper_range --video calib_raw/pinch.mov

촬영: 수집 때처럼 손을 공중에 두고 엄지·검지를 최대로 벌렸다 완전히 붙였다를 5~10회 반복.
노이즈에 강하도록 극값 대신 하위/상위 percentile 사용.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..config import add_config_args, load_config
from ..core.camera import Intrinsics
from ..core.landmarks import INDEX_TIP, THUMB_TIP
from ..core.lifting import HandModel, PalmLifter
from ..core.tracker import HandTracker
from ..core.video import add_source_args, open_source


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    add_source_args(p, stride=1)
    p.add_argument("--low-pct", type=float, default=5.0)
    p.add_argument("--high-pct", type=float, default=95.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    lifter = PalmLifter.from_config(cfg, HandModel.load(cfg.resolve(cfg.paths.hand_model)), intr)
    ds = []
    with HandTracker.from_config(cfg) as tracker, open_source(a) as src:
        for frame, t, _ in src:
            det = tracker.detect(frame, t)
            if det is None:
                continue
            r = lifter.lift(det, t)
            if r is not None and r.reproj_px <= cfg.lifting.max_reproj_px:
                ds.append(np.linalg.norm(r.kp3d_cam[THUMB_TIP] - r.kp3d_cam[INDEX_TIP]))
    if len(ds) < 30:
        raise SystemExit(f"유효 프레임 {len(ds)}개: 너무 적음")
    ds = np.array(ds)
    lo, hi = np.percentile(ds, a.low_pct), np.percentile(ds, a.high_pct)
    out = cfg.resolve(a.out or cfg.paths.gripper_range)
    out.write_text(json.dumps({"d_closed": float(lo), "d_open": float(hi), "n": int(len(ds)),
                               "pct": [a.low_pct, a.high_pct]}, indent=2))
    print(f"엄지–검지 거리: 닫힘 {lo * 100:.1f}cm, 열림 {hi * 100:.1f}cm (n={len(ds)})\n저장: {out}")


if __name__ == "__main__":
    main()
