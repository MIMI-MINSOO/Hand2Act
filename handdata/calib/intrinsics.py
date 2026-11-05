"""ChArUco로 카메라 intrinsic 캘리브레이션.

    python -m handdata.calib.intrinsics --video calib_raw/intrinsic.mov

촬영 팁: 실제 수집과 같은 해상도·방향·렌즈(1x)로, 보드를 화면 곳곳(가장자리 포함)과 여러 기울기로.
iPhone은 영상 안정화(OIS/EIS)와 자동 줌 전환을 끄고 촬영 (켜져 있으면 K가 프레임마다 달라짐).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget
from ..core.camera import Intrinsics
from ..core.video import add_source_args, open_source


def calibrate(objs, imgs, size, flags=0):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [o.astype(np.float32).reshape(-1, 1, 3) for o in objs],
        [i.astype(np.float32).reshape(-1, 1, 2) for i in imgs], size, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, axis=1)))))
    return rms, K, dist.reshape(-1), np.array(per_view)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    add_source_args(p, stride=10)
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--max-views", type=int, default=60)
    p.add_argument("--fix-k3", action="store_true", help="k3=0 고정 (왜곡 작은 렌즈에서 과적합 방지)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    board = CharucoTarget(cfg.board)
    objs, imgs, size = [], [], None
    with open_source(a) as src:
        for frame, _, idx in src:
            size = (frame.shape[1], frame.shape[0])
            found = board.detect(frame)
            if found is not None and len(found[0]) >= a.min_corners:
                objs.append(found[0]); imgs.append(found[1])
    print(f"보드 검출된 view: {len(objs)}")
    if len(objs) < 8:
        raise SystemExit("view가 너무 적음 (8장 이상 필요). 보드를 더 다양하게 찍어주세요.")
    if len(objs) > a.max_views:
        sel = np.linspace(0, len(objs) - 1, a.max_views).astype(int)
        objs, imgs = [objs[i] for i in sel], [imgs[i] for i in sel]

    flags = cv2.CALIB_FIX_K3 if a.fix_k3 else 0
    rms, K, dist, per_view = calibrate(objs, imgs, size, flags)
    bad = per_view > 3 * np.median(per_view)
    if bad.any() and (~bad).sum() >= 8:                       # 튀는 view 제거 후 1회 재보정
        objs = [o for o, b in zip(objs, bad) if not b]
        imgs = [i for i, b in zip(imgs, bad) if not b]
        print(f"오차 큰 view {bad.sum()}개 제거 후 재보정")
        rms, K, dist, per_view = calibrate(objs, imgs, size, flags)

    intr = Intrinsics(K, dist, size[0], size[1], float(rms),
                      {"n_views": len(objs), "per_view_median_px": float(np.median(per_view)),
                       "source": a.video or a.images or a.camera, "fix_k3": a.fix_k3})
    out = cfg.resolve(a.out or cfg.paths.intrinsics)
    intr.save(out)
    np.set_printoptions(precision=3, suppress=True)
    print(f"RMS {rms:.3f}px (view별 median {np.median(per_view):.3f}px), {size[0]}x{size[1]}")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}\ndist={dist}")
    print(f"저장: {out}")
    if rms > 1.0:
        print("⚠ RMS > 1px: 보드 실측 크기, 평탄도, 흔들림(모션블러)을 확인하세요.")


if __name__ == "__main__":
    main()
