"""녹화 영상 → 로봇 데이터 에피소드 (npz + json).

    python -m handdata.apps.offline --video data/raw/ep001.mov --out data/ep001 --preview

Stage 1 (느림) : MediaPipe 손 검출 + ChArUco 보드 pose → detections.npz 캐시
Stage 2 (빠름) : PnP lifting → world(base) 변환 → EE pose·그리퍼 → episode.npz
lifting 설정만 바꿔 다시 돌릴 때(ablation)는 --reuse-detections 로 Stage 1 생략:

    python -m handdata.apps.offline --video ... --out data/ep001 --reuse-detections \
        --set lifting.finger_method=ray_bone --name episode_raybone.npz
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
from ..core.lifting import HandModel
from ..core.tracker import HandTracker
from ..core.video import FrameSource
from ..dataio.episode import (EpisodeBuffer, flag_summary, load_detections, save_detections,
                              save_episode)
from ..pipeline import FrameProcessor
from ..viz import draw_frame


def run_detection(cfg, intr, video, stride, max_frames):
    board = CharucoTarget(cfg.board)
    ts, dets, boards = [], [], []
    t0 = time.perf_counter()
    with HandTracker.from_config(cfg) as tracker, FrameSource(video, stride, max_frames) as src:
        for frame, t, idx in src:
            if not ts:
                intr.check_frame(frame)
            dets.append(tracker.detect(frame, t))
            boards.append(board.pose(frame, intr))
            ts.append(t)
            if len(ts) % 100 == 0:
                print(f"  {len(ts)} / {src.n_frames // stride} frames  ({len(ts) / (time.perf_counter() - t0):.1f} fps)")
        fps = src.fps
    return ts, dets, boards, {"video": str(video), "stride": stride, "fps": fps}


def process(cfg, intr, model, ts, dets, boards) -> EpisodeBuffer:
    proc = FrameProcessor(cfg, intr, model)
    if not proc.gripper_calibrated:
        print("⚠ gripper_range.json 없음 → 기본 범위 사용 (calib.gripper_range 실행 권장)")
    buf = EpisodeBuffer()
    for t, d, b in zip(ts, dets, boards):
        buf.add(proc.process(float(t), d, b))
    return buf


def write_preview(intr, model, video, stride, rows, out_path):
    with FrameSource(video, stride, len(rows)) as src:
        h, w = intr.height, intr.width
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src.fps / stride, (w, h))
        for (frame, _, _), r in zip(src, rows):
            vw.write(draw_frame(frame, intr, r, model.palm))
        vw.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="에피소드 폴더")
    p.add_argument("--name", default="episode.npz")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--reuse-detections", action="store_true")
    p.add_argument("--preview", action="store_true", help="오버레이 영상 preview.mp4 저장")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    model = HandModel.load(cfg.resolve(cfg.paths.hand_model))
    out = Path(a.out)
    det_path = out / "detections.npz"

    if a.reuse_detections and det_path.exists():
        ts, dets, boards, dmeta = load_detections(det_path)
        print(f"[1/2] 검출 캐시 사용: {det_path} ({len(ts)} frames)")
    else:
        print("[1/2] MediaPipe + ChArUco 검출")
        ts, dets, boards, dmeta = run_detection(cfg, intr, a.video, a.stride, a.max_frames)
        save_detections(det_path, ts, dets, boards, dmeta)

    print("[2/2] lifting + world 변환 + retarget")
    buf = process(cfg, intr, model, ts, dets, boards)
    arrays = buf.to_arrays()
    save_episode(out, arrays, {"source": dmeta, "config": cfg.to_plain(),
                               "hand_model": {"method": model.method, "wrist_middle_m": model.wrist_middle,
                                              "h_offset": model.h_offset},
                               "intrinsics": {"rms": intr.rms, "size": [intr.width, intr.height]}}, a.name)

    v, rp = arrays["valid"], arrays["reproj_px"]
    print(f"\n유효 프레임 {v.sum()}/{len(v)} ({100 * v.mean():.1f}%)  "
          f"재투영 median {np.nanmedian(rp):.2f}px")
    print("플래그:", {k: n for k, n in flag_summary(arrays["flags"]).items() if n})
    print(f"저장: {out / a.name}")

    if a.preview:
        write_preview(intr, model, dmeta["video"] if a.reuse_detections else a.video,
                      int(dmeta["stride"]), buf.rows, out / "preview.mp4")
        print(f"preview: {out / 'preview.mp4'}")


if __name__ == "__main__":
    main()
