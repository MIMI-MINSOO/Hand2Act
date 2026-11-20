"""Interactive two-camera ChArUco capture for fixed stereo calibration.

Keys: ``s``/space saves a pair, ``q``/escape exits. The board must be stationary
within each pair because equal timestamp domains do not prove exposure sync.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget
from ..core.realsense_stereo import DualRealSense
from ..core.stereo import match_common_charuco


def _overlay(image, found, label, blur_score):
    out = image.copy()
    if found is not None:
        _, points, ids = found
        for point, corner_id in zip(points.astype(int), ids):
            cv2.circle(out, tuple(point), 4, (0, 255, 0), -1)
            cv2.putText(out, str(int(corner_id)), tuple(point + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{label} corners={0 if found is None else len(found[2])} blur={blur_score:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _motion_px(previous, current):
    if previous is None or current is None:
        return 0.0
    prev = {int(i): p for p, i in zip(previous[1], previous[2])}
    cur = {int(i): p for p, i in zip(current[1], current[2])}
    common = sorted(set(prev) & set(cur))
    return float(np.median([np.linalg.norm(cur[i] - prev[i]) for i in common])) if common else 0.0


def _next_pair_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (root / f"pair_{index:04d}").exists():
        index += 1
    return root / f"pair_{index:04d}"


def save_pair(root, packet0, packet1, gap_ms, gap_clock, found0, found1,
              common_count, blur, motion, cfg, intrinsics):
    pair_dir = _next_pair_dir(Path(root))
    pair_dir.mkdir(parents=False, exist_ok=False)
    paths = [pair_dir / "cam0.png", pair_dir / "cam1.png"]
    if not cv2.imwrite(str(paths[0]), packet0.image_bgr):
        raise IOError(f"failed to save {paths[0]}")
    if not cv2.imwrite(str(paths[1]), packet1.image_bgr):
        raise IOError(f"failed to save {paths[1]}")
    meta = {
        "serial_cam0": packet0.serial, "serial_cam1": packet1.serial,
        "intrinsics_cam0": str(intrinsics[0]), "intrinsics_cam1": str(intrinsics[1]),
        "frames": [
            {"frame_number": packet0.frame_number, "sensor_timestamp_ms": packet0.sensor_timestamp_ms,
             "timestamp_domain": packet0.timestamp_domain, "host_received_ns": packet0.host_received_ns},
            {"frame_number": packet1.frame_number, "sensor_timestamp_ms": packet1.sensor_timestamp_ms,
             "timestamp_domain": packet1.timestamp_domain, "host_received_ns": packet1.host_received_ns},
        ],
        "pair_gap_ms": gap_ms, "pair_gap_clock": gap_clock,
        "detected_ids_cam0": [] if found0 is None else found0[2].astype(int).tolist(),
        "detected_ids_cam1": [] if found1 is None else found1[2].astype(int).tolist(),
        "common_id_count": common_count, "blur_laplacian": blur,
        "board_motion_px_since_previous": motion,
        "board": cfg.to_plain()["board"],
        "warning": "Keep board stationary within each pair; timestamps are not proven exposure sync.",
    }
    temp = pair_dir / "metadata.json.tmp"
    temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temp.replace(pair_dir / "metadata.json")
    return pair_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--out", default="calib_raw/stereo")
    parser.add_argument("--min-common", type=int, default=8)
    parser.add_argument("--blur-threshold", type=float, default=60.0)
    parser.add_argument("--motion-threshold-px", type=float, default=4.0)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    stereo = cfg.stereo
    serials = (str(stereo.serial_cam0), str(stereo.serial_cam1))
    intrinsics = (cfg.resolve(stereo.intrinsics_cam0), cfg.resolve(stereo.intrinsics_cam1))
    target = CharucoTarget(cfg.board)
    previous = [None, None]
    saved = 0
    with DualRealSense(*serials, stereo.width, stereo.height, stereo.fps) as source:
        while True:
            packet0, packet1, gap_ms, gap_clock = source.read_nearest(
                lookahead_frames=stereo.get("pair_lookahead_frames", 0))
            found0 = target.detect_with_ids(packet0.image_bgr)
            found1 = target.detect_with_ids(packet1.image_bgr)
            common = match_common_charuco(found0, found1, args.min_common) if found0 and found1 else None
            common_count = 0 if common is None else len(common[3])
            blur = [float(cv2.Laplacian(p.image_bgr, cv2.CV_64F).var()) for p in (packet0, packet1)]
            motion = [_motion_px(previous[i], f) for i, f in enumerate((found0, found1))]
            previous = [found0, found1]
            views = [_overlay(packet0.image_bgr, found0, "cam0", blur[0]),
                     _overlay(packet1.image_bgr, found1, "cam1", blur[1])]
            panel = np.hstack(views)
            warnings = []
            if min(blur) < args.blur_threshold:
                warnings.append("BLUR")
            if max(motion) > args.motion_threshold_px:
                warnings.append("BOARD MOVING")
            if common_count < args.min_common:
                warnings.append("TOO FEW COMMON IDS")
            if gap_ms > float(stereo.max_pair_gap_ms):
                warnings.append("LARGE SYNC GAP")
            lines = [f"common={common_count} gap={gap_ms:.2f}ms({gap_clock}) saved={saved}",
                     f"fps=({source.measured_fps(0):.1f},{source.measured_fps(1):.1f})  s/space=save q=quit",
                     " ".join(warnings)]
            for row, text in enumerate(lines):
                cv2.putText(panel, text, (10, 455 - 25 * (2 - row)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if row == 2 and warnings else (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Hand2Act stereo calibration capture", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("s"), 32):
                if common_count < args.min_common:
                    print(f"not saved: only {common_count} common IDs")
                    continue
                try:
                    path = save_pair(cfg.resolve(args.out), packet0, packet1, gap_ms, gap_clock,
                                     found0, found1, common_count, blur, motion, cfg, intrinsics)
                    saved += 1
                    print(f"saved {path}")
                except Exception as exc:
                    print(f"save failed (existing pairs preserved): {exc}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
