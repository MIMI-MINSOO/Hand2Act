"""Evaluate mono-A, mono-B and stereo from one recorded RGB/detection cache.

The recording is produced by ``stereo_live --record-dir``. Absolute accuracy
is reported only when an independent ``--gt`` NPZ is supplied; otherwise the
report uses stability, consistency and cross-method disagreement terminology.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.camera import Intrinsics
from ..core.landmarks import CONNECTIONS
from ..core.lifting import HandModel
from ..core.stereo import StereoCalibration
from ..core.stereo_evaluation import (MethodTrajectory, evaluate_mono, evaluate_stereo,
                                      summarize_methods)
from ..core.tracker import HandTracker


def _load_recording(root: Path):
    pose_path = root / "stereo_pose.npz"
    if not pose_path.exists():
        raise SystemExit(f"recording metadata missing: {pose_path}")
    z = np.load(pose_path, allow_pickle=False)
    files0 = sorted((root / "cam0").glob("*.jpg"))
    files1 = sorted((root / "cam1").glob("*.jpg"))
    n = min(len(files0), len(files1), len(z["sensor_timestamp_ms_cam0"]))
    if n == 0:
        raise SystemExit("recording has no complete RGB pairs")
    shared = False
    if 'timestamp_domain_cam0' in z.files and 'timestamp_domain_cam1' in z.files:
        d0, d1 = z['timestamp_domain_cam0'][:n], z['timestamp_domain_cam1'][:n]
        shared = (np.array_equal(d0, d1) and all(str(d).split('.')[-1] in
                  {'system_time', 'global_time'} for d in d0))
    key, scale = ('sensor_timestamp_ms', 1000.) if shared else ('host_received_ns', 1e9)
    t0 = np.asarray(z[f'{key}_cam0'][:n], float) / scale
    t1 = np.asarray(z[f'{key}_cam1'][:n], float) / scale
    return files0[:n], files1[:n], t0, t1, z


def _detect_or_load(cfg, root, files0, files1, t0, t1, reuse):
    path = root / "stereo_detections.npz"
    if reuse and path.exists():
        z = np.load(path, allow_pickle=False)
        return {key: z[key] for key in z.files}, path
    n = len(files0)
    cache = {"t0": t0, "t1": t1,
             "found0": np.zeros(n, bool), "found1": np.zeros(n, bool),
             "uv0": np.full((n, 21, 2), np.nan), "uv1": np.full((n, 21, 2), np.nan),
             "uvz0": np.full((n, 21, 3), np.nan), "uvz1": np.full((n, 21, 3), np.nan),
             "world0": np.full((n, 21, 3), np.nan), "world1": np.full((n, 21, 3), np.nan),
             "score0": np.full(n, np.nan), "score1": np.full(n, np.nan),
             "side0": np.full(n, "", dtype="<U5"), "side1": np.full(n, "", dtype="<U5"),
             "side_matched0": np.zeros(n, bool), "side_matched1": np.zeros(n, bool),
             "detection_ms0": np.full(n, np.nan), "detection_ms1": np.full(n, np.nan)}
    # VIDEO mode requires independent monotonically increasing clocks.
    local0, local1 = t0 - t0[0], t1 - t1[0]
    with HandTracker.from_config(cfg) as tracker0, HandTracker.from_config(cfg) as tracker1:
        for i, (p0, p1) in enumerate(zip(files0, files1)):
            images = (cv2.imread(str(p0)), cv2.imread(str(p1)))
            if images[0] is None or images[1] is None:
                continue
            for camera, (tracker, image, timestamp) in enumerate(
                    ((tracker0, images[0], local0[i]), (tracker1, images[1], local1[i]))):
                started = time.perf_counter_ns()
                det = tracker.detect(image, float(timestamp))
                cache[f"detection_ms{camera}"][i] = (time.perf_counter_ns() - started) / 1e6
                if det is None:
                    continue
                cache[f"found{camera}"][i] = True
                for key in ("uv", "uvz", "world", "score", "side", "side_matched"):
                    cache[f"{key}{camera}"][i] = getattr(det, key)
    np.savez_compressed(path, **cache)
    return cache, path


def _experiments(n, default, segments_path):
    labels = np.full(n, default, dtype="<U16")
    if not segments_path:
        return labels
    with Path(segments_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            start, end = int(row["start_frame"]), int(row["end_frame"])
            labels[max(start, 0):min(end + 1, n)] = row["experiment"]
    return labels


def _lifter_options(cfg):
    L = cfg.lifting
    return {"pnp_method": L.pnp_method, "refine_lm": L.refine_lm,
            "finger_method": L.finger_method, "max_reproj_px": L.max_reproj_px,
            "prev_max_gap_sec": L.prev_max_gap_sec,
            "flip_err_ratio": L.get("flip_err_ratio", 3.0)}


def _write_frames_csv(path, experiments, t0, t1, methods):
    fields = ["frame", "experiment", "t_cam0_s", "t_cam1_s", "signed_offset_ms",
              "pair_gap_ms"]
    for name in methods:
        fields += [f"{name}_valid", f"{name}_valid_joints", f"{name}_reproj_px",
                   f"{name}_palm_x_m", f"{name}_palm_y_m", f"{name}_palm_z_m",
                   f"{name}_pinch_m"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(t0)):
            row = {"frame": i, "experiment": experiments[i], "t_cam0_s": t0[i],
                   "t_cam1_s": t1[i], "signed_offset_ms": (t1[i] - t0[i]) * 1000,
                   "pair_gap_ms": abs(t1[i] - t0[i]) * 1000}
            for name, method in methods.items():
                row.update({f"{name}_valid": int(method.valid[i]),
                            f"{name}_valid_joints": method.valid_joints[i],
                            f"{name}_reproj_px": method.reproj_px[i],
                            f"{name}_palm_x_m": method.palm_position_cam0[i, 0],
                            f"{name}_palm_y_m": method.palm_position_cam0[i, 1],
                            f"{name}_palm_z_m": method.palm_position_cam0[i, 2],
                            f"{name}_pinch_m": method.pinch_distance_m[i]})
            writer.writerow(row)


def _plot_summary(path, methods, gap_ms):
    canvas = np.full((720, 1200, 3), 248, np.uint8)
    colors = {"B1_A": (220, 80, 60), "B1_B": (60, 150, 220),
              "B2_nearest": (50, 170, 60), "B2_interpolated": (160, 80, 180)}
    cv2.putText(canvas, "Palm trajectories in cam0 (x/y/z vs frame)", (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    n = len(gap_ms)
    for axis in range(3):
        y0, y1 = 70 + axis * 160, 200 + axis * 160
        cv2.rectangle(canvas, (60, y0), (1140, y1), (180, 180, 180), 1)
        all_values = np.concatenate([m.palm_position_cam0[:, axis] for m in methods.values()])
        finite = all_values[np.isfinite(all_values)]
        lo, hi = ((float(np.min(finite)), float(np.max(finite))) if len(finite) else (0., 1.))
        if hi <= lo:
            hi = lo + 1
        for name, method in methods.items():
            points = []
            for i, value in enumerate(method.palm_position_cam0[:, axis]):
                if np.isfinite(value):
                    points.append((int(60 + i * 1080 / max(n - 1, 1)),
                                   int(y1 - (value - lo) * (y1 - y0) / (hi - lo))))
                elif len(points) > 1:
                    cv2.polylines(canvas, [np.asarray(points)], False, colors[name], 2)
                    points = []
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points)], False, colors[name], 2)
        cv2.putText(canvas, f"{'xyz'[axis]} [{lo:.3f}, {hi:.3f}] m", (65, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 40, 40), 1)
    cv2.putText(canvas, "Valid ratio", (60, 590), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2)
    for j, (name, method) in enumerate(methods.items()):
        ratio = float(method.valid.mean())
        x = 60 + j * 270
        cv2.rectangle(canvas, (x, 620), (x + int(220 * ratio), 655), colors[name], -1)
        cv2.putText(canvas, f"{name}: {ratio:.1%}", (x, 680), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (20, 20, 20), 1)
    cv2.imwrite(str(path), canvas)


def _write_video(path, files0, files1, cache, methods, fps):
    first = cv2.imread(str(files0[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    for i, (p0, p1) in enumerate(zip(files0, files1)):
        a, b = cv2.imread(str(p0)), cv2.imread(str(p1))
        if a is None or b is None:
            continue
        for image, camera in ((a, 0), (b, 1)):
            if cache[f"found{camera}"][i]:
                uv = np.asarray(cache[f"uv{camera}"][i], int)
                for u, v in CONNECTIONS:
                    cv2.line(image, tuple(uv[u]), tuple(uv[v]), (0, 230, 255), 2)
                for point in uv:
                    cv2.circle(image, tuple(point), 2, (255, 255, 255), -1)
        panel = np.hstack([a, b])
        y = 20
        for name, method in methods.items():
            pos = method.palm_position_cam0[i]
            text = f"{name}: " + (np.array2string(pos, precision=3) + " m" if method.valid[i] else "INVALID")
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (80, 255, 80) if method.valid[i] else (80, 80, 255), 1, cv2.LINE_AA)
            y += 19
        writer.write(panel)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-detections", action="store_true")
    parser.add_argument("--experiment", choices=["static", "dynamic", "pinch", "occlusion"],
                        default="static")
    parser.add_argument("--segments", help="CSV: start_frame,end_frame,experiment")
    parser.add_argument("--gt", help="optional independent NPZ with palm_position_cam0")
    parser.add_argument("--max-interp-gap-ms", type=float, default=50.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    stereo = cfg.stereo
    calibration_path = cfg.resolve(stereo.extrinsics)
    if not calibration_path.exists():
        raise SystemExit(f"verified stereo calibration missing: {calibration_path}")
    calibration = StereoCalibration.load(calibration_path)
    if not calibration.metadata.get("board", {}).get("physical_verified", False):
        raise SystemExit("stereo calibration is not marked as physically verified")
    if (calibration.serial_cam0, calibration.serial_cam1) != (
            str(stereo.serial_cam0), str(stereo.serial_cam1)):
        raise SystemExit("stereo calibration serial assignment does not match config")
    intr0, intr1 = (Intrinsics.load(cfg.resolve(stereo.intrinsics_cam0)),
                    Intrinsics.load(cfg.resolve(stereo.intrinsics_cam1)))
    model_path = cfg.resolve(cfg.paths.hand_model)
    if not model_path.exists():
        raise SystemExit(f"monocular hand model missing: {model_path}")
    model = HandModel.load(model_path)
    root, out = cfg.resolve(args.recording), cfg.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files0, files1, t0, t1, recording = _load_recording(root)
    cache, cache_path = _detect_or_load(cfg, root, files0, files1, t0, t1,
                                        args.reuse_detections)
    options = _lifter_options(cfg)
    methods = {
        "B1_A": evaluate_mono(cache, 0, t0, intr0, model, calibration,
                               cfg.lifting.max_reproj_px, options),
        "B1_B": evaluate_mono(cache, 1, t1, intr1, model, calibration,
                               cfg.lifting.max_reproj_px, options),
        "B2_nearest": evaluate_stereo(
            cache, t0, t1, intr0, intr1, calibration, interpolated=False,
            max_interp_gap_sec=args.max_interp_gap_ms / 1000,
            max_reproj_px=stereo.max_reproj_px,
            min_ray_angle_deg=stereo.min_ray_angle_deg,
            bone_rel_tolerance=stereo.bone_rel_tolerance),
        "B2_interpolated": evaluate_stereo(
            cache, t0, t1, intr0, intr1, calibration, interpolated=True,
            max_interp_gap_sec=args.max_interp_gap_ms / 1000,
            max_reproj_px=stereo.max_reproj_px,
            min_ray_angle_deg=stereo.min_ray_angle_deg,
            bone_rel_tolerance=stereo.bone_rel_tolerance),
    }
    experiments = _experiments(len(t0), args.experiment, args.segments)
    gt = gt_quat = gt_points = gt_pinch = None
    if args.gt:
        gt_file = np.load(cfg.resolve(args.gt), allow_pickle=False)
        gt = np.asarray(gt_file["palm_position_cam0"], float)
        if len(gt) != len(t0):
            raise SystemExit("GT frame count does not match recording")
        gt_quat = (np.asarray(gt_file["palm_quaternion_xyzw"], float)
                   if "palm_quaternion_xyzw" in gt_file.files else None)
        gt_points = (np.asarray(gt_file["points_cam0"], float)
                     if "points_cam0" in gt_file.files else None)
        gt_pinch = (np.asarray(gt_file["pinch_distance_m"], float)
                    if "pinch_distance_m" in gt_file.files else None)
    summary = summarize_methods(methods, experiments, gt, gt_quat, gt_points, gt_pinch)
    summary["detection_2d"] = {
        "metric_note": "availability/confidence only; no labeled 2D GT was supplied",
        "cam0_found_ratio": float(np.mean(cache["found0"])),
        "cam1_found_ratio": float(np.mean(cache["found1"])),
        "both_found_ratio": float(np.mean(cache["found0"] & cache["found1"])),
        "cam0_score_median": (float(np.nanmedian(cache["score0"]))
                              if np.isfinite(cache["score0"]).any() else None),
        "cam1_score_median": (float(np.nanmedian(cache["score1"]))
                              if np.isfinite(cache["score1"]).any() else None),
    }
    signed_ms = (t1 - t0) * 1000
    gap_ms = np.abs(signed_ms)
    duration = max(float(max(t0[-1] - t0[0], t1[-1] - t1[0])), 1e-9)
    frame0 = np.asarray(recording["frame_number_cam0"][:len(t0)], int)
    frame1 = np.asarray(recording["frame_number_cam1"][:len(t0)], int)
    live_processing = (np.asarray(recording["processing_latency_ms"][:len(t0)], float)
                       if "processing_latency_ms" in recording.files else None)
    summary["timing"] = {
        "timestamp_semantics": "shared system/global domain when recorded; otherwise host receive; exposure unknown",
        "n_pairs": len(t0), "paired_output_fps": float((len(t0) - 1) / duration),
        "selected_frame_skips_cam0": int(np.maximum(np.diff(frame0) - 1, 0).sum()),
        "selected_frame_skips_cam1": int(np.maximum(np.diff(frame1) - 1, 0).sum()),
        "signed_offset_median_ms": float(np.median(signed_ms)),
        "signed_offset_p05_ms": float(np.percentile(signed_ms, 5)),
        "signed_offset_p95_ms": float(np.percentile(signed_ms, 95)),
        "pair_gap_median_ms": float(np.median(gap_ms)),
        "pair_gap_p95_ms": float(np.percentile(gap_ms, 95)),
        "pair_gap_max_ms": float(np.max(gap_ms)),
        "detection_processing_ms_cam0_median": float(np.nanmedian(cache["detection_ms0"])),
        "detection_processing_ms_cam1_median": float(np.nanmedian(cache["detection_ms1"])),
        "live_processing_ms_median": (float(np.nanmedian(live_processing))
                                      if live_processing is not None else None),
        "latency_note": "offline detector processing time; not live end-to-end latency",
    }
    summary["inputs"] = {"recording": str(root), "detection_cache": str(cache_path),
                         "extrinsics": str(calibration_path), "gt": args.gt,
                         "coordinate_frame": "cam0 (D435i), not robot base"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_frames_csv(out / "frames.csv", experiments, t0, t1, methods)
    arrays = {"t_cam0_s": t0, "t_cam1_s": t1, "experiment": experiments}
    for name, method in methods.items():
        arrays.update({f"{name}_points_cam0": method.points_cam0,
                       f"{name}_palm_position_cam0": method.palm_position_cam0,
                       f"{name}_palm_quaternion_xyzw": method.palm_quaternion_xyzw,
                       f"{name}_pinch_distance_m": method.pinch_distance_m,
                       f"{name}_valid": method.valid})
    np.savez_compressed(out / "trajectories.npz", **arrays)
    _plot_summary(out / "comparison.png", methods, gap_ms)
    if args.video:
        _write_video(out / "comparison.mp4", files0, files1, cache, methods,
                     max(summary["timing"]["paired_output_fps"], 1.0))
    print(f"saved evaluation: {out}")
    print(json.dumps(summary["timing"], indent=2))


if __name__ == "__main__":
    main()
