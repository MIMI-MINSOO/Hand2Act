"""Live MediaPipe×2 stereo hand pose in the cam0 coordinate frame.

This command requires a verified ``stereo_extrinsics.json``. It never labels
cam0 coordinates as robot-base coordinates. Keys: q/escape quit, i/k/j/l rotate
the 3D view.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.camera import Intrinsics
from ..core.landmarks import CONNECTIONS
from ..core.realsense_stereo import DualRealSense
from ..core.stereo import (StereoCalibration, stereo_hand_pose,
                           triangulate_landmarks)
from ..core.tracker import HandTracker
from ..core.stereo_sync import FrameSynchronizer, interpolate_detection
from ..robot.retarget import load_gripper_range
from ..viz import put_lines


def _draw_2d(image, detection, color=(0, 255, 255)):
    out = image.copy()
    if detection is None:
        return out
    points = detection.uv.astype(int)
    for a, b in CONNECTIONS:
        cv2.line(out, tuple(points[a]), tuple(points[b]), color, 2)
    for point in points:
        cv2.circle(out, tuple(point), 3, (255, 255, 255), -1)
    return out


def _draw_3d(points, valid, yaw, pitch, size=(640, 480)):
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    if points is None or not np.isfinite(points).any():
        return canvas
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
    P = np.asarray(points) @ R.T
    center = np.nanmean(P[valid], axis=0) if valid.any() else np.zeros(3)
    P = (P - center) * 1800
    uv = np.c_[size[0] / 2 + P[:, 0], size[1] / 2 - P[:, 1]].astype(int)
    for a, b in CONNECTIONS:
        if valid[a] and valid[b]:
            cv2.line(canvas, tuple(uv[a]), tuple(uv[b]), (0, 220, 255), 2)
    for i, point in enumerate(uv):
        if valid[i]:
            cv2.circle(canvas, tuple(point), 4, (255, 255, 255), -1)
    cv2.putText(canvas, "3D cam0 frame (j/l yaw, i/k pitch)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (180, 255, 180), 1, cv2.LINE_AA)
    return canvas


class StereoRecorder:
    def __init__(self, root):
        self.root = Path(root) if root else None
        self.rows = []
        self.raw_rows = []
        self.sync_stats = {}
        if self.root:
            if any(self.root.glob('*.npz')):
                raise FileExistsError(f'refusing to overwrite recording: {self.root}')
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "cam0").mkdir(exist_ok=True)
            (self.root / "cam1").mkdir(exist_ok=True)

    def add_raw(self, camera, packet, detection):
        if self.root is None:
            return
        self.raw_rows.append({
            'camera': camera, 'frame_number': packet.frame_number,
            'sensor_timestamp_ms': packet.sensor_timestamp_ms,
            'host_received_ns': packet.host_received_ns,
            'timestamp_domain': packet.timestamp_domain,
            'sensor_metadata': json.dumps(packet.metadata),
            'uv': np.full((21, 2), np.nan) if detection is None else detection.uv.copy(),
            'side': '' if detection is None else detection.side,
            'side_matched': False if detection is None else detection.side_matched,
        })

    def add(self, index, packet0, packet1, gap_ms, tri, pose, processing_latency_ms,
            sync_info=None):
        if self.root is None:
            return
        for camera, packet in enumerate((packet0, packet1)):
            ok, encoded = cv2.imencode('.jpg', packet.image_bgr)
            if not ok:
                raise IOError('RGB encoding failed')
            (self.root / f'cam{camera}' / f'{index:06d}.jpg').write_bytes(encoded.tobytes())
        self.rows.append({
            "host_received_ns_cam0": packet0.host_received_ns,
            "host_received_ns_cam1": packet1.host_received_ns,
            "sensor_timestamp_ms_cam0": packet0.sensor_timestamp_ms,
            "sensor_timestamp_ms_cam1": packet1.sensor_timestamp_ms,
            'timestamp_domain_cam0': packet0.timestamp_domain,
            'timestamp_domain_cam1': packet1.timestamp_domain,
            "frame_number_cam0": packet0.frame_number, "frame_number_cam1": packet1.frame_number,
            "pair_gap_ms": gap_ms,
            "processing_latency_ms": processing_latency_ms,
            **(sync_info or {}),
            "points_cam0": np.full((21, 3), np.nan) if tri is None else tri.points_cam0,
            "joint_valid": np.zeros(21, bool) if tri is None else tri.valid,
            "joint_reason": np.array(["not_triangulated"] * 21 if tri is None
                                     else ["|".join(reason) for reason in tri.reasons]),
            "palm_position_cam0": np.full(3, np.nan) if pose is None else pose.palm_position_cam0,
            "palm_quaternion_xyzw": np.full(4, np.nan) if pose is None else pose.palm_quaternion_xyzw,
            "pinch_distance_m": np.nan if pose is None else pose.pinch_distance_m,
            "pose_valid": False if pose is None else pose.valid,
        })

    def close(self, metadata):
        if self.root is None:
            return
        if self.rows:
            keys = self.rows[0]
            arrays = {key: np.asarray([row[key] for row in self.rows]) for key in keys}
            np.savez_compressed(self.root / "stereo_pose.npz", **arrays)
        if self.raw_rows:
            np.savez_compressed(self.root / 'raw_landmarks.npz', **{
                k: np.asarray([r[k] for r in self.raw_rows]) for k in self.raw_rows[0]})
        metadata['sync_stats'] = self.sync_stats
        if self.rows:
            metadata['host_receive_to_output_ms_median'] = float(np.median([
                row['host_receive_to_output_ms'] for row in self.rows]))
            metadata['sync_wait_ms_median'] = float(np.median([
                row['sync_wait_ms'] for row in self.rows]))
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def aligned_detections(source, trackers, mode, recorder, max_wait_ms=60., max_span_ms=50.):
    """Preserve baseline consumption; improved modes consume raw events once."""
    if mode == 'nearest':
        while True:
            a, b, gap, clock = source.read_nearest()
            d0 = trackers[0].detect(a.image_bgr, a.sensor_timestamp_ms / 1000)
            d1 = trackers[1].detect(b.image_bgr, b.sensor_timestamp_ms / 1000)
            recorder.add_raw(0, a, d0)
            recorder.add_raw(1, b, d1)
            yield a, b, d0, d1, gap, clock, {'interpolated': False,
                'sync_wait_ms': 0., 'interpolation_reason': '',
                'frame_left': b.frame_number, 'frame_right': b.frame_number,
                'timestamp_left_ms': b.sensor_timestamp_ms,
                'timestamp_right_ms': b.sensor_timestamp_ms, 'alpha': 0.}
    aligner = FrameSynchronizer(mode, max_wait_ms, max_span_ms)
    detections = {}
    while True:
        camera, packet = source.read_event()
        det = trackers[camera].detect(packet.image_bgr, packet.sensor_timestamp_ms / 1000)
        detections[camera, packet.frame_number] = det
        recorder.add_raw(camera, packet, det)
        for pair in aligner.push(camera, packet, time.monotonic_ns()):
            recorder.sync_stats = {'selected_unique': [len(x) for x in aligner.used],
                'skipped': aligner.skipped, 'expired_cam0': aligner.expired,
                'endpoint_reuses': aligner.reuses}
            a, l, r = pair.reference, pair.left, pair.right
            d0 = detections.get((0, a.frame_number))
            dl, dr = detections.get((1, l.frame_number)), detections.get((1, r.frame_number))
            d1, reason = (interpolate_detection(pair, (d0, dl, dr))
                          if mode == 'interpolate' else (dl, ''))
            yield a, l, d0, d1, pair.gap_ms, pair.clock, {
                'interpolated': mode == 'interpolate' and d1 is not None,
                'sync_wait_ms': max(0., (pair.released_ns - a.host_received_ns) / 1e6),
                'interpolation_reason': reason, 'frame_left': l.frame_number,
                'frame_right': r.frame_number, 'timestamp_left_ms': l.sensor_timestamp_ms,
                'timestamp_right_ms': r.sensor_timestamp_ms, 'alpha': pair.alpha}
        if len(detections) > 192:
            for key in list(detections)[:64]:
                del detections[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--record-dir")
    parser.add_argument('--sync-mode', choices=['nearest', 'lookahead', 'interpolate'], default='nearest')
    parser.add_argument('--sync-max-wait-ms', type=float, default=60.)
    parser.add_argument('--sync-max-span-ms', type=float, default=50.)
    parser.add_argument('--duration', type=float, default=0., help='bounded hardware test seconds')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    stereo = cfg.stereo
    calibration_path = cfg.resolve(stereo.extrinsics)
    if not calibration_path.exists():
        raise SystemExit(f"verified stereo calibration missing: {calibration_path}")
    calibration = StereoCalibration.load(calibration_path)
    if (calibration.serial_cam0, calibration.serial_cam1) != (
            str(stereo.serial_cam0), str(stereo.serial_cam1)):
        raise SystemExit("stereo calibration serial assignment does not match config")
    intr0 = Intrinsics.load(cfg.resolve(stereo.intrinsics_cam0))
    intr1 = Intrinsics.load(cfg.resolve(stereo.intrinsics_cam1))
    d_closed, d_open, gripper_calibrated = load_gripper_range(
        cfg.resolve(cfg.paths.gripper_range), *cfg.gripper.default_range)
    recorder = StereoRecorder(cfg.resolve(args.record_dir) if args.record_dir else None)
    yaw, pitch, index = 0.0, 0.0, 0
    reference_bones = None
    valid_frames = 0
    started = time.monotonic()
    with HandTracker.from_config(cfg) as tracker0, HandTracker.from_config(cfg) as tracker1, \
            DualRealSense(stereo.serial_cam0, stereo.serial_cam1,
                          stereo.width, stereo.height, stereo.fps) as source:
        started = time.monotonic()
        try:
            for packet0, packet1, det0, det1, gap_ms, gap_clock, sync_info in aligned_detections(
                    source, (tracker0, tracker1), args.sync_mode, recorder,
                    args.sync_max_wait_ms, args.sync_max_span_ms):
                processing_started = time.perf_counter_ns()
                tri = pose = None
                flags = []
                if args.sync_mode != 'interpolate' and gap_ms > stereo.max_pair_gap_ms:
                    flags.append("SYNC_GAP")
                elif det0 is None or det1 is None:
                    flags.append("HAND_MISSING")
                elif det0.side != det1.side:
                    flags.append("HANDEDNESS_MISMATCH")
                else:
                    tri = triangulate_landmarks(
                        det0.uv, det1.uv, intr0, intr1, calibration,
                        stereo.max_reproj_px, stereo.min_ray_angle_deg,
                        reference_bones, stereo.bone_rel_tolerance)
                    if reference_bones is None and tri.valid.all():
                        reference_bones = np.array([
                            np.linalg.norm(tri.points_cam0[a] - tri.points_cam0[b])
                            for a, b in CONNECTIONS])
                    pose = stereo_hand_pose(
                        tri, det0.side, stereo.get("ee_offset_palm", [0, 0, 0]),
                        (d_closed, d_open) if gripper_calibrated else None)
                    flags.extend(pose.quality_flags)
                    valid_frames += int(pose.valid)
                panel = np.hstack([_draw_2d(packet0.image_bgr, det0),
                                   _draw_2d(packet1.image_bgr, det1)])
                valid_count = 0 if tri is None else int(tri.valid.sum())
                reproj = np.nan if tri is None or not tri.valid.any() else float(np.median(tri.reproj_px[tri.valid]))
                lines = [
                    f"fps=({source.measured_fps(0):.1f},{source.measured_fps(1):.1f}) gap={gap_ms:.2f}ms {gap_clock}",
                    f"detect=({det0 is not None},{det1 is not None}) joints={valid_count}/21 reproj={reproj:.2f}px",
                    "flags=" + (",".join(flags) if flags else "OK"),
                ]
                if pose is not None:
                    lines += [f"palm[cam0]={np.array2string(pose.palm_position_cam0, precision=3)}m",
                              f"quat_xyzw={np.array2string(pose.palm_quaternion_xyzw, precision=3)}",
                              f"pinch={pose.pinch_distance_m:.4f}m gripper={pose.gripper:.3f}"]
                put_lines(panel, lines, (80, 255, 80) if not flags else (80, 160, 255), scale=.5)
                if not args.headless:
                    cv2.imshow("Hand2Act stereo RGB", panel)
                    cv2.imshow("Hand2Act stereo 3D", _draw_3d(
                        None if tri is None else tri.points_cam0,
                        np.zeros(21, bool) if tri is None else tri.valid, yaw, pitch))
                recorder.add(index, packet0, packet1, gap_ms, tri, pose,
                             (time.perf_counter_ns() - processing_started) / 1e6,
                             {**sync_info, 'host_receive_to_output_ms':
                              (time.monotonic_ns() - packet0.host_received_ns) / 1e6})
                index += 1
                if args.duration and time.monotonic() - started >= args.duration:
                    break
                key = (cv2.waitKey(1) & 0xFF) if not args.headless else 255
                if key in (27, ord("q")):
                    break
                yaw += 5 * ((key == ord("l")) - (key == ord("j")))
                pitch += 5 * ((key == ord("k")) - (key == ord("i")))
        finally:
            recorder.close({
                "coordinate_frame": "cam0 (not robot base)",
                "serial_cam0": calibration.serial_cam0, "serial_cam1": calibration.serial_cam1,
                "extrinsics": str(calibration_path), "n_frames": index,
                "pair_lookahead_frames": stereo.get("pair_lookahead_frames", 0),
                'sync_mode': args.sync_mode, 'queue_drops': source.queue_drops,
                'timing_note': 'host receive to output excludes unknown exposure/USB delay',
                "n_valid_palm": valid_frames, "valid_ratio": valid_frames / max(index, 1),
                "elapsed_sec": time.monotonic() - started,
            })
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
