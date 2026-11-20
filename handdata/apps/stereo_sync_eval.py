"""Record raw RGB timing once; replay nearest/lookahead/interpolate identically.

Capture: --capture-seconds 10 --data data/sync_001.npz
Replay: --data data/sync_001.npz --out results/sync_001.json
Optional --led-roi0/1 x y w h measures a user-provided common flashing LED.
Offline replay latency is a buffering lower bound, not measured live latency.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.camera import Intrinsics
from ..core.realsense_stereo import CameraFrame, DualRealSense
from ..core.stereo import StereoCalibration, stereo_hand_pose, triangulate_landmarks
from ..core.stereo_sync import FrameSynchronizer, interpolate_detection
from ..core.tracker import HandDetection, HandTracker


def distribution(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return ({'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)),
             'max': float(np.max(x))} if len(x) else None)


def led_diagnostic(times0, signal0, times1, signal1, max_offset_ms=100.):
    """Match rising brightness edges with bounded, unique event correspondence.

    User must ensure sparse, distinct flashes. Frame-rate quantization and
    exposure/rolling shutter bound interpretation; no automatic clock correction.
    """
    def edges(t, signal):
        signal = np.asarray(signal, float)
        if len(signal) < 3 or np.ptp(signal) < 20:
            return np.array([])
        threshold = (float(np.min(signal)) + float(np.max(signal))) / 2
        return np.asarray(t)[np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1]
    a, b = edges(times0, signal0), edges(times1, signal1)
    pairs, used = [], set()
    for ta in a:
        candidates = [j for j in range(len(b)) if j not in used and
                      abs(b[j] - ta) <= max_offset_ms / 1000]
        if candidates:
            j = min(candidates, key=lambda j: abs(b[j] - ta))
            used.add(j)
            pairs.append((ta, b[j] - ta))
    if len(pairs) < 3:
        return {'matched_events': len(pairs), 'offset_ms': None, 'drift_ppm': None}
    x, y = np.asarray(pairs).T
    slope, intercept = np.polyfit(x - x[0], y, 1)
    return {'matched_events': len(pairs), 'offset_ms': float(intercept * 1000),
            'drift_ppm': float(slope * 1e6),
            'residual_ms': distribution((y - intercept - slope * (x - x[0])) * 1000),
            'uncertainty': 'frame-quantized LED onset; rolling shutter/exposure remain unknown'}


def capture(cfg, path, seconds):
    rows = []
    s = cfg.stereo
    with DualRealSense(s.serial_cam0, s.serial_cam1, s.width, s.height, s.fps,
                       queue_size=64) as source:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            camera, p = source.read_event()
            rows.append((camera, p))
        drops = source.queue_drops.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.partial.npz')
    if temporary.exists():
        raise FileExistsError(temporary)
    np.savez_compressed(temporary, camera=[c for c, p in rows],
        image=np.stack([p.image_bgr for c, p in rows]),
        serial=[p.serial for c, p in rows], frame=[p.frame_number for c, p in rows],
        sensor_ms=[p.sensor_timestamp_ms for c, p in rows],
        domain=[p.timestamp_domain for c, p in rows], host_ns=[p.host_received_ns for c, p in rows],
        metadata=[json.dumps(p.metadata) for c, p in rows], queue_drops=drops)
    temporary.rename(path)
    print(f'captured {len(rows)} raw frames; queue drops={drops}; {path}')


def replay(cfg, path, out, experiment, roi0=None, roi1=None):
    with np.load(path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    cache_path = path.with_name(path.stem + '_landmarks.npz')
    n = len(z['camera'])
    if cache_path.exists():
        cache = dict(np.load(cache_path, allow_pickle=False))
        if len(cache['uv']) != n or not np.array_equal(cache['host_ns'], z['host_ns']):
            raise ValueError('cache does not match raw recording')
    else:
        cache = {'uv': np.full((n, 21, 2), np.nan), 'side': np.full(n, '', dtype='<U5'),
                 'matched': np.zeros(n, bool), 'detect_ms': np.zeros(n), 'host_ns': z['host_ns']}
        with HandTracker.from_config(cfg) as t0, HandTracker.from_config(cfg) as t1:
            for i, c in enumerate(z['camera']):
                start = time.perf_counter_ns()
                d = (t0, t1)[c].detect(z['image'][i], (z['host_ns'][i] - z['host_ns'][0]) / 1e9)
                cache['detect_ms'][i] = (time.perf_counter_ns() - start) / 1e6
                if d is not None:
                    cache['uv'][i], cache['side'][i], cache['matched'][i] = d.uv, d.side, d.side_matched
        np.savez_compressed(cache_path, **cache)
    s = cfg.stereo
    calibration = StereoCalibration.load(cfg.resolve(s.extrinsics))
    intr0, intr1 = [Intrinsics.load(cfg.resolve(s[k])) for k in ('intrinsics_cam0', 'intrinsics_cam1')]
    packets, detections = [], {}
    for i, c in enumerate(z['camera']):
        p = CameraFrame(str(z['serial'][i]), z['image'][i], int(z['frame'][i]),
                        float(z['sensor_ms'][i]), str(z['domain'][i]), int(z['host_ns'][i]),
                        json.loads(str(z['metadata'][i])))
        packets.append((int(c), p))
        uv = cache['uv'][i]
        detections[c, p.frame_number] = (HandDetection(uv, np.full((21, 3), np.nan),
             np.full((21, 3), np.nan), str(cache['side'][i]), 1., bool(cache['matched'][i]))
             if cache['side'][i] else None)
    duration = (z['host_ns'][-1] - z['host_ns'][0]) / 1e9
    report = {'experiment': experiment, 'duration_sec': duration,
              'raw_frames': [int(np.sum(z['camera'] == c)) for c in (0, 1)],
              'queue_drops': z['queue_drops'].tolist(),
              'metadata_first_frames': [json.loads(str(z['metadata'][np.where(z['camera'] == c)[0][0]])) for c in (0, 1)],
              'domains': sorted(set(z['domain'].tolist())),
              'detection_ms': distribution(cache['detect_ms']),
              'latency_semantics': 'offline replay buffer delay lower bound; live E2E unmeasured; exposure time unknown',
              'modes': {}}
    report['detection_found_frames'] = [int(np.sum((z['camera'] == c) & (cache['side'] != '')))
                                         for c in (0, 1)]
    report['sensor_minus_host_clock'] = []
    for c in (0, 1):
        ix = np.where(z['camera'] == c)[0]
        host = z['host_ns'][ix].astype(float) / 1e9
        sensor = z['sensor_ms'][ix] / 1000
        # Epochs differ; subtract first sample rather than interpret the intercept.
        residual = ((sensor - sensor[0]) - (host - host[0])) * 1000
        report['sensor_minus_host_clock'].append({
            'relative_residual_ms': distribution(residual),
            'apparent_slope_ppm': float(np.polyfit(host - host[0], residual / 1000, 1)[0] * 1e6),
            'note': 'delivery jitter plus clock rate; short recording cannot establish drift'})
    arrays = {}
    for mode in ('nearest', 'lookahead', 'interpolate'):
        sync = FrameSynchronizer(mode)
        pairs = []
        for c, p in packets:
            pairs.extend(sync.push(c, p))
        pairs.extend(sync.drain(int(z['host_ns'][-1]) + 60_000_000))
        gaps, waits, valid, joints, positions, output_times, flags = [], [], [], [], [], [], []
        interp_success = 0
        compute_ms = []
        for pair in pairs:
            start = time.perf_counter_ns()
            a, l, r = pair.reference, pair.left, pair.right
            d0, dl, dr = [detections.get(key) for key in
                          ((0, a.frame_number), (1, l.frame_number), (1, r.frame_number))]
            d1, reason = (interpolate_detection(pair, (d0, dl, dr))
                          if mode == 'interpolate' else (dl, ''))
            interp_success += int(mode == 'interpolate' and d1 is not None)
            tri = pose = None
            if (d0 is not None and d1 is not None and d0.side == d1.side and
                    (mode == 'interpolate' or pair.gap_ms <= s.max_pair_gap_ms)):
                tri = triangulate_landmarks(d0.uv, d1.uv, intr0, intr1, calibration,
                                            s.max_reproj_px, s.min_ray_angle_deg)
                pose = stereo_hand_pose(tri, d0.side)
            compute_ms.append((time.perf_counter_ns() - start) / 1e6)
            gaps.append(pair.gap_ms)
            waits.append((pair.released_ns - a.host_received_ns) / 1e6)
            valid.append(pose is not None and pose.valid)
            joints.append(0 if tri is None else int(tri.valid.sum()))
            positions.append(np.full(3, np.nan) if pose is None else pose.palm_position_cam0)
            output_times.append(a.host_received_ns)
            flags.append(reason)
        positions = np.asarray(positions).reshape(-1, 3)
        valid = np.asarray(valid, bool)
        consecutive = valid[1:] & valid[:-1]
        steps = np.linalg.norm(np.diff(positions, axis=0)[consecutive], axis=1)
        jitter = (np.linalg.norm(positions[valid] - np.median(positions[valid], axis=0), axis=1)
                  if valid.any() and experiment == 'static' else [])
        report['modes'][mode] = {
            'paired_fps': len(pairs) / duration, 'valid_pose_fps_over_recorded_time': int(valid.sum()) / duration,
            'gap_ms': distribution(gaps), 'buffer_delay_ms_lower_bound': distribution(waits),
            'gap_semantics': ('maximum raw endpoint distance from reference' if mode == 'interpolate'
                              else 'raw selected pair timestamp difference'),
            'geometry_processing_ms': distribution(compute_ms),
            'selected_unique_frames': [len(u) for u in sync.used],
            'unselected_raw_frames': [report['raw_frames'][c] - len(sync.used[c]) for c in (0, 1)],
            'cam0_expired': sync.expired, 'cam1_endpoint_reuses': sync.reuses,
            'interpolation_success_ratio_over_cam0': (interp_success / report['raw_frames'][0]
                                                       if mode == 'interpolate' else None),
            'valid_frame_ratio_over_cam0': int(valid.sum()) / report['raw_frames'][0],
            'valid_joint_ratio_over_cam0': sum(joints) / (21 * report['raw_frames'][0]),
            'static_jitter_m': distribution(jitter), 'consecutive_displacement_m': distribution(steps),
            'interpretation': 'offline availability/continuity; not live FPS or dynamic accuracy'}
        arrays.update({mode + '_palm': positions, mode + '_valid': valid,
                       mode + '_reference_host_ns': output_times, mode + '_reason': flags})
    if roi0 and roi1:
        signals, times = [], []
        for c, roi in enumerate((roi0, roi1)):
            x, y, w, h = roi
            ix = np.where(z['camera'] == c)[0]
            signals.append([float(np.mean(z['image'][i, y:y+h, x:x+w])) for i in ix])
            times.append(z['host_ns'][ix] / 1e9)
        report['led'] = led_diagnostic(times[0], signals[0], times[1], signals[1])
        report['led']['clock_basis'] = 'host receive event onset; includes transport latency'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(out.with_suffix('.npz'), **arrays)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='results/stereo_sync.json')
    p.add_argument('--capture-seconds', type=float, default=0.)
    p.add_argument('--experiment', choices=['static', 'dynamic'], default='dynamic')
    p.add_argument('--led-roi0', nargs=4, type=int)
    p.add_argument('--led-roi1', nargs=4, type=int)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    path = cfg.resolve(a.data)
    if a.capture_seconds:
        if path.exists():
            raise SystemExit('refusing to overwrite raw recording')
        capture(cfg, path, a.capture_seconds)
    replay(cfg, path, cfg.resolve(a.out), a.experiment, a.led_roi0, a.led_roi1)


if __name__ == '__main__':
    main()
