"""Probe two serial-bound RealSense RGB cameras and export factory intrinsics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..core.camera import intrinsics_from_realsense


def _device_info(device, rs, field):
    return device.get_info(field) if device.supports(field) else None


def _color_profiles(device, rs):
    profiles = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video = profile.as_video_stream_profile()
            profiles.add((video.width(), video.height(), profile.fps(), str(profile.format())))
    return sorted(profiles)


def enumerate_devices(rs):
    """Enumerate devices without hiding per-device access failures."""
    rows = []
    try:
        device_list = rs.context().query_devices()
        reported_count = len(device_list)
        devices = list(device_list)
    except RuntimeError as exc:
        return [{"sdk_reported_device_count": locals().get("reported_count"),
                 "enumeration_error": str(exc), "error": str(exc)}]
    for index, device in enumerate(devices):
        try:
            rows.append({
                "name": _device_info(device, rs, rs.camera_info.name),
                "serial": _device_info(device, rs, rs.camera_info.serial_number),
                "firmware": _device_info(device, rs, rs.camera_info.firmware_version),
                "usb_type": _device_info(device, rs, rs.camera_info.usb_type_descriptor),
                "color_profiles": _color_profiles(device, rs),
            })
        except RuntimeError as exc:
            rows.append({"device_index_for_diagnostics_only": index, "error": str(exc)})
    return rows


def choose_common_profile(devices, preferred=(1280, 720, 30)):
    common = None
    for device in devices:
        modes = {(w, h, fps) for w, h, fps, _ in device["color_profiles"]}
        common = modes if common is None else common & modes
    if not common:
        raise RuntimeError("the selected devices have no common RGB resolution/FPS")
    if preferred in common:
        return preferred
    return max(common, key=lambda mode: (mode[2] == 30, mode[0] * mode[1], mode[2]))


def summarize_samples(samples):
    if not samples:
        return {"n_frames": 0}
    host = np.asarray([s["host_received_ns"] for s in samples], dtype=np.int64)
    numbers = np.asarray([s["frame_number"] for s in samples], dtype=np.int64)
    intervals_ms = np.diff(host) / 1e6
    sensor_intervals_ms = np.diff([s["sensor_timestamp_ms"] for s in samples])
    elapsed = (host[-1] - host[0]) / 1e9 if len(host) > 1 else 0.0
    return {
        "n_frames": len(samples),
        "measured_fps": float((len(samples) - 1) / elapsed) if elapsed > 0 else None,
        "host_interval_median_ms": float(np.median(intervals_ms)) if len(intervals_ms) else None,
        "host_interval_p95_ms": float(np.percentile(intervals_ms, 95)) if len(intervals_ms) else None,
        "host_interval_jitter_std_ms": float(np.std(intervals_ms)) if len(intervals_ms) else None,
        "sensor_interval_median_ms": (float(np.median(sensor_intervals_ms))
                                      if len(sensor_intervals_ms) else None),
        "sensor_interval_jitter_std_ms": (float(np.std(sensor_intervals_ms))
                                          if len(sensor_intervals_ms) else None),
        "frame_number_drops": int(np.maximum(np.diff(numbers) - 1, 0).sum()),
        "timestamp_domains": sorted({s["timestamp_domain"] for s in samples}),
        "first_frame": samples[0], "last_frame": samples[-1],
    }


def stream_devices(rs, devices, profile, duration, out_dir, capture_count=0, capture_every=15):
    width, height, fps = profile
    pipelines, samples, active, captures = [], {}, {}, {}
    try:
        for device in devices:
            serial = device["serial"]
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline_profile = pipeline.start(config)
            video = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = intrinsics_from_realsense(video.get_intrinsics(), serial=serial, fps=fps)
            path = out_dir / f"intrinsics_color_{serial}_{width}x{height}_{fps}.json"
            intr.save(path)
            active[serial] = {"intrinsics_file": str(path), "K": intr.K.tolist(),
                              "dist": intr.dist.tolist(), **intr.meta}
            pipelines.append((serial, pipeline))
            samples[serial] = []
            captures[serial] = []
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            for serial, pipeline in pipelines:
                frame = pipeline.wait_for_frames(2000).get_color_frame()
                samples[serial].append({
                    "frame_number": int(frame.get_frame_number()),
                    "sensor_timestamp_ms": float(frame.get_timestamp()),
                    "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                    "host_received_ns": time.monotonic_ns(),
                })
                if (len(captures[serial]) < capture_count
                        and (len(samples[serial]) - 1) % max(1, capture_every) == 0):
                    import cv2
                    capture_dir = out_dir / "captures" / serial
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    path = capture_dir / f"color_{frame.get_frame_number():06d}.png"
                    if not cv2.imwrite(str(path), np.asanyarray(frame.get_data())):
                        raise RuntimeError(f"failed to save capture: {path}")
                    captures[serial].append(str(path))
    finally:
        for _, pipeline in pipelines:
            pipeline.stop()
    return {serial: {"active": active[serial], "statistics": summarize_samples(rows),
                     "captures": captures[serial]}
            for serial, rows in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="append", help="serial to use (repeat twice)")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", default="calib/realsense_probe")
    parser.add_argument("--capture-count", type=int, default=0,
                        help="save this many BGR PNG frames per camera")
    parser.add_argument("--capture-every", type=int, default=15,
                        help="save every Nth received frame")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    import pyrealsense2 as rs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = enumerate_devices(rs)
    report = {"host_clock": "time.monotonic_ns", "devices": devices}
    report_path = out_dir / "probe_report.json"
    if args.list_only or any("error" in device for device in devices):
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if any("error" in device for device in devices):
            raise SystemExit(f"device access failed; report saved to {report_path}")
        return
    selected = [device for device in devices
                if not args.serial or device["serial"] in args.serial]
    if len(selected) != 2:
        raise SystemExit(f"expected exactly two selected RealSense devices, got {len(selected)}")
    if args.serial and len(set(args.serial)) != 2:
        raise SystemExit("repeat --serial with two distinct serial numbers")
    profile = choose_common_profile(selected, (args.width, args.height, args.fps))
    report["selected_serials"] = [device["serial"] for device in selected]
    report["active_profile"] = {"width": profile[0], "height": profile[1], "fps": profile[2]}
    try:
        report["stream_results"] = stream_devices(
            rs, selected, profile, args.duration, out_dir,
            capture_count=max(0, args.capture_count), capture_every=max(1, args.capture_every))
    except (RuntimeError, ValueError) as exc:
        report["stream_error"] = str(exc)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(f"streaming failed; report saved to {report_path}: {exc}") from exc
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
