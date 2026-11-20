"""Shared offline mono-vs-stereo evaluation without camera or UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Intrinsics
from .geometry import apply_T, inv_T
from .landmarks import INDEX_TIP, PALM, THUMB_TIP
from .landmarks import BONES
from .lifting import HandModel, PalmLifter
from .stereo import (StereoCalibration, interpolate_landmarks, stereo_hand_pose,
                     triangulate_landmarks)
from .tracker import HandDetection
from .stereo_sync import interpolate_detection
from types import SimpleNamespace


@dataclass
class MethodTrajectory:
    points_cam0: np.ndarray
    palm_position_cam0: np.ndarray
    palm_quaternion_xyzw: np.ndarray
    pinch_distance_m: np.ndarray
    valid: np.ndarray
    reproj_px: np.ndarray
    valid_joints: np.ndarray


def detection_from_cache(cache: dict, camera: int, index: int) -> HandDetection | None:
    if not bool(cache[f"found{camera}"][index]):
        return None
    return HandDetection(cache[f"uv{camera}"][index], cache[f"uvz{camera}"][index],
                         cache[f"world{camera}"][index], str(cache[f"side{camera}"][index]),
                         float(cache[f"score{camera}"][index]),
                         bool(cache[f"side_matched{camera}"][index]))


def _empty_trajectory(n: int) -> MethodTrajectory:
    return MethodTrajectory(np.full((n, 21, 3), np.nan), np.full((n, 3), np.nan),
                            np.full((n, 4), np.nan),
                            np.full(n, np.nan), np.zeros(n, bool),
                            np.full(n, np.nan), np.zeros(n, int))


def evaluate_mono(cache: dict, camera: int, times: np.ndarray, intr: Intrinsics,
                  model: HandModel, calibration: StereoCalibration,
                  max_reproj_px: float, lifter_options: dict) -> MethodTrajectory:
    """Run the existing PnP hand model and express output in cam0."""
    out = _empty_trajectory(len(times))
    lifter = PalmLifter(model, intr, **lifter_options)
    T_cam0_cam = np.eye(4) if camera == 0 else inv_T(calibration.T_cam1_cam0)
    for i, t in enumerate(times):
        det = detection_from_cache(cache, camera, i)
        if det is None:
            continue
        lift = lifter.lift(det, float(t))
        if lift is None:
            continue
        points = apply_T(T_cam0_cam, lift.kp3d_cam)
        out.points_cam0[i] = points
        out.palm_position_cam0[i] = apply_T(T_cam0_cam, lift.T_cam_hand[:3, 3][None])[0]
        out.palm_quaternion_xyzw[i] = Rotation.from_matrix(
            T_cam0_cam[:3, :3] @ lift.T_cam_hand[:3, :3]).as_quat()
        out.pinch_distance_m[i] = np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP])
        out.reproj_px[i] = lift.reproj_px
        out.valid[i] = lift.reproj_px <= max_reproj_px
        out.valid_joints[i] = 21 if out.valid[i] else 0
    return out


def evaluate_stereo(cache: dict, times0: np.ndarray, times1: np.ndarray,
                    intr0: Intrinsics, intr1: Intrinsics,
                    calibration: StereoCalibration, *, interpolated: bool,
                    max_interp_gap_sec: float, max_reproj_px: float,
                    min_ray_angle_deg: float, bone_rel_tolerance: float) -> MethodTrajectory:
    out = _empty_trajectory(len(times0))
    uv1_series = np.asarray(cache["uv1"], float).copy()
    uv1_series[~np.asarray(cache["found1"], bool)] = np.nan
    for i, t0 in enumerate(times0):
        if not bool(cache["found0"][i]):
            continue
        uv1 = None
        if interpolated:
            right = int(np.searchsorted(times1, t0))
            left = right - 1
            if right < len(times1) and times1[right] == t0:
                left = right
            if left < 0 or right >= len(times1):
                continue
            span = times1[right] - times1[left]
            if span < 0 or span > max_interp_gap_sec:
                continue
            pair = SimpleNamespace(alpha=(t0 - times1[left]) / span if span else 0.)
            aligned, _ = interpolate_detection(pair, (
                detection_from_cache(cache, 0, i), detection_from_cache(cache, 1, left),
                detection_from_cache(cache, 1, right)))
            if aligned is not None:
                uv1 = aligned.uv
        elif bool(cache["found1"][i]):
            uv1 = uv1_series[i]
        if uv1 is None:
            continue
        tri = triangulate_landmarks(
            cache["uv0"][i], uv1, intr0, intr1, calibration,
            max_reproj_px=max_reproj_px, min_ray_angle_deg=min_ray_angle_deg,
            bone_rel_tolerance=bone_rel_tolerance)
        out.points_cam0[i] = tri.points_cam0
        out.valid_joints[i] = int(tri.valid.sum())
        if tri.valid.any():
            out.reproj_px[i] = float(np.median(tri.reproj_px[tri.valid]))
        side = str(cache["side0"][i]) or "right"
        pose = stereo_hand_pose(tri, side)
        out.valid[i] = pose.valid
        out.palm_position_cam0[i] = pose.palm_position_cam0
        out.palm_quaternion_xyzw[i] = pose.palm_quaternion_xyzw
        out.pinch_distance_m[i] = pose.pinch_distance_m
    return out


def summarize_methods(methods: dict[str, MethodTrajectory], experiments,
                      gt_position_cam0: np.ndarray | None = None,
                      gt_quaternion_xyzw: np.ndarray | None = None,
                      gt_points_cam0: np.ndarray | None = None,
                      gt_pinch_distance_m: np.ndarray | None = None) -> dict:
    """Aggregate validity and agreement; absolute error exists only with external GT."""
    experiments = np.asarray(experiments)
    result = {"accuracy_note": (
        "absolute errors use supplied independent GT" if gt_position_cam0 is not None else
        "no independent 3D GT supplied; pairwise values are agreement, not accuracy")}
    for experiment in sorted(set(experiments.tolist())):
        mask = experiments == experiment
        block = {"n_frames": int(mask.sum()), "methods": {}}
        common = mask.copy()
        for method in methods.values():
            common &= method.valid
        block["all_methods_common_valid_frames"] = int(common.sum())
        for name, method in methods.items():
            valid = mask & method.valid
            values = {
                "valid_frames": int(valid.sum()),
                "valid_ratio": float(valid.sum() / max(mask.sum(), 1)),
                "median_reprojection_px": (float(np.nanmedian(method.reproj_px[valid]))
                                             if valid.any() else None),
                "median_valid_joints": (float(np.median(method.valid_joints[mask]))
                                         if mask.any() else None),
            }
            if valid.any():
                lengths = np.stack([
                    np.linalg.norm(method.points_cam0[:, a] - method.points_cam0[:, b], axis=1)
                    for a, b in BONES], axis=1)
                values["bone_length_std_median_m"] = float(
                    np.nanmedian(np.nanstd(np.where(valid[:, None], lengths, np.nan), axis=0)))
                values["pinch_distance_std_m"] = float(np.nanstd(method.pinch_distance_m[valid]))
                if experiment == "static":
                    pos = method.palm_position_cam0[valid]
                    jitter = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
                    values["static_pose_jitter_median_m"] = float(np.median(jitter))
                    values["static_pose_jitter_p95_m"] = float(np.percentile(jitter, 95))
            if gt_position_cam0 is not None:
                gt_ok = np.isfinite(gt_position_cam0).all(1)
                use = valid & gt_ok
                err = np.linalg.norm(method.palm_position_cam0[use] - gt_position_cam0[use], axis=1)
                values["palm_position_error_median_m"] = float(np.median(err)) if len(err) else None
                values["palm_position_error_p95_m"] = float(np.percentile(err, 95)) if len(err) else None
            if gt_quaternion_xyzw is not None:
                gt_ok = np.isfinite(gt_quaternion_xyzw).all(1)
                use = valid & gt_ok
                angle = []
                for q, qgt in zip(method.palm_quaternion_xyzw[use], gt_quaternion_xyzw[use]):
                    angle.append(np.degrees((Rotation.from_quat(qgt).inv() *
                                             Rotation.from_quat(q)).magnitude()))
                values["palm_orientation_error_median_deg"] = (
                    float(np.median(angle)) if angle else None)
            if gt_points_cam0 is not None:
                gt_ok = np.isfinite(gt_points_cam0).all((1, 2))
                use = valid & gt_ok
                joint = np.linalg.norm(method.points_cam0[use] - gt_points_cam0[use], axis=2)
                values["joint_error_median_m"] = float(np.median(joint)) if joint.size else None
            if gt_pinch_distance_m is not None:
                use = valid & np.isfinite(gt_pinch_distance_m) & np.isfinite(method.pinch_distance_m)
                pinch_error = np.abs(method.pinch_distance_m[use] - gt_pinch_distance_m[use])
                values["pinch_error_median_m"] = (
                    float(np.median(pinch_error)) if len(pinch_error) else None)
            block["methods"][name] = values
        if "B2_nearest" in methods:
            ref = methods["B2_nearest"]
            for name in ("B1_A", "B1_B"):
                use = mask & methods[name].valid & ref.valid
                err = np.linalg.norm(methods[name].palm_position_cam0[use] -
                                     ref.palm_position_cam0[use], axis=1)
                block["methods"][name]["agreement_to_B2_nearest_median_m"] = (
                    float(np.median(err)) if len(err) else None)
                joint = np.linalg.norm(methods[name].points_cam0[use] -
                                       ref.points_cam0[use], axis=2)
                block["methods"][name]["joint_disagreement_to_B2_median_m"] = (
                    float(np.nanmedian(joint)) if joint.size else None)
                pinch = np.abs(methods[name].pinch_distance_m[use] -
                               ref.pinch_distance_m[use])
                block["methods"][name]["pinch_disagreement_to_B2_median_m"] = (
                    float(np.nanmedian(pinch)) if len(pinch) and np.isfinite(pinch).any() else None)
                angles = []
                for q, qr in zip(methods[name].palm_quaternion_xyzw[use],
                                 ref.palm_quaternion_xyzw[use]):
                    if np.isfinite(q).all() and np.isfinite(qr).all():
                        angles.append(np.degrees((Rotation.from_quat(qr).inv() *
                                                 Rotation.from_quat(q)).magnitude()))
                block["methods"][name]["orientation_disagreement_to_B2_median_deg"] = (
                    float(np.median(angles)) if angles else None)
        result[experiment] = block
    return result
