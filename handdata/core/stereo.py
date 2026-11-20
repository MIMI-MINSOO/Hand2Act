"""Stereo calibration and metric hand geometry in the cam0 coordinate frame."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Intrinsics
from .geometry import inv_T, make_T, palm_frame, rot_angle
from .landmarks import CONNECTIONS, INDEX_TIP, PALM, THUMB_TIP


@dataclass
class StereoCalibration:
    serial_cam0: str
    serial_cam1: str
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    intrinsics_cam0: str = ""
    intrinsics_cam1: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def T_cam1_cam0(self) -> np.ndarray:
        return make_T(self.R, self.t)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t))

    def save(self, path) -> None:
        data = {
            "serial_cam0": self.serial_cam0, "serial_cam1": self.serial_cam1,
            "width": self.width, "height": self.height,
            "intrinsics_cam0": self.intrinsics_cam0,
            "intrinsics_cam1": self.intrinsics_cam1,
            "R": np.asarray(self.R).tolist(), "t": np.asarray(self.t).reshape(3).tolist(),
            "T_cam1_cam0": self.T_cam1_cam0.tolist(), "baseline_m": self.baseline_m,
            "T_cam0_cam1": inv_T(self.T_cam1_cam0).tolist(),
            "metadata": self.metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "StereoCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["serial_cam0"], data["serial_cam1"],
                   np.asarray(data["R"], float), np.asarray(data["t"], float),
                   int(data["width"]), int(data["height"]),
                   data.get("intrinsics_cam0", ""), data.get("intrinsics_cam1", ""),
                   data.get("metadata", {}))


def match_common_charuco(found0, found1, min_common: int = 6):
    """Match two ``detect_with_ids`` results by sorted ChArUco corner ID."""
    if found0 is None or found1 is None:
        return None
    obj0, img0, ids0 = found0
    obj1, img1, ids1 = found1
    map0 = {int(i): k for k, i in enumerate(np.asarray(ids0).reshape(-1))}
    map1 = {int(i): k for k, i in enumerate(np.asarray(ids1).reshape(-1))}
    common = np.array(sorted(set(map0) & set(map1)), dtype=np.int32)
    if len(common) < min_common:
        return None
    ix0 = [map0[int(i)] for i in common]
    ix1 = [map1[int(i)] for i in common]
    obj = np.asarray(obj0, float)[ix0]
    if not np.allclose(obj, np.asarray(obj1, float)[ix1], atol=1e-9):
        raise ValueError("same ChArUco IDs map to different object points")
    return obj, np.asarray(img0, float)[ix0], np.asarray(img1, float)[ix1], common


def _solve_board_pose(obj, img, intr: Intrinsics) -> np.ndarray:
    n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        np.asarray(obj, np.float64), np.asarray(img, np.float64), intr.K, intr.dist,
        flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        raise ValueError("board pose solve failed")
    best = int(np.argmin(np.asarray(errors).reshape(-1)))
    rvec, tvec = cv2.solvePnPRefineLM(
        np.asarray(obj, np.float64), np.asarray(img, np.float64), intr.K, intr.dist,
        rvecs[best].copy(), tvecs[best].copy())
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, np.asarray(tvec).reshape(3))


def _fit_stereo(observations, indices, intr0: Intrinsics, intr1: Intrinsics):
    objs = [np.asarray(observations[i][0], np.float32).reshape(-1, 1, 3) for i in indices]
    imgs0 = [np.asarray(observations[i][1], np.float32).reshape(-1, 1, 2) for i in indices]
    imgs1 = [np.asarray(observations[i][2], np.float32).reshape(-1, 1, 2) for i in indices]
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
        objs, imgs0, imgs1, intr0.K.copy(), intr0.dist.copy(),
        intr1.K.copy(), intr1.dist.copy(), (intr0.width, intr0.height),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-8))
    return float(rms), np.asarray(R, float), np.asarray(t, float).reshape(3)


def _pair_relative_pose(observation, intr0, intr1):
    obj, img0, img1, *_ = observation
    T0b = _solve_board_pose(obj, img0, intr0)
    T1b = _solve_board_pose(obj, img1, intr1)
    return T1b @ inv_T(T0b)


def _evaluate_pairs(observations, indices, intr0, intr1, R, t):
    per_pair, all_errors, all_metric_errors = [], [], []
    T10 = make_T(R, t)
    for i in indices:
        obj, img0, img1, *_ = observations[i]
        T0b = _solve_board_pose(obj, img0, intr0)
        P0 = np.asarray(obj) @ T0b[:3, :3].T + T0b[:3, 3]
        P1 = P0 @ T10[:3, :3].T + T10[:3, 3]
        e0 = np.linalg.norm(intr0.project(P0) - img0, axis=1)
        e1 = np.linalg.norm(intr1.project(P1) - img1, axis=1)
        errors = np.concatenate([e0, e1])
        all_errors.extend(errors.tolist())
        xy0, xy1 = intr0.normalized(img0), intr1.normalized(img1)
        Xh = cv2.triangulatePoints(
            np.c_[np.eye(3), np.zeros(3)], np.c_[R, np.asarray(t).reshape(3, 1)],
            xy0.T, xy1.T).T
        X = Xh[:, :3] / Xh[:, 3, None]
        # The shortest non-zero board-point distance is one ChArUco square.
        # Comparing only those neighbouring pairs avoids an O(N^2) statistic
        # being dominated by long diagonals while remaining board-layout agnostic.
        d_obj = np.linalg.norm(obj[:, None] - obj[None, :], axis=2)
        positive = d_obj[d_obj > 1e-9]
        metric_errors = []
        if len(positive):
            edge = float(np.min(positive))
            a, b = np.where(np.triu(np.isclose(d_obj, edge, rtol=1e-4, atol=1e-9), 1))
            metric_errors = (np.linalg.norm(X[a] - X[b], axis=1) - d_obj[a, b]).tolist()
            all_metric_errors.extend(metric_errors)
        per_pair.append({"index": int(i), "n_common": int(len(obj)),
                         "median_px": float(np.median(errors)),
                         "p95_px": float(np.percentile(errors, 95)),
                         "corner_distance_median_abs_mm": (
                             float(np.median(np.abs(metric_errors)) * 1000)
                             if metric_errors else None),
                         "corner_distance_p95_abs_mm": (
                             float(np.percentile(np.abs(metric_errors), 95) * 1000)
                             if metric_errors else None)})
    return {"n_pairs": len(per_pair), "n_observations": len(all_errors),
            "median_px": float(np.median(all_errors)) if all_errors else None,
            "p95_px": float(np.percentile(all_errors, 95)) if all_errors else None,
            "corner_distance_n": len(all_metric_errors),
            "corner_distance_median_abs_mm": (
                float(np.median(np.abs(all_metric_errors)) * 1000)
                if all_metric_errors else None),
            "corner_distance_p95_abs_mm": (
                float(np.percentile(np.abs(all_metric_errors), 95) * 1000)
                if all_metric_errors else None),
            "per_pair": per_pair}


def calibrate_stereo(observations, intr0: Intrinsics, intr1: Intrinsics,
                     holdout_every: int = 5, min_pairs: int = 8,
                     holdout_indices=None) -> dict:
    """Fit fixed-intrinsic ``T_cam1_cam0`` with pose-consistency rejection."""
    if intr0.width != intr1.width or intr0.height != intr1.height:
        raise ValueError("stereo intrinsics must use the same active image size")
    if len(observations) < min_pairs:
        raise ValueError(f"need at least {min_pairs} usable stereo pairs")
    explicit_holdout = holdout_indices is not None
    if explicit_holdout:
        holdout = sorted(set(int(i) for i in holdout_indices))
        if any(i < 0 or i >= len(observations) for i in holdout):
            raise ValueError("holdout index is outside observation range")
    else:
        holdout = [i for i in range(len(observations))
                   if holdout_every > 0 and i % holdout_every == 0]
    train = [i for i in range(len(observations)) if i not in holdout]
    if len(train) < min_pairs:
        if explicit_holdout:
            raise ValueError(f"need at least {min_pairs} calibration pairs excluding holdout")
        train, holdout = list(range(len(observations))), []
    initial_rms, R, t = _fit_stereo(observations, train, intr0, intr1)
    Tfit = make_T(R, t)
    deviations = []
    for i in train:
        Ti = _pair_relative_pose(observations[i], intr0, intr1)
        deviations.append((i, rot_angle(Ti[:3, :3], R),
                           float(np.linalg.norm(Ti[:3, 3] - t))))
    rot = np.array([d[1] for d in deviations])
    trans = np.array([d[2] for d in deviations])
    robust_limit = lambda x, floor: max(float(np.median(x) + 3 * 1.4826 * np.median(
        np.abs(x - np.median(x)))), floor)
    rot_limit = robust_limit(rot, np.deg2rad(0.5))
    trans_limit = robust_limit(trans, 0.005)
    inliers = [i for i, r, d in deviations if r <= rot_limit and d <= trans_limit]
    rejected = [i for i in train if i not in inliers]
    if len(inliers) >= min_pairs and rejected:
        rms, R, t = _fit_stereo(observations, inliers, intr0, intr1)
    else:
        rms, inliers = initial_rms, train
    return {
        "R": R, "t": t, "baseline_m": float(np.linalg.norm(t)),
        "stereo_rms_px": rms, "initial_rms_px": initial_rms,
        "train_indices": train, "inlier_indices": inliers, "rejected_indices": rejected,
        "holdout_indices": holdout,
        "pose_deviations": [{"index": int(i), "rotation_deg": float(np.rad2deg(r)),
                             "translation_m": float(d)} for i, r, d in deviations],
        "rotation_outlier_limit_deg": float(np.rad2deg(rot_limit)),
        "translation_outlier_limit_m": trans_limit,
        "fit_evaluation": _evaluate_pairs(observations, inliers, intr0, intr1, R, t),
        "holdout_evaluation": _evaluate_pairs(observations, holdout, intr0, intr1, R, t),
    }


def interpolate_landmarks(times, landmarks, target_time: float, max_gap_sec: float = 0.05):
    """Linearly interpolate offline 2D landmarks without coupling to capture."""
    times = np.asarray(times, float)
    landmarks = np.asarray(landmarks, float)
    if times.ndim != 1 or landmarks.shape[0] != len(times):
        raise ValueError("times and landmarks length mismatch")
    right = int(np.searchsorted(times, target_time, side="left"))
    if right < len(times) and times[right] == target_time:
        return landmarks[right].copy(), np.isfinite(landmarks[right]).all(axis=-1)
    if right == 0 or right == len(times):
        return None, None
    left = right - 1
    gap = times[right] - times[left]
    if gap <= 0 or gap > max_gap_sec:
        return None, None
    alpha = (target_time - times[left]) / gap
    valid = np.isfinite(landmarks[left]).all(axis=-1) & np.isfinite(landmarks[right]).all(axis=-1)
    out = (1 - alpha) * landmarks[left] + alpha * landmarks[right]
    out[~valid] = np.nan
    return out, valid


@dataclass
class TriangulationResult:
    points_cam0: np.ndarray
    valid: np.ndarray
    reasons: list[list[str]]
    reproj_px: np.ndarray
    ray_angle_deg: np.ndarray


def triangulate_landmarks(uv0, uv1, intr0: Intrinsics, intr1: Intrinsics,
                          calibration: StereoCalibration, max_reproj_px: float = 3.0,
                          min_ray_angle_deg: float = 1.0,
                          reference_bones: np.ndarray | None = None,
                          bone_rel_tolerance: float = 0.35) -> TriangulationResult:
    """Triangulate corresponding raw pixels using normalized undistorted rays."""
    uv0, uv1 = np.asarray(uv0, float), np.asarray(uv1, float)
    if uv0.shape != uv1.shape or uv0.ndim != 2 or uv0.shape[1] != 2:
        raise ValueError("uv0 and uv1 must have matching (N,2) shape")
    n = len(uv0)
    reasons = [[] for _ in range(n)]
    finite = np.isfinite(uv0).all(1) & np.isfinite(uv1).all(1)
    xy0 = np.full_like(uv0, np.nan)
    xy1 = np.full_like(uv1, np.nan)
    if finite.any():
        xy0[finite] = intr0.normalized(uv0[finite])
        xy1[finite] = intr1.normalized(uv1[finite])
    P0 = np.c_[np.eye(3), np.zeros(3)]
    P1 = np.c_[calibration.R, calibration.t.reshape(3, 1)]
    X = np.full((n, 3), np.nan)
    if finite.any():
        Xh = cv2.triangulatePoints(P0, P1, xy0[finite].T, xy1[finite].T).T
        good_w = np.abs(Xh[:, 3]) > 1e-12
        Xi = np.full((len(Xh), 3), np.nan)
        Xi[good_w] = Xh[good_w, :3] / Xh[good_w, 3, None]
        X[finite] = Xi
    X1 = X @ calibration.R.T + calibration.t
    reproj = np.full((n, 2), np.inf)
    computable = finite & np.isfinite(X).all(1)
    if computable.any():
        reproj[computable, 0] = np.linalg.norm(intr0.project(X[computable]) - uv0[computable], axis=1)
        reproj[computable, 1] = np.linalg.norm(intr1.project(X1[computable]) - uv1[computable], axis=1)
    C1 = -calibration.R.T @ calibration.t
    ray_angle = np.zeros(n)
    for i in range(n):
        if not finite[i]:
            reasons[i].append("missing_2d")
            continue
        if not np.isfinite(X[i]).all():
            reasons[i].append("homogeneous_failure")
            continue
        d0 = X[i] / np.linalg.norm(X[i])
        d1 = (X[i] - C1) / np.linalg.norm(X[i] - C1)
        ray_angle[i] = np.degrees(np.arccos(np.clip(d0 @ d1, -1.0, 1.0)))
        if X[i, 2] <= 0 or X1[i, 2] <= 0:
            reasons[i].append("nonpositive_depth")
        if np.max(reproj[i]) > max_reproj_px:
            reasons[i].append("high_reprojection")
        if ray_angle[i] < min_ray_angle_deg:
            reasons[i].append("small_ray_angle")
    valid = np.array([not r for r in reasons])
    for k, (a, b) in enumerate(CONNECTIONS):
        if a >= n or b >= n or not (valid[a] and valid[b]):
            continue
        length = float(np.linalg.norm(X[a] - X[b]))
        bad = not 0.003 <= length <= 0.15
        if reference_bones is not None and k < len(reference_bones):
            ref = float(reference_bones[k])
            bad = bad or abs(length - ref) > bone_rel_tolerance * max(ref, 1e-6)
        if bad:
            reasons[a].append("bone_length")
            reasons[b].append("bone_length")
            valid[a] = valid[b] = False
    return TriangulationResult(X, valid, reasons, np.max(reproj, axis=1), ray_angle)


@dataclass
class StereoHandPose:
    valid: bool
    palm_position_cam0: np.ndarray
    palm_quaternion_xyzw: np.ndarray
    ee_position_cam0: np.ndarray
    pinch_distance_m: float
    pinch_valid: bool
    gripper: float
    quality_flags: list[str]


def stereo_hand_pose(tri: TriangulationResult, side: str = "right",
                     ee_offset_palm=(0.0, 0.0, 0.0), gripper_range=None) -> StereoHandPose:
    flags = []
    palm_ok = bool(np.all(tri.valid[np.asarray(PALM)]))
    if palm_ok:
        position, R = palm_frame(tri.points_cam0[np.asarray(PALM)], side)
        quat = Rotation.from_matrix(R).as_quat()
        ee = position + R @ np.asarray(ee_offset_palm, float)
    else:
        flags.append("invalid_palm")
        position = quat = ee = np.full(3 if not palm_ok else 4, np.nan)
        quat = np.full(4, np.nan)
    pinch_ok = bool(tri.valid[THUMB_TIP] and tri.valid[INDEX_TIP])
    pinch = (float(np.linalg.norm(tri.points_cam0[THUMB_TIP] - tri.points_cam0[INDEX_TIP]))
             if pinch_ok else float("nan"))
    if not pinch_ok:
        flags.append("invalid_pinch")
    gripper = float("nan")
    if pinch_ok and gripper_range is not None:
        closed, opened = gripper_range
        gripper = float(np.clip((pinch - closed) / max(opened - closed, 1e-9), 0, 1))
    return StereoHandPose(palm_ok, position, quat, ee, pinch, pinch_ok, gripper, flags)
