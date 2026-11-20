"""Synthetic M2/M3 stereo geometry tests; no camera or MediaPipe model needed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handdata.core.geometry import make_T, palm_frame, rot_angle
from handdata.core.landmarks import CONNECTIONS, PALM
from handdata.core.realsense_stereo import CameraFrame, nearest_frame_pair
from handdata.core.stereo import (StereoCalibration, calibrate_stereo, interpolate_landmarks,
                                  match_common_charuco, stereo_hand_pose,
                                  triangulate_landmarks)
from handdata.core.stereo_evaluation import MethodTrajectory, summarize_methods

from .synthetic import _RIGHT_HAND, camera


def _stereo_setup():
    intr0 = camera(640, 480, 620.0)
    intr1 = camera(640, 480, 625.0)
    R = Rotation.from_euler("xyz", [0.5, -2.0, 0.3], degrees=True).as_matrix()
    t = np.array([-0.18, 0.002, 0.004])
    calib = StereoCalibration("cam0", "cam1", R, t, 640, 480)
    return intr0, intr1, calib


def _project_stereo(P0, intr0, intr1, calib):
    return intr0.project(P0), intr1.project(P0 @ calib.R.T + calib.t)


def _board_observations(n=18, noise=0.05):
    intr0, intr1, calib = _stereo_setup()
    obj = np.array([[x, y, 0.0] for y in np.arange(1, 7) * .04
                    for x in np.arange(1, 5) * .04])
    rng = np.random.default_rng(8)
    observations = []
    for i in range(n):
        Rb = Rotation.from_euler("xyz", rng.uniform([-20, -20, -15], [20, 20, 15]),
                                 degrees=True).as_matrix()
        tb = np.array([rng.uniform(-.12, .05), rng.uniform(-.12, .03), rng.uniform(.55, .9)])
        P0 = obj @ Rb.T + tb
        uv0, uv1 = _project_stereo(P0, intr0, intr1, calib)
        uv0 += rng.normal(0, noise, uv0.shape)
        uv1 += rng.normal(0, noise, uv1.shape)
        observations.append((obj.copy(), uv0, uv1, np.arange(len(obj)), f"pair_{i:04d}"))
    return intr0, intr1, calib, observations


def test_common_charuco_correspondence_sorted_by_same_ids():
    obj = np.c_[np.arange(5), np.zeros((5, 2))].astype(float)
    found0 = (obj[[3, 1, 4]], np.array([[30, 0], [10, 0], [40, 0.]]), np.array([3, 1, 4]))
    found1 = (obj[[4, 3, 0]], np.array([[41, 0], [31, 0], [1, 0.]]), np.array([4, 3, 0]))
    matched = match_common_charuco(found0, found1, min_common=2)
    np.testing.assert_array_equal(matched[3], [3, 4])
    np.testing.assert_allclose(matched[1], [[30, 0], [40, 0]])
    np.testing.assert_allclose(matched[2], [[31, 0], [41, 0]])
    bad = (found1[0].copy(), found1[1], found1[2])
    bad[0][1, 0] += 1
    with pytest.raises(ValueError, match="different object points"):
        match_common_charuco(found0, bad, min_common=2)


def test_stereo_calibration_recovers_relative_transform_and_rejects_outlier():
    intr0, intr1, truth, observations = _board_observations()
    observations[7] = (*observations[7][:2], observations[7][2] + np.array([80., -30.]),
                       *observations[7][3:])
    result = calibrate_stereo(observations, intr0, intr1, holdout_every=5, min_pairs=8)
    assert rot_angle(result["R"], truth.R) < np.deg2rad(.2)
    assert np.linalg.norm(result["t"] - truth.t) < .003
    assert 7 in result["rejected_indices"]
    assert result["holdout_evaluation"]["p95_px"] < .5
    assert result["holdout_evaluation"]["corner_distance_p95_abs_mm"] < 1.0


def test_stereo_calibration_respects_explicit_independent_holdout():
    intr0, intr1, _, observations = _board_observations(n=16)
    holdout = [12, 13, 14, 15]
    result = calibrate_stereo(observations, intr0, intr1, min_pairs=8,
                              holdout_indices=holdout)
    assert result["holdout_indices"] == holdout
    assert set(result["train_indices"]).isdisjoint(holdout)
    assert result["holdout_evaluation"]["n_pairs"] == 4
    with pytest.raises(ValueError, match="excluding holdout"):
        calibrate_stereo(observations[:10], intr0, intr1, min_pairs=8,
                         holdout_indices=[2, 3, 4])


def test_noiseless_triangulation_palm_and_pinch():
    intr0, intr1, calib = _stereo_setup()
    Rhand = Rotation.from_euler("xyz", [5, -8, 12], degrees=True).as_matrix()
    P0 = _RIGHT_HAND @ Rhand.T + np.array([-.04, -.02, .65])
    uv0, uv1 = _project_stereo(P0, intr0, intr1, calib)
    tri = triangulate_landmarks(uv0, uv1, intr0, intr1, calib,
                                max_reproj_px=.01, min_ray_angle_deg=.1)
    assert tri.valid.all()
    assert np.max(np.linalg.norm(tri.points_cam0 - P0, axis=1)) < 1e-8
    pose = stereo_hand_pose(tri, "right", gripper_range=(.02, .12))
    expected_o, expected_R = palm_frame(P0[np.asarray(PALM)], "right")
    assert pose.valid and pose.pinch_valid
    np.testing.assert_allclose(pose.palm_position_cam0, expected_o, atol=1e-8)
    assert rot_angle(Rotation.from_quat(pose.palm_quaternion_xyzw).as_matrix(), expected_R) < 1e-8
    assert abs(pose.pinch_distance_m - np.linalg.norm(P0[4] - P0[8])) < 1e-8


def test_triangulation_marks_occlusion_depth_and_small_angle_invalid():
    intr0, intr1, calib = _stereo_setup()
    P0 = _RIGHT_HAND + np.array([0, 0, .7])
    uv0, uv1 = _project_stereo(P0, intr0, intr1, calib)
    uv1[8] = np.nan
    tri = triangulate_landmarks(uv0, uv1, intr0, intr1, calib)
    assert not tri.valid[8] and "missing_2d" in tri.reasons[8]

    behind = np.array([[.02, .01, -.6]])
    b0, b1 = _project_stereo(behind, intr0, intr1, calib)
    bad = triangulate_landmarks(b0, b1, intr0, intr1, calib, min_ray_angle_deg=0)
    assert not bad.valid[0] and "nonpositive_depth" in bad.reasons[0]

    tiny = StereoCalibration("a", "b", np.eye(3), np.array([-.0001, 0, 0]), 640, 480)
    far = np.array([[0., 0., 10.]])
    f0, f1 = _project_stereo(far, intr0, intr1, tiny)
    angle = triangulate_landmarks(f0, f1, intr0, intr1, tiny, min_ray_angle_deg=1)
    assert not angle.valid[0] and "small_ray_angle" in angle.reasons[0]


def test_nearest_frame_pair_uses_sensor_time_and_reports_gap():
    image = np.zeros((1, 1, 3), np.uint8)
    make = lambda serial, ts, host: CameraFrame(serial, image, 1, ts, "system_time", host)
    frames0 = [make("0", 100.0, 10), make("0", 133.0, 20)]
    frames1 = [make("1", 105.0, 100), make("1", 132.0, 200)]
    i, j, gap, clock = nearest_frame_pair(frames0, frames1)
    assert (i, j) == (1, 1) and gap == 1.0 and clock == "sensor"


def test_stereo_calibration_roundtrip_includes_inverse(tmp_path):
    _, _, calibration = _stereo_setup()
    path = tmp_path / "stereo.json"
    calibration.save(path)
    loaded = StereoCalibration.load(path)
    np.testing.assert_allclose(loaded.T_cam1_cam0, calibration.T_cam1_cam0)
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    T01 = np.asarray(raw["T_cam0_cam1"])
    np.testing.assert_allclose(T01 @ calibration.T_cam1_cam0, np.eye(4), atol=1e-12)


def test_bone_consistency_and_offline_landmark_interpolation():
    intr0, intr1, calib = _stereo_setup()
    P0 = _RIGHT_HAND + np.array([0, 0, .7])
    uv0, uv1 = _project_stereo(P0, intr0, intr1, calib)
    reference = np.array([np.linalg.norm(P0[a] - P0[b]) for a, b in CONNECTIONS])
    reference[0] *= 3
    tri = triangulate_landmarks(uv0, uv1, intr0, intr1, calib,
                                reference_bones=reference, bone_rel_tolerance=.2)
    assert "bone_length" in tri.reasons[CONNECTIONS[0][0]]

    series = np.stack([uv0, uv0 + 10])
    series[1, 8] = np.nan
    interpolated, valid = interpolate_landmarks([0.0, .02], series, .01, max_gap_sec=.03)
    np.testing.assert_allclose(interpolated[valid], (uv0 + 5)[valid])
    assert not valid[8] and np.isnan(interpolated[8]).all()
    assert interpolate_landmarks([0.0, .2], series, .1, max_gap_sec=.03) == (None, None)


def test_evaluation_summary_separates_agreement_from_accuracy():
    n = 5
    def trajectory(offset):
        points = np.tile(_RIGHT_HAND[None], (n, 1, 1)) + np.array([0, 0, .7]) + offset
        return MethodTrajectory(points, points[:, 0], np.tile([0., 0., 0., 1.], (n, 1)),
                                np.linalg.norm(points[:, 4] - points[:, 8], axis=1),
                                np.ones(n, bool), np.full(n, .2), np.full(n, 21))
    methods = {"B1_A": trajectory(np.array([.01, 0, 0])),
               "B1_B": trajectory(np.array([-.01, 0, 0])),
               "B2_nearest": trajectory(np.zeros(3)),
               "B2_interpolated": trajectory(np.zeros(3))}
    report = summarize_methods(methods, np.array(["static"] * n))
    assert "not accuracy" in report["accuracy_note"]
    assert report["static"]["methods"]["B1_A"]["agreement_to_B2_nearest_median_m"] == pytest.approx(.01)
    assert "palm_position_error_median_m" not in report["static"]["methods"]["B1_A"]

    gt = methods["B2_nearest"].palm_position_cam0
    with_gt = summarize_methods(methods, np.array(["static"] * n), gt)
    assert with_gt["static"]["methods"]["B1_A"]["palm_position_error_median_m"] == pytest.approx(.01)
