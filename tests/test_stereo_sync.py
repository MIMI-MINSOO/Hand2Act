import numpy as np

from handdata.core.realsense_stereo import CameraFrame, nearest_frame_pair
from handdata.core.stereo_sync import FrameSynchronizer, interpolate_detection
from handdata.core.tracker import HandDetection
from handdata.apps.stereo_sync_eval import led_diagnostic
from handdata.core.stereo import StereoCalibration, triangulate_landmarks
from .synthetic import _RIGHT_HAND, camera


def packet(c, i, ms, domain='system_time'):
    return CameraFrame(str(c), np.zeros((2, 2, 3), np.uint8), i, ms,
                       domain, int((ms + 1000) * 1e6))


def test_device_local_clocks_are_not_compared():
    a = packet(0, 1, 0, 'hardware_clock')
    b = packet(1, 1, 10000, 'hardware_clock')
    b.host_received_ns = a.host_received_ns + 2_000_000
    assert nearest_frame_pair([a], [b])[2:] == (2., 'host_receive')


def test_lookahead_preserves_reference_and_no_endpoint_reuse():
    sync = FrameSynchronizer('lookahead')
    events = [(0, packet(0, i, i * 33.3)) for i in range(100)]
    events += [(1, packet(1, i, i * 33.3 + 10)) for i in range(100)]
    events.sort(key=lambda x: x[1].host_received_ns)
    pairs = []
    for c, p in events:
        pairs.extend(sync.push(c, p))
    assert len(pairs) == 100
    assert len({p.reference.frame_number for p in pairs}) == 100
    assert sync.reuses == 0 and sync.skipped == [0, 0]


def test_interpolation_brackets_and_rejects_large_gap_and_side_switch():
    sync = FrameSynchronizer('interpolate')
    assert not sync.push(1, packet(1, 0, 0))
    assert not sync.push(0, packet(0, 0, 10))
    pair, = sync.push(1, packet(1, 1, 30))
    assert abs(pair.alpha - 1 / 3) < 1e-6
    uv = np.zeros((21, 2)); uv[9] = [100, 0]
    def det(uv, side='right'):
        return HandDetection(uv, np.zeros((21, 3)), np.zeros((21, 3)), side, 1., True)
    result, reason = interpolate_detection(pair, (det(uv), det(uv), det(uv + 6)))
    np.testing.assert_allclose(result.uv, uv + 2)
    assert not reason
    assert interpolate_detection(pair, (det(uv), det(uv), None))[0] is None
    assert interpolate_detection(pair, (det(uv), det(uv), det(uv, 'left')))[0] is None
    # No extrapolation at the beginning or across a missing-frame gap.
    s = FrameSynchronizer('interpolate')
    s.push(0, packet(0, 0, 0))
    assert not s.push(1, packet(1, 0, 10))
    assert s.expired == 1
    s.push(0, packet(0, 1, 20))
    assert not s.push(1, packet(1, 1, 100))
    assert s.expired == 2


def test_interpolation_reference_once_endpoints_intentionally_shared():
    s = FrameSynchronizer('interpolate')
    events = [(0, packet(0, i, i * 33 + 10)) for i in range(20)]
    events += [(1, packet(1, i, i * 33)) for i in range(21)]
    events.sort(key=lambda x: x[1].host_received_ns)
    pairs = []
    for c, p in events:
        pairs.extend(s.push(c, p))
    assert len(pairs) == 20
    assert len(s.used[0]) == 20 and len(s.used[1]) == 21
    assert s.reuses == 19


def test_lookahead_recovers_after_asymmetric_startup():
    s = FrameSynchronizer('lookahead')
    for i in range(10):
        s.push(0, packet(0, i, i * 33.3))
    pairs = s.push(1, packet(1, 0, 310))
    assert all(p.gap_ms <= 60 for p in pairs)
    assert s.expired >= 8


def test_led_requires_events_and_recovers_known_offset():
    t = np.arange(300) / 30
    signal = np.where((np.arange(300) % 60) < 3, 255., 0.)
    report = led_diagnostic(t, signal, t + .02, signal)
    assert report['matched_events'] >= 3
    assert abs(report['offset_ms'] - 20) < 1e-6
    assert abs(report['drift_ppm']) < 1e-6
    assert led_diagnostic(t, np.zeros(300), t, np.zeros(300))['offset_ms'] is None


def test_linear_motion_interpolation_restores_same_time_geometry():
    intr = camera(640, 480, 620.)
    calib = StereoCalibration('0', '1', np.eye(3), np.array([-.18, 0, 0]), 640, 480)
    def points(ms):
        return _RIGHT_HAND + np.array([ms * .0005, 0, .7])
    uv0 = intr.project(points(10))
    uvl = intr.project(points(0) + calib.t)
    uvr = intr.project(points(30) + calib.t)
    def d(uv):
        return HandDetection(uv, np.zeros((21, 3)), np.zeros((21, 3)), 'right', 1., True)
    s = FrameSynchronizer('interpolate')
    s.push(1, packet(1, 0, 0))
    s.push(0, packet(0, 0, 10))
    pair, = s.push(1, packet(1, 1, 30))
    aligned, _ = interpolate_detection(pair, (d(uv0), d(uvl), d(uvr)))
    tri = triangulate_landmarks(uv0, aligned.uv, intr, intr, calib)
    assert tri.valid.all()
    np.testing.assert_allclose(tri.points_cam0, points(10), atol=1e-8)
