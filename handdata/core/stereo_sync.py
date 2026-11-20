"""Bounded, reference-camera time alignment, shared by live and replay.

No exposure synchronization is implied. Endpoint reuse for interpolation is
intentional; each cam0 reference produces at most one output.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .realsense_stereo import frame_time_ns, nearest_frame_pair, shared_timestamp_domain
from .stereo import interpolate_landmarks
from .tracker import HandDetection


@dataclass
class SyncPair:
    reference: object
    left: object
    right: object
    alpha: float
    gap_ms: float
    clock: str
    released_ns: int


class FrameSynchronizer:
    def __init__(self, mode='nearest', max_wait_ms=60., max_span_ms=50., capacity=32,
                 max_nearest_gap_ms=20.):
        if mode not in ('nearest', 'lookahead', 'interpolate'):
            raise ValueError('unknown sync mode')
        self.mode, self.wait_ns = mode, int(max_wait_ms * 1e6)
        self.max_span_ms, self.capacity = max_span_ms, capacity
        self.max_nearest_gap_ns = int(max_nearest_gap_ms * 1e6)
        self.queues = [deque(), deque()]
        self.skipped = [0, 0]
        self.expired = 0
        self.used = [set(), set()]
        self.reuses = 0

    def push(self, camera, packet, now_ns=None):
        q = self.queues[camera]
        q.append(packet)
        if len(q) > self.capacity:
            q.popleft()
            self.skipped[camera] += 1
        return self.drain(packet.host_received_ns if now_ns is None else now_ns)

    def drain(self, now_ns):
        output = []
        q0, q1 = self.queues
        while q0 and q1:
            a = q0[0]
            shared = shared_timestamp_domain(a, q1[0])
            clock = 'sensor' if shared else 'host_receive'
            timestamp = lambda p: frame_time_ns(p, shared)
            ta = timestamp(a)
            if self.mode == 'nearest':
                i, j, gap, clock = nearest_frame_pair(q0, q1)
                for _ in range(i):
                    q0.popleft()
                    self.skipped[0] += 1
                a = q0.popleft()
                for _ in range(j):
                    q1.popleft()
                    self.skipped[1] += 1
                b = q1.popleft()
                left = right = b
                alpha = 0.
            else:
                # Keep only the last predecessor. Do not discard cam0 references
                # to chase the globally smallest gap in the two buffers.
                while len(q1) >= 2 and timestamp(q1[1]) <= ta:
                    old = q1.popleft()
                    if old.frame_number not in self.used[1]:
                        self.skipped[1] += 1
                future = next((p for p in q1 if timestamp(p) >= ta), None)
                expired = now_ns - a.host_received_ns >= self.wait_ns
                if future is None and not expired:
                    break
                if self.mode == 'interpolate':
                    left, right = q1[0], future
                    if (right is None or timestamp(left) > ta or
                            (timestamp(right) - timestamp(left)) / 1e6 > self.max_span_ms):
                        q0.popleft()
                        self.expired += 1
                        continue
                    span = timestamp(right) - timestamp(left)
                    alpha = (ta - timestamp(left)) / span if span else 0.
                    gap = max(abs(ta - timestamp(left)), abs(timestamp(right) - ta)) / 1e6
                else:
                    candidates = list(q1)
                    b = min(candidates, key=lambda p: abs(timestamp(p) - ta))
                    if abs(timestamp(b) - ta) > self.max_nearest_gap_ns:
                        # Startup/asymmetric stream stalls must not reserve
                        # every future frame for an already obsolete reference.
                        q0.popleft()
                        self.expired += 1
                        continue
                    left = right = b
                    alpha = 0.
                    gap = abs(timestamp(b) - ta) / 1e6
                    # One-to-one selection: never select b again. Leave future
                    # candidates for later references rather than draining both.
                    while q1 and q1[0] is not b:
                        q1.popleft()
                        self.skipped[1] += 1
                    q1.popleft()
                q0.popleft()
            self.used[0].add(a.frame_number)
            for b in {p.frame_number: p for p in (left, right)}.values():
                if b.frame_number in self.used[1]:
                    self.reuses += 1
                self.used[1].add(b.frame_number)
            output.append(SyncPair(a, left, right, alpha, gap, clock, now_ns))
        return output


def interpolate_detection(pair, detections, max_motion_palm=0.75):
    """Conservative single-hand association; MediaPipe exposes no track ID here.

    Refuse detection gaps, side changes, unmatched handedness and large jumps.
    Visibility is unavailable: nonfinite joints are masked, but hidden joints
    inferred confidently by MediaPipe cannot be identified by this function.
    """
    d0, dl, dr = detections
    if any(d is None for d in detections):
        return None, 'missing_detection'
    if not (d0.side == dl.side == dr.side and all(d.side_matched for d in detections)):
        return None, 'hand_identity_or_side'
    palm = [0, 5, 9, 13, 17]
    scale = max(np.linalg.norm(dl.uv[0] - dl.uv[9]), 1.)
    if (not np.isfinite(dl.uv[palm]).all() or not np.isfinite(dr.uv[palm]).all() or
            np.max(np.linalg.norm(dr.uv[palm] - dl.uv[palm], axis=1)) > max_motion_palm * scale):
        return None, 'hand_discontinuity'
    uv, valid = interpolate_landmarks([0., 1.], np.stack([dl.uv, dr.uv]), pair.alpha,
                                       max_gap_sec=1.)
    uv[~valid] = np.nan
    return HandDetection(uv, dl.uvz.copy(), dl.world.copy(), dl.side,
                         min(dl.score, dr.score), True), ''
