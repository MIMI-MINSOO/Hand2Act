"""Serial-bound dual RealSense RGB capture with nearest-frame pairing."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    serial: str
    image_bgr: np.ndarray
    frame_number: int
    sensor_timestamp_ms: float
    timestamp_domain: str
    host_received_ns: int
    metadata: dict = field(default_factory=dict)


def shared_timestamp_domain(a, b):
    # hardware_clock is device-local even when both domain names match.
    return (a.timestamp_domain == b.timestamp_domain and
            a.timestamp_domain.split('.')[-1] in {'system_time', 'global_time'})


def frame_time_ns(frame: CameraFrame, shared_domain: bool) -> int:
    return (int(round(frame.sensor_timestamp_ms * 1e6))
            if shared_domain else frame.host_received_ns)


def nearest_frame_pair(frames0, frames1):
    """Return ``(i, j, gap_ms, clock)`` for the closest buffered pair."""
    if not frames0 or not frames1:
        return None
    shared = shared_timestamp_domain(frames0[0], frames1[0])
    best = min(((i, j, abs(frame_time_ns(a, shared) - frame_time_ns(b, shared)))
                for i, a in enumerate(frames0) for j, b in enumerate(frames1)),
               key=lambda x: x[2])
    return best[0], best[1], best[2] / 1e6, "sensor" if shared else "host_receive"


class DualRealSense:
    def __init__(self, serial0: str, serial1: str, width=640, height=480, fps=30,
                 queue_size=8):
        import pyrealsense2 as rs
        self.rs = rs
        self.serials = (str(serial0), str(serial1))
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.queues = [deque(maxlen=queue_size), deque(maxlen=queue_size)]
        self.pipelines = []
        self.profiles = []
        self.cond = threading.Condition()
        self.stop_event = threading.Event()
        self.threads = []
        self.errors = []
        self.host_history = [deque(maxlen=60), deque(maxlen=60)]
        self.queue_drops = [0, 0]

    def start(self):
        try:
            for serial in self.serials:
                pipeline, config = self.rs.pipeline(), self.rs.config()
                config.enable_device(serial)
                config.enable_stream(self.rs.stream.color, self.width, self.height,
                                     self.rs.format.bgr8, self.fps)
                self.profiles.append(pipeline.start(config))
                self.pipelines.append(pipeline)
        except Exception:
            self.close()
            raise
        for index in range(2):
            thread = threading.Thread(target=self._worker, args=(index,), daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def _worker(self, index):
        try:
            while not self.stop_event.is_set():
                color = self.pipelines[index].wait_for_frames(1000).get_color_frame()
                if not color:
                    continue
                metadata = {}
                for name in ('frame_timestamp', 'sensor_timestamp', 'actual_exposure',
                             'backend_timestamp', 'time_of_arrival'):
                    key = getattr(self.rs.frame_metadata_value, name, None)
                    if key is not None and color.supports_frame_metadata(key):
                        metadata[name] = int(color.get_frame_metadata(key))
                packet = CameraFrame(
                    self.serials[index], np.asanyarray(color.get_data()).copy(),
                    int(color.get_frame_number()), float(color.get_timestamp()),
                    str(color.get_frame_timestamp_domain()), time.monotonic_ns(), metadata)
                with self.cond:
                    if len(self.queues[index]) == self.queues[index].maxlen:
                        self.queue_drops[index] += 1
                    self.queues[index].append(packet)
                    self.host_history[index].append(packet.host_received_ns)
                    self.cond.notify_all()
        except RuntimeError as exc:
            if not self.stop_event.is_set():
                with self.cond:
                    self.errors.append((self.serials[index], str(exc)))
                    self.cond.notify_all()

    def read_event(self, timeout_sec=2.0):
        """Consume one raw frame, in host-arrival order, without pairing losses."""
        deadline = time.monotonic() + timeout_sec
        with self.cond:
            while not any(self.queues):
                if self.errors:
                    raise RuntimeError(str(self.errors))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('RGB frame timeout')
                self.cond.wait(remaining)
            index = min((i for i in range(2) if self.queues[i]),
                        key=lambda i: self.queues[i][0].host_received_ns)
            return index, self.queues[index].popleft()

    def read_nearest(self, timeout_sec=2.0, lookahead_frames=0):
        """Return a buffered nearest pair after limited future-frame lookahead.

        An optional one-frame lookahead prevents immediately pairing the first two frames
        merely because both queues became non-empty. It adds roughly one frame
        of online latency and may skip input frames, but makes the closest choice
        over observed neighbours. The zero-lookahead default preserves throughput.
        It still does not imply hardware/exposure synchronization.
        """
        deadline = time.monotonic() + timeout_sec
        required = max(int(lookahead_frames) + 1, 1)
        with self.cond:
            while not (len(self.queues[0]) >= required and len(self.queues[1]) >= required):
                if self.errors:
                    raise RuntimeError(f"RealSense capture failed: {self.errors}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for both RGB cameras")
                self.cond.wait(remaining)
            selected = nearest_frame_pair(self.queues[0], self.queues[1])
            i, j, gap_ms, clock = selected
            a = b = None
            for _ in range(i + 1):
                a = self.queues[0].popleft()
            for _ in range(j + 1):
                b = self.queues[1].popleft()
        return a, b, gap_ms, clock

    def measured_fps(self, index):
        values = self.host_history[index]
        if len(values) < 2:
            return 0.0
        return (len(values) - 1) / ((values[-1] - values[0]) / 1e9)

    def close(self):
        self.stop_event.set()
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        for thread in self.threads:
            thread.join(timeout=1.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
