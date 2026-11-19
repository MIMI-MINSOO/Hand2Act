"""프레임 소스: 영상 파일 / 카메라 인덱스 / 스트림 URL 공통 인터페이스."""

from __future__ import annotations

import os
import time

import cv2


def _open_windows_camera(index: int, width, height):
    """Windows 웹캠: MSMF 먼저, 요청 해상도가 안 잡히면 DirectShow.

    노트북 웹캠은 DirectShow + 1280x720에서 YUY2만 나와 10fps로 떨어지는 경우가 많음 (MSMF는 30fps).
    반대로 MSMF가 해상도 설정을 무시하는 장치도 있어 확인 후 fallback.
    """
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and (not (width and height) or frame.shape[1::-1] == (width, height)):
            return cap
        cap.release()
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)          # 마지막 수단: 해상도는 장치가 주는 대로
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class FrameSource:
    """for frame, t_sec, idx in FrameSource(src): ...

    - 파일: t = 영상 timestamp (CAP_PROP_POS_MSEC, 가변 프레임레이트 대응)
    - 카메라/URL: t = 수신 시각 (perf_counter 기준)
    """

    def __init__(self, src, stride: int = 1, max_frames: int | None = None,
                 width: int | None = None, height: int | None = None, latest_only: bool | None = None):
        self.is_file = isinstance(src, str) and not src.isdigit() and "://" not in src
        if str(src).isdigit() and os.name == "nt":
            self.cap = _open_windows_camera(int(src), width, height)
        else:
            self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
            if width and height and not self.is_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise IOError(f"소스를 열 수 없음: {src}")
        self.stride, self.max_frames = max(1, stride), max_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else -1
        # 실시간 소스: 처리가 카메라보다 느리면 드라이버 버퍼에 옛 프레임이 쌓여 지연이 커짐
        # → 별도 스레드가 계속 읽고 가장 최신 프레임만 넘김 (오래된 프레임은 버림)
        self.latest_only = (not self.is_file) if latest_only is None else latest_only
        self._grabber = None

    def __iter__(self):
        if self.latest_only:
            yield from self._iter_latest()
            return
        idx, out, t0 = -1, 0, time.perf_counter()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            if idx % self.stride:
                continue
            if self.is_file:
                ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                t = ms / 1000.0 if ms > 0 or idx == 0 else idx / self.fps
            else:
                t = time.perf_counter() - t0
            yield frame, t, idx
            out += 1
            if self.max_frames and out >= self.max_frames:
                break

    def _iter_latest(self):
        import threading
        cond, state = threading.Condition(), {"frame": None, "t": 0.0, "idx": -1, "stop": False, "eof": False}
        t0 = time.perf_counter()

        def grab():
            while not state["stop"]:
                ok, frame = self.cap.read()
                t = time.perf_counter() - t0          # 수신 시각 (처리 시각이 아님)
                with cond:
                    if not ok:
                        state["eof"] = True
                        cond.notify_all()
                        return
                    state["frame"], state["t"], state["idx"] = frame, t, state["idx"] + 1
                    cond.notify_all()

        self._grabber = (threading.Thread(target=grab, daemon=True), state)
        self._grabber[0].start()
        last, out = -1, 0
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: state["idx"] != last or state["eof"], timeout=5.0)
                    if state["idx"] == last:
                        break
                    frame, t, last = state["frame"], state["t"], state["idx"]
                yield frame, t, last
                out += 1
                if self.max_frames and out >= self.max_frames:
                    break
        finally:
            state["stop"] = True

    def close(self):
        if self._grabber is not None:
            self._grabber[1]["stop"] = True
            self._grabber[0].join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ImageFolderSource:
    """이미지 파일들(glob)을 프레임처럼 순회. t = idx / fps."""

    def __init__(self, pattern: str, stride: int = 1, max_frames: int | None = None, fps: float = 30.0):
        import glob
        self.files = sorted(glob.glob(pattern))[::max(1, stride)][:max_frames]
        if not self.files:
            raise IOError(f"이미지가 없음: {pattern}")
        self.fps, self.is_file, self.n_frames = fps, True, len(self.files)

    def __iter__(self):
        for i, f in enumerate(self.files):
            img = cv2.imread(f)
            if img is not None:
                yield img, i / self.fps, i

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def add_source_args(p, stride: int = 1) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="영상 파일")
    g.add_argument("--images", help="이미지 glob, 예: 'shots/*.jpg'")
    g.add_argument("--camera", help="카메라 인덱스 또는 스트림 URL")
    p.add_argument("--stride", type=int, default=stride, help="N프레임마다 1장 사용")
    p.add_argument("--max-frames", type=int, default=None)


def open_source(args):
    if args.images:
        return ImageFolderSource(args.images, args.stride, args.max_frames)
    return FrameSource(args.video or args.camera, args.stride, args.max_frames)
