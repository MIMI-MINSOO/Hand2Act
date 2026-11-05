"""프레임 소스: 영상 파일 / 카메라 인덱스 / 스트림 URL 공통 인터페이스."""

from __future__ import annotations

import time

import cv2


class FrameSource:
    """for frame, t_sec, idx in FrameSource(src): ...

    - 파일: t = 영상 timestamp (CAP_PROP_POS_MSEC, 가변 프레임레이트 대응)
    - 카메라/URL: t = 수신 시각 (perf_counter 기준)
    """

    def __init__(self, src, stride: int = 1, max_frames: int | None = None,
                 width: int | None = None, height: int | None = None):
        self.is_file = isinstance(src, str) and not src.isdigit() and "://" not in src
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            raise IOError(f"소스를 열 수 없음: {src}")
        if width and height and not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stride, self.max_frames = max(1, stride), max_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else -1

    def __iter__(self):
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

    def close(self):
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
