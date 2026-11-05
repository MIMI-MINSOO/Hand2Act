"""카메라 intrinsic (K, distortion) 로드/저장과 투영·역투영."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms: float = float("nan")
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Intrinsics":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["K"], float), np.array(d["dist"], float).reshape(-1),
                   int(d["width"]), int(d["height"]), float(d.get("rms", "nan")), d.get("meta", {}))

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "K": self.K.tolist(), "dist": self.dist.tolist(),
            "width": self.width, "height": self.height, "rms": self.rms, "meta": self.meta,
        }, indent=2))

    @property
    def f(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2)

    def check_frame(self, frame: np.ndarray) -> None:
        """캘리브레이션 해상도/방향과 다르면 K가 틀리므로 바로 에러."""
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"영상 해상도 {w}x{h} ≠ intrinsic {self.width}x{self.height}. "
                "같은 해상도·방향(세로/가로)으로 캘리브레이션해야 합니다.")

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        """픽셀 (N,2) → 왜곡 제거된 정규화 좌표 (N,2)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """픽셀 (N,2) → 카메라 좌표계 단위 광선 (N,3)."""
        xy = self.normalized(uv)
        r = np.hstack([xy, np.ones((len(xy), 1))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, P_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표계 3D (N,3) → 픽셀 (N,2), 왜곡 포함."""
        P = np.asarray(P_cam, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)
