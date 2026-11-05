"""ChArUco 보드: 검출, pose 추정, 광선-평면 교차.

보드 좌표계(OpenCV 규약): 원점 = 보드 좌상단 코너, x = 열 방향, y = 행 방향,
z = x × y (보드 뒷면 쪽, 즉 책상에 놓으면 아래쪽).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import T_from_rvec_tvec, apply_T, inv_T


@dataclass
class BoardPose:
    T_cam_board: np.ndarray
    n_corners: int
    reproj_px: float


class CharucoTarget:
    def __init__(self, cfg):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        self.square = float(cfg.square_length)
        self.board = cv2.aruco.CharucoBoard(
            (int(cfg.squares_x), int(cfg.squares_y)), self.square, float(cfg.marker_length), dictionary)
        if cfg.get("legacy_pattern", False):
            self.board.setLegacyPattern(True)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.min_corners = int(cfg.get("min_corners", 6))

    # ------------------------------------------------------------ 검출
    def detect(self, img: np.ndarray):
        """→ (obj_points (N,3), img_points (N,2)) 또는 None."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            return None
        obj, imgp = self.board.matchImagePoints(corners, ids)
        obj = np.asarray(obj, np.float64).reshape(-1, 3)
        imgp = np.asarray(imgp, np.float64).reshape(-1, 2)
        if not self._spread_ok(obj):
            return None  # 코너가 한 줄에 몰리면 pose가 퇴화함
        return obj, imgp

    def _spread_ok(self, obj: np.ndarray) -> bool:
        hull = cv2.convexHull(obj[:, :2].astype(np.float32))
        return cv2.contourArea(hull) > 2.0 * self.square ** 2

    # ------------------------------------------------------------ pose
    def pose(self, img: np.ndarray, intr: Intrinsics) -> BoardPose | None:
        found = self.detect(img)
        if found is None:
            return None
        obj, imgp = found
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, imgp, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        i = int(np.argmin(np.asarray(errs).reshape(-1)))
        rvec, tvec = cv2.solvePnPRefineLM(obj, imgp, intr.K, intr.dist,
                                          rvecs[i].copy(), tvecs[i].copy())
        proj, _ = cv2.projectPoints(obj, rvec, tvec, intr.K, intr.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1))))
        return BoardPose(T_from_rvec_tvec(rvec, tvec), len(obj), err)

    def generate_image(self, size_px: tuple[int, int], margin_px: int) -> np.ndarray:
        return self.board.generateImage(size_px, marginSize=margin_px)


def backproject_to_plane(uv: np.ndarray, intr: Intrinsics, T_cam_board: np.ndarray,
                         offset: float = 0.0) -> np.ndarray:
    """픽셀 (N,2)의 광선을 보드 평면(카메라 쪽으로 offset[m]만큼 띄운 면)과 교차 → 보드 좌표계 3D (N,3).

    손을 책상에 펴 놓으면 MediaPipe keypoint(관절 중심)는 책상 면보다 손 두께의 절반쯤 위에 있음.
    offset은 그 높이를 보정하기 위한 값.
    """
    T_board_cam = inv_T(T_cam_board)
    C = T_board_cam[:3, 3]                                    # 보드 좌표계에서 본 카메라 중심
    d = intr.rays(uv) @ T_board_cam[:3, :3].T                 # 보드 좌표계 광선 방향
    z0 = np.sign(C[2]) * offset                               # 카메라가 있는 쪽으로 offset
    s = (z0 - C[2]) / d[:, 2]
    return C + s[:, None] * d


def board_to_cam(T_cam_board: np.ndarray, P_board: np.ndarray) -> np.ndarray:
    return apply_T(T_cam_board, P_board)
