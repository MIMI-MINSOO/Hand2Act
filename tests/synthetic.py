"""테스트용 합성 카메라·손·보드."""

from __future__ import annotations

import numpy as np

from handdata.core.camera import Intrinsics
from handdata.core.geometry import make_T
from handdata.core.landmarks import PALM
from handdata.core.lifting import HandModel, hand_points_in_palm_frame
from handdata.core.tracker import HandDetection

# 오른손, palm frame 근사 좌표 [m] (x: 손가락 방향, y: 엄지 반대쪽, z: 손바닥 쪽)
# 펴진 손. 실제 손 비율을 대략 흉내냄
_RIGHT_HAND = np.array([
    [0.000, 0.000, 0.0],     # 0 wrist
    [0.025, -0.025, 0.005],  # 1 thumb cmc
    [0.045, -0.045, 0.008],  # 2 thumb mcp
    [0.065, -0.060, 0.008],  # 3 thumb ip
    [0.085, -0.072, 0.008],  # 4 thumb tip
    [0.090, -0.022, 0.0],    # 5 index mcp
    [0.130, -0.026, 0.0],    # 6
    [0.155, -0.028, 0.0],    # 7
    [0.177, -0.030, 0.0],    # 8 index tip
    [0.092, 0.000, 0.0],     # 9 middle mcp
    [0.137, 0.000, 0.0],
    [0.165, 0.000, 0.0],
    [0.190, 0.000, 0.0],     # 12
    [0.087, 0.020, 0.0],     # 13 ring mcp
    [0.127, 0.023, 0.0],
    [0.152, 0.025, 0.0],
    [0.174, 0.026, 0.0],     # 16
    [0.078, 0.038, 0.0],     # 17 pinky mcp
    [0.108, 0.044, 0.0],
    [0.126, 0.047, 0.0],
    [0.143, 0.049, 0.0],     # 20
])


def camera(w=1920, h=1080, f=1500.0, dist=None) -> Intrinsics:
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return Intrinsics(K, np.zeros(5) if dist is None else np.asarray(dist, float), w, h)


def hand_model(side="right") -> HandModel:
    P = hand_points_in_palm_frame(_RIGHT_HAND.copy(), "right")
    return HandModel.from_points(P, side, method="synthetic")


def random_pose(rng, dist=(0.35, 0.6)) -> np.ndarray:
    """카메라 앞, 손바닥이 대체로 카메라 쪽을 보는 pose."""
    from scipy.spatial.transform import Rotation
    # 기본: 손가락은 화면 위쪽(-y cam), 손바닥 법선은 카메라 쪽(-z cam)
    R0 = np.stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]], axis=1)
    dR = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix()
    t = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.1), rng.uniform(*dist)])
    return make_T(dR @ R0, t)


def detection(model: HandModel, intr: Intrinsics, T_cam_hand: np.ndarray, rng=None,
              noise_px=0.0, world_noise=0.0) -> HandDetection:
    P_cam = model.all21 @ T_cam_hand[:3, :3].T + T_cam_hand[:3, 3]
    uv = intr.project(P_cam)
    if rng is not None and noise_px > 0:
        uv = uv + rng.normal(0, noise_px, uv.shape)
    # MediaPipe 이미지 상대 깊이 흉내: (x px, y px, (Z - Z_wrist) * f)
    uvz = np.hstack([uv, ((P_cam[:, 2] - P_cam[0, 2]) * intr.f)[:, None]])
    # MediaPipe world landmark 흉내: 카메라 축 정렬, 손 중심 원점, scale 약간 틀림(평균 손)
    world = (P_cam - P_cam[PALM].mean(0)) * 0.93
    if rng is not None and world_noise > 0:
        world = world + rng.normal(0, world_noise, world.shape)
    return HandDetection(uv, uvz, world, model.side, 0.99, True)
