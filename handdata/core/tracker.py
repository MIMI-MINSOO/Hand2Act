"""MediaPipe HandLandmarker 래퍼 (Tasks API, VIDEO 모드).

출력 HandDetection
- uv    (21,2): 픽셀 좌표 (PnP 입력)
- uvz   (21,3): (x px, y px, z·W px). MediaPipe 상대 깊이를 x와 같은 스케일로 둔 의사 3D.
                축이 카메라와 정렬되어 있어 PnP flip 해 고를 때 손바닥 법선 힌트로 사용
- world (21,3): MediaPipe world landmark (m, 손 중심 원점). 손가락 상대 형상에 사용
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HandDetection:
    uv: np.ndarray
    uvz: np.ndarray
    world: np.ndarray
    side: str            # 실제 손 ('right'/'left'), 미러링 보정 후
    score: float
    side_matched: bool   # 설정한 손(hand.side)과 일치 여부


def _flip(side: str) -> str:
    return "left" if side == "right" else "right"


class HandTracker:
    def __init__(self, model_path, target_side="right", input_mirrored=False,
                 accept_other_side=True, min_detection_conf=0.5, min_presence_conf=0.5,
                 min_tracking_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                                   RunningMode)
        self._mp = mp
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,  # 2개까지 검출 후 target_side 선택
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._lm = HandLandmarker.create_from_options(opts)
        self.target_side = target_side
        self.input_mirrored = input_mirrored
        self.accept_other_side = accept_other_side
        self._last_ms = -1

    @classmethod
    def from_config(cls, cfg):
        h = cfg.hand
        return cls(cfg.resolve(cfg.paths.mediapipe_model), h.side, h.input_mirrored,
                   h.get("accept_other_side", True), h.get("min_detection_conf", 0.5),
                   h.get("min_presence_conf", 0.5), h.get("min_tracking_conf", 0.5))

    def detect(self, bgr: np.ndarray, t_sec: float) -> HandDetection | None:
        ms = int(round(t_sec * 1000))
        if ms <= self._last_ms:          # VIDEO 모드는 timestamp가 엄격히 증가해야 함
            ms = self._last_ms + 1
        self._last_ms = ms

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                       data=rgb), ms)
        if not res.hand_landmarks:
            return None

        cands = []
        for i, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0]
            # MediaPipe는 입력이 미러링(셀피)이라고 가정하고 handedness를 냄 → 후면 카메라면 뒤집기
            side = cat.category_name.lower()
            if not self.input_mirrored:
                side = _flip(side)
            uvz = np.array([[p.x * w, p.y * h, p.z * w] for p in lms])
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]])
            cands.append(HandDetection(uvz[:, :2].copy(), uvz, world, side, float(cat.score),
                                       side == self.target_side))

        matched = [c for c in cands if c.side_matched]
        if matched:
            return max(matched, key=lambda c: c.score)
        if self.accept_other_side and len(cands) == 1:
            return cands[0]   # 한 손만 보이면 handedness 오분류로 보고 사용 (side_matched=False 기록)
        return None

    def close(self) -> None:
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
