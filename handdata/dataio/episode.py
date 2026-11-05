"""저장 포맷 (npz + json).

<episode_dir>/
  detections.npz  : MediaPipe·보드 검출 캐시 (lifting 설정만 바꿔 재처리할 때 재사용)
  episode.npz     : 프레임별 결과 (아래 KEYS)
  episode_proc.npz: postprocess 결과 (보간·smoothing·IK)
  meta.json       : 좌표계·단위 규약, 설정 스냅샷, 통계
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..core.board import BoardPose
from ..core.tracker import HandDetection
from ..pipeline import FLAG_NAMES, FrameResult

CONVENTIONS = {
    "units": "meter, radian",
    "quat": "xyzw (scipy)",
    "T_a_b": "4x4, p_a = T_a_b @ p_b",
    "frames": {
        "cam": "OpenCV 카메라 (x 오른쪽, y 아래, z 앞)",
        "board": "ChArUco 보드 (원점 좌상단 코너, z = 보드 뒷면)",
        "base": "로봇 base = T_base_board(config) · board",
        "hand": "palm frame (원점 wrist, x→middle MCP, z 손바닥 쪽 법선)",
        "ee": "tool frame (retarget.tool_axes_in_palm / orientation 설정)",
    },
    "gripper": "0=닫힘, 1=열림 (엄지–검지 거리를 캘리브레이션 범위로 정규화)",
    "flags": {str(k): v for k, v in FLAG_NAMES.items()},
}

# ------------------------------------------------------------------ 검출 캐시


def save_detections(path, ts, dets: list, boards: list, meta: dict) -> None:
    T = len(ts)
    a = {
        "t": np.asarray(ts, float),
        "hand_found": np.array([d is not None for d in dets]),
        "uv": np.full((T, 21, 2), np.nan), "uvz": np.full((T, 21, 3), np.nan),
        "world": np.full((T, 21, 3), np.nan), "score": np.full(T, np.nan),
        "side_matched": np.zeros(T, bool), "side": np.array([""] * T, dtype="<U5"),
        "board_found": np.array([b is not None for b in boards]),
        "T_cam_board": np.full((T, 4, 4), np.nan), "board_n": np.zeros(T, int),
        "board_reproj": np.full(T, np.nan),
    }
    for i, (d, b) in enumerate(zip(dets, boards)):
        if d is not None:
            a["uv"][i], a["uvz"][i], a["world"][i] = d.uv, d.uvz, d.world
            a["score"][i], a["side_matched"][i], a["side"][i] = d.score, d.side_matched, d.side
        if b is not None:
            a["T_cam_board"][i], a["board_n"][i], a["board_reproj"][i] = b.T_cam_board, b.n_corners, b.reproj_px
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **a)


def load_detections(path):
    """→ (ts, dets, boards, meta)."""
    z = np.load(path, allow_pickle=False)
    dets, boards = [], []
    for i in range(len(z["t"])):
        dets.append(HandDetection(z["uv"][i], z["uvz"][i], z["world"][i], str(z["side"][i]),
                                  float(z["score"][i]), bool(z["side_matched"][i]))
                    if z["hand_found"][i] else None)
        boards.append(BoardPose(z["T_cam_board"][i], int(z["board_n"][i]), float(z["board_reproj"][i]))
                      if z["board_found"][i] else None)
    return z["t"], dets, boards, json.loads(str(z["meta"]))

# ------------------------------------------------------------------ 에피소드


class EpisodeBuffer:
    def __init__(self):
        self.rows: list[FrameResult] = []

    def add(self, r: FrameResult) -> None:
        self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def to_arrays(self) -> dict:
        R = self.rows
        T_ee = np.stack([r.T_base_ee for r in R]) if R else np.zeros((0, 4, 4))
        return {
            "t": np.array([r.t for r in R]),
            "valid": np.array([r.valid for r in R]),
            "flags": np.array([r.flags for r in R], np.int32),
            "ee_pos": T_ee[:, :3, 3].copy(),
            "ee_quat": np.stack([r.ee_quat for r in R]) if R else np.zeros((0, 4)),
            "gripper": np.array([r.gripper for r in R]),
            "pinch_dist": np.array([r.pinch_dist for r in R]),
            "reproj_px": np.array([r.reproj_px for r in R]),
            "T_base_ee": T_ee,
            "T_base_hand": np.stack([r.T_base_hand for r in R]) if R else np.zeros((0, 4, 4)),
            "T_cam_hand": np.stack([r.T_cam_hand for r in R]) if R else np.zeros((0, 4, 4)),
            "T_cam_board": np.stack([r.T_cam_board for r in R]) if R else np.zeros((0, 4, 4)),
            "kp3d_base": np.stack([r.kp3d_base for r in R]) if R else np.zeros((0, 21, 3)),
            "kp3d_cam": np.stack([r.kp3d_cam for r in R]) if R else np.zeros((0, 21, 3)),
            "uv": np.stack([r.uv for r in R]) if R else np.zeros((0, 21, 2)),
        }


def flag_summary(flags: np.ndarray) -> dict:
    return {name: int(np.sum((flags & bit) > 0)) for bit, name in FLAG_NAMES.items()}


def save_episode(out_dir, arrays: dict, meta: dict, name: str = "episode.npz") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / name, **arrays)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "conventions": CONVENTIONS,
            "n_frames": int(len(arrays["t"])), "n_valid": int(np.sum(arrays["valid"])),
            "flag_counts": flag_summary(arrays["flags"]), **meta}
    mp = out / "meta.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old[name.replace(".npz", "")] = meta
    mp.write_text(json.dumps(old, indent=2, ensure_ascii=False, default=str))
    return out / name


def load_episode(out_dir, name: str = "episode.npz") -> dict:
    z = np.load(Path(out_dir) / name)
    return {k: z[k] for k in z.files}
