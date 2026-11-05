"""사용자 손 모델(metric) 만들기.

[board] 체커보드 평면 방식 (기본)
    손을 ChArUco 보드 옆 책상 위에 쫙 펴서 올려놓고 위에서 3~5초 촬영 (손 위치·카메라 각도 조금씩 바꿔가며).
    보드 PnP로 책상 평면을 알고 있으므로, 각 keypoint 픽셀의 광선과 평면(+손 두께 offset)의 교차점 = metric 3D.
    → 손바닥 5점 형상 + 21점 뼈 길이를 직접 실측.

    python -m handdata.calib.hand_model --video calib_raw/hand_flat.mov

[ruler] 자 방식 (비교용)
    MediaPipe world landmark 형상을 그대로 쓰고, 자로 잰 wrist–middle MCP 길이로 scale만 맞춤.

    python -m handdata.calib.hand_model --method ruler --ruler-wrist-middle 0.088 --video hand_any.mov \
        --out calib/hand_model_ruler.json
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget, backproject_to_plane
from ..core.camera import Intrinsics
from ..core.landmarks import CHAINS, INDEX_MCP, MIDDLE_MCP, PALM, PINKY_MCP, WRIST
from ..core.lifting import HandModel, hand_points_in_palm_frame
from ..core.tracker import HandTracker
from ..core.video import add_source_args, open_source
from ..viz import draw_axes


def points_on_table(uv21, intr: Intrinsics, T_cam_board, offset: float, side: str) -> np.ndarray:
    """책상 위 손 21점 픽셀 → palm frame metric 좌표 (21,3)."""
    P_board = backproject_to_plane(uv21, intr, T_cam_board, offset)
    return hand_points_in_palm_frame(P_board, side)


def model_from_observations(obs: list, side: str, method: str, offset: float,
                            ruler_len: float | None = None, meta: dict | None = None) -> HandModel:
    """여러 프레임 palm frame 점들 → median 모델."""
    A = np.stack(obs)                                     # (N,21,3)
    M = np.median(A, axis=0)
    if method == "ruler":
        M = M * (ruler_len / np.linalg.norm(M[MIDDLE_MCP] - M[WRIST]))
        A = A * (ruler_len / np.linalg.norm(A[:, MIDDLE_MCP] - A[:, WRIST], axis=1))[:, None, None]
    std_mm = (np.linalg.norm(A - M, axis=2).std(axis=0) * 1000).tolist()
    flat_mm = float(np.abs(M[PALM, 2]).max() * 1000)      # 손바닥 5점이 평면에서 벗어난 정도
    meta = dict(meta or {}, palm_flatness_mm=flat_mm,
                palm_width_m=float(np.linalg.norm(M[PINKY_MCP] - M[INDEX_MCP])))
    return HandModel.from_points(M, side, method=method, h_offset=offset, n_frames=len(obs),
                                 std_mm=std_mm, meta=meta)


def print_summary(m: HandModel) -> None:
    print(f"\n손 모델 ({m.method}, {m.side}, {m.n_frames} frames, offset={m.h_offset * 1000:.0f}mm)")
    print(f"  wrist–middle MCP : {m.wrist_middle * 1000:6.1f} mm   ← 자로 재서 비교해 보세요")
    print(f"  index–pinky MCP  : {m.meta['palm_width_m'] * 1000:6.1f} mm")
    print(f"  손바닥 평탄도     : {m.meta['palm_flatness_mm']:6.1f} mm")
    bl = m.bone_lengths
    names = ["thumb", "index", "middle", "ring", "pinky"]
    for name, chain in zip(names, CHAINS):
        segs = [bl[(a, b)] * 1000 for a, b in zip(chain[:-1], chain[1:])]
        print(f"  {name:6s} 뼈 길이 : " + " / ".join(f"{s:4.1f}" for s in segs) + " mm")
    std = np.array(m.std_mm)
    print(f"  프레임 간 흔들림(std): 손바닥 {std[PALM].mean():.1f} mm, 전체 {std.mean():.1f} mm")
    if std[PALM].mean() > 3:
        print("  ⚠ 흔들림이 큼: 손을 완전히 펴서 책상에 붙였는지, 보드가 함께 잘 보이는지 확인")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    add_source_args(p, stride=3)
    p.add_argument("--method", choices=["board", "ruler"], default="board")
    p.add_argument("--offset", type=float, default=None, help="책상~관절중심 높이 [m]")
    p.add_argument("--ruler-wrist-middle", type=float, default=None, help="ruler 방식: wrist–middle MCP 실측 [m]")
    p.add_argument("--out", default=None)
    p.add_argument("--debug-image", default="calib/hand_model_check.jpg")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.method == "ruler" and not a.ruler_wrist_middle:
        raise SystemExit("--method ruler 에는 --ruler-wrist-middle 필요")
    intr = Intrinsics.load(cfg.resolve(cfg.paths.intrinsics))
    board = CharucoTarget(cfg.board)
    side = cfg.hand.side
    offset = cfg.hand_model.plane_offset if a.offset is None else a.offset
    if a.max_frames is None:
        a.max_frames = cfg.hand_model.max_frames * 3

    obs, mismatch, last = [], 0, None
    with HandTracker.from_config(cfg) as tracker, open_source(a) as src:
        for frame, t, idx in src:
            intr.check_frame(frame)
            det = tracker.detect(frame, t)
            if det is None:
                continue
            mismatch += not det.side_matched
            if a.method == "board":
                bp = board.pose(frame, intr)
                if bp is None:
                    continue
                obs.append(points_on_table(det.uv, intr, bp.T_cam_board, offset, side))
                last = (frame.copy(), det, bp)
            else:
                obs.append(hand_points_in_palm_frame(det.world, side))
            if len(obs) >= cfg.hand_model.max_frames:
                break

    print(f"사용 프레임 {len(obs)}개 (handedness 불일치 {mismatch})")
    if len(obs) < 10:
        raise SystemExit("유효 프레임이 10개 미만. 손과 보드가 동시에 잘 보이게 다시 촬영하세요.")
    model = model_from_observations(obs, side, a.method, offset if a.method == "board" else 0.0,
                                    a.ruler_wrist_middle,
                                    {"source": a.video or a.images or a.camera,
                                     "square_length": cfg.board.square_length})
    out = cfg.resolve(a.out or cfg.paths.hand_model)
    model.save(out)
    print_summary(model)
    print(f"저장: {out}")

    if last is not None and a.debug_image:
        img, det, bp = last
        draw_axes(img, intr, bp.T_cam_board, 0.08, 3)
        for i, uv in enumerate(det.uv.astype(int)):
            cv2.circle(img, tuple(uv), 5, (0, 0, 255) if i in PALM else (255, 255, 0), -1)
        path = cfg.resolve(a.debug_image)
        cv2.imwrite(str(path), img)
        print(f"확인용 이미지: {path}")


if __name__ == "__main__":
    main()
