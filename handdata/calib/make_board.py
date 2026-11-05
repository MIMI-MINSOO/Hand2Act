"""인쇄용 ChArUco 보드 PNG 생성.

    python -m handdata.calib.make_board --out calib/charuco_A4.png

100% 배율(실제 크기)로 인쇄한 뒤 사각형 한 칸 크기를 자로 재서 configs의 board.square_length,
marker_length를 실측값으로 고칠 것. 평평한 판(폼보드 등)에 붙이면 정확도가 좋아짐.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget

PAPER_MM = {"A4": (210, 297), "A3": (297, 420), "letter": (216, 279)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--out", default="calib/charuco_A4.png")
    p.add_argument("--paper", default="A4", choices=list(PAPER_MM))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--hand-marker-id", type=int, default=None, help="손등 GT용 ArUco 마커도 같이 출력")
    p.add_argument("--hand-marker-mm", type=float, default=30.0)
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    b = cfg.board
    target = CharucoTarget(b)
    px_per_mm = a.dpi / 25.4
    board_mm = (b.squares_x * b.square_length * 1000, b.squares_y * b.square_length * 1000)
    paper_mm = PAPER_MM[a.paper]
    if board_mm[0] > paper_mm[0] - 10 or board_mm[1] > paper_mm[1] - 10:
        raise SystemExit(f"보드 {board_mm[0]:.0f}x{board_mm[1]:.0f}mm가 {a.paper}에 안 들어감")

    board_px = (int(round(board_mm[0] * px_per_mm)), int(round(board_mm[1] * px_per_mm)))
    img = target.generate_image(board_px, 0)
    W, H = int(round(paper_mm[0] * px_per_mm)), int(round(paper_mm[1] * px_per_mm))
    canvas = np.full((H, W), 255, np.uint8)
    y0, x0 = (H - board_px[1]) // 2, (W - board_px[0]) // 2
    canvas[y0:y0 + board_px[1], x0:x0 + board_px[0]] = img
    label = (f"ChArUco {b.squares_x}x{b.squares_y} {b.dictionary} square={b.square_length * 1000:.1f}mm "
             f"marker={b.marker_length * 1000:.1f}mm  | print at 100% scale")
    cv2.putText(canvas, label, (x0, max(40, y0 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)

    out = cfg.resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"저장: {out}  ({a.paper}, {a.dpi}dpi, 보드 {board_mm[0]:.0f}x{board_mm[1]:.0f}mm)")

    # 손등 GT 평가용 단일 ArUco 마커 (보드 id와 겹치지 않게), 별도 A4
    if a.hand_marker_id is not None:
        n_board_markers = (b.squares_x * b.squares_y) // 2
        if a.hand_marker_id < n_board_markers:
            raise SystemExit(f"--hand-marker-id는 보드 마커 id(0~{n_board_markers - 1})와 겹치면 안 됨")
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b.dictionary))
        m_px = int(round(a.hand_marker_mm * px_per_mm))
        page = np.full((H, W), 255, np.uint8)
        y, x = (H - m_px) // 2, (W - m_px) // 2
        page[y:y + m_px, x:x + m_px] = cv2.aruco.generateImageMarker(d, a.hand_marker_id, m_px)
        cv2.putText(page, f"{b.dictionary} id={a.hand_marker_id}  {a.hand_marker_mm:.0f}mm (black square)",
                    (x - 200, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
        mp = out.with_name(out.stem + f"_hand_marker{a.hand_marker_id}.png")
        cv2.imwrite(str(mp), page)
        print(f"손등 마커: {mp}  (검은 정사각형 한 변 실측해서 eval_marker_gt --marker-size에 입력)")


if __name__ == "__main__":
    main()
