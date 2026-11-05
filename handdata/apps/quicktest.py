"""캘리브레이션 없이 노트북 웹캠으로 EE pose 추출을 바로 확인하는 테스트 모드.

    python -m handdata.apps.quicktest --wrist-middle 0.09

- intrinsics.json이 없거나 해상도가 다르면 → 화각(--hfov) 가정으로 근사 K 사용
  (깊이 절대값은 f 오차만큼 비례해서 틀림. 상대 움직임·자세·그리퍼는 확인 가능)
- 손 모델이 없으면 → 시작 후 손을 펴서 카메라를 향해 1~2초 들고 있으면 자 방식으로 즉석 생성
  (--wrist-middle: 손목 주름 근처 ~ 가운뎃손가락 뿌리 관절 길이를 자로 잰 값 [m])
- ChArUco 보드가 화면에 보이면 base 기준 pose도 같이 표시

화면: 빨간 점 = 손바닥 PnP 점, 초록 원 = 손 모델 재투영(겹쳐야 정상), 좌표축 = palm frame
우측 하단 = 위에서 내려다본 EE 궤적 (카메라 x–z, 손을 앞뒤로 움직이면 위아래로 움직여야 함)
키: k 손 모델 다시 잡기 · r 기록 시작/정지 · m 미러 화면 전환 · q 종료
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..calib.hand_model import model_from_observations
from ..config import add_config_args, load_config
from ..core.board import CharucoTarget
from ..core.camera import Intrinsics
from ..core.geometry import quat_from_R
from ..core.lifting import HandModel, hand_points_in_palm_frame
from ..core.tracker import HandTracker
from ..core.video import FrameSource
from ..dataio.episode import EpisodeBuffer, save_episode
from ..pipeline import NO_HAND, PNP_FAIL, FrameProcessor
from ..viz import draw_frame, put_lines  # 화면 글자는 ASCII만 (OpenCV putText는 한글 불가)


def approx_intrinsics(w: int, h: int, hfov_deg: float) -> Intrinsics:
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return Intrinsics(K, np.zeros(5), w, h, meta={"approx_hfov_deg": hfov_deg})


def get_intrinsics(cfg, frame, hfov):
    h, w = frame.shape[:2]
    p = cfg.resolve(cfg.paths.intrinsics)
    if p.exists():
        intr = Intrinsics.load(p)
        if (intr.width, intr.height) == (w, h):
            return intr, f"캘리브레이션 K ({p.name})"
    return approx_intrinsics(w, h, hfov), f"근사 K (HFOV {hfov:.0f}° 가정, f={(w / 2) / np.tan(np.radians(hfov) / 2):.0f}px)"


class TopView:
    """카메라 위에서 내려다본 EE 궤적: 가로 = x, 세로 = 깊이 z."""

    def __init__(self, size=220, x_range=(-0.3, 0.3), z_range=(0.15, 0.9), n=90):
        self.size, self.xr, self.zr = size, x_range, z_range
        self.pts = deque(maxlen=n)

    def add(self, p):
        self.pts.append(p)

    def draw(self, img, mirror: bool = False):
        s = self.size
        sx = -1.0 if mirror else 1.0     # 미러 화면이면 좌우도 같이 뒤집어 화면과 방향 일치
        H, W = img.shape[:2]
        x0, y0 = W - s - 12, H - s - 12
        roi = img[y0:y0 + s, x0:x0 + s]
        roi[:] = (roi * 0.35).astype(np.uint8)
        to_px = lambda p: (int((sx * p[0] - self.xr[0]) / (self.xr[1] - self.xr[0]) * s),
                           int(s - (p[2] - self.zr[0]) / (self.zr[1] - self.zr[0]) * s))
        for z in np.arange(0.2, 0.9, 0.1):
            y = to_px((0, 0, z))[1]
            cv2.line(roi, (0, y), (s, y), (70, 70, 70), 1)
            cv2.putText(roi, f"{z:.1f}m", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        pts = [to_px(p) for p in self.pts]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(roi, a, b, (0, 200, 255), 2)
        if pts:
            cv2.circle(roi, pts[-1], 5, (0, 0, 255), -1)
        cv2.putText(roi, "top view (cam x-z)", (4, s - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--camera", default="0", help="웹캠 인덱스 (외장 카메라면 1, 2 ...) 또는 스트림 URL")
    p.add_argument("--video", default=None, help="웹캠 대신 영상 파일로 테스트")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hfov", type=float, default=65.0, help="근사 K용 가로 화각 [deg]. 노트북 웹캠 보통 60~75")
    p.add_argument("--wrist-middle", type=float, default=0.09, help="자로 잰 wrist–middle MCP 길이 [m]")
    p.add_argument("--hand-model", default=None, help="이미 만든 hand_model.json 사용")
    p.add_argument("--calib-frames", type=int, default=30)
    p.add_argument("--out", default="data/quicktest")
    p.add_argument("--no-window", action="store_true", help="화면 없이 콘솔 출력만 (원격/헤드리스)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--print-every", type=float, default=0.5, help="콘솔 출력 간격 [s]")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    side = cfg.hand.side
    board = CharucoTarget(cfg.board)
    model = HandModel.load(a.hand_model) if a.hand_model else None
    intr = proc = buf = None
    obs, mirror, last_print, rec_t0 = [], True, -np.inf, 0.0
    top = TopView()

    with HandTracker.from_config(cfg) as tracker, \
            FrameSource(a.video or a.camera, max_frames=a.max_frames, width=a.width, height=a.height) as src:
        for frame, t, _ in src:
            if intr is None:
                intr, how = get_intrinsics(cfg, frame, a.hfov)
                print(f"영상 {frame.shape[1]}x{frame.shape[0]}, {how}")
                if model is None:
                    print("손을 펴서 카메라를 향해 들고 1~2초 유지하세요 (손 모델 생성)")
            det = tracker.detect(frame, t)
            lines, color = [], (80, 255, 80)

            # ---------------------------------------------------- 1) 즉석 손 모델
            if model is None:
                if det is not None:
                    obs.append(hand_points_in_palm_frame(det.world, side))
                lines = [f"[HAND MODEL] open your hand toward the camera and hold  {len(obs)}/{a.calib_frames}",
                         f"scale: wrist-middle MCP = {a.wrist_middle * 100:.1f} cm (--wrist-middle)"]
                color = (0, 220, 255)
                if len(obs) >= a.calib_frames:
                    model = model_from_observations(obs, side, "ruler", 0.0, a.wrist_middle,
                                                    {"source": "quicktest"})
                    proc = None
                    print(f"손 모델 생성 완료: 손바닥 폭 {model.meta['palm_width_m'] * 100:.1f}cm, "
                          f"프레임 간 흔들림 {np.mean(np.array(model.std_mm)):.1f}mm")
                r = None
            # ---------------------------------------------------- 2) EE pose
            else:
                if proc is None:
                    proc = FrameProcessor(cfg, intr, model)
                r = proc.process(t, det, board.pose(frame, intr))
                if not r.flags & (NO_HAND | PNP_FAIL):
                    T_cam_ee = proc.retarget.ee_pose(r.T_cam_hand, r.kp3d_cam)
                    pc = T_cam_ee[:3, 3]
                    rpy = Rotation.from_matrix(T_cam_ee[:3, :3]).as_euler("xyz", degrees=True)
                    top.add(pc)
                    lines = [f"EE[cam]  x={pc[0] * 100:+6.1f}  y={pc[1] * 100:+6.1f}  z={pc[2] * 100:6.1f} cm",
                             f"rpy[cam] {rpy[0]:+6.0f} {rpy[1]:+6.0f} {rpy[2]:+6.0f} deg",
                             f"gripper {r.gripper:4.2f} (d={r.pinch_dist * 100:4.1f}cm)  "
                             f"reproj {r.reproj_px:4.1f}px  flip={r.choice}"]
                    if r.valid:
                        pb = r.ee_pos
                        lines.append(f"EE[base] x={pb[0] * 100:+6.1f}  y={pb[1] * 100:+6.1f}  z={pb[2] * 100:6.1f} cm (board)")
                    if r.reproj_px > cfg.lifting.max_reproj_px:
                        color = (0, 140, 255)
                    if t - last_print >= a.print_every:
                        last_print = t
                        print(f"t={t:6.2f}s  cam xyz=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:.3f})m  "
                              f"rpy=({rpy[0]:+.0f},{rpy[1]:+.0f},{rpy[2]:+.0f})  grip={r.gripper:.2f}  "
                              f"reproj={r.reproj_px:.1f}px")
                    if buf is not None:
                        r.extra = (pc, quat_from_R(T_cam_ee[:3, :3]))
                else:
                    lines, color = ["no hand" if r.flags & NO_HAND else "PnP failed"], (80, 80, 255)
                if buf is not None:
                    r.t = t - rec_t0
                    buf.add(r)

            if a.no_window:
                continue
            vis = draw_frame(frame.copy(), intr, r, model.palm, hud=False) if r is not None else frame.copy()
            if det is not None and r is None:
                for uv in det.uv.astype(int):
                    cv2.circle(vis, tuple(uv), 4, (0, 220, 255), -1)
            if mirror:
                vis = cv2.flip(vis, 1)                      # 거울처럼 보이게 (글자는 뒤집기 후에 그림)
            top.draw(vis, mirror)
            if buf is not None:
                lines.append(f"REC {len(buf)} frames")
            put_lines(vis, lines + ["k: redo hand model  r: record  m: mirror  q: quit"], color)
            cv2.imshow("Hand2Act quicktest", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                mirror = not mirror
            if k == ord("k"):
                model, obs, proc = None, [], None
                print("손 모델 다시 생성")
            if k == ord("r") and model is not None:
                if buf is None:
                    buf, rec_t0 = EpisodeBuffer(), t
                    print("● 기록 시작")
                else:
                    _save(buf, a.out, cfg, intr, model)
                    buf = None

    if buf is not None and len(buf):
        _save(buf, a.out, cfg, intr, model)
    cv2.destroyAllWindows()


def _save(buf: EpisodeBuffer, out_root, cfg, intr, model):
    arr = buf.to_arrays()
    T = len(buf.rows)
    arr["ee_pos_cam"] = np.full((T, 3), np.nan)
    arr["ee_quat_cam"] = np.full((T, 4), np.nan)
    for i, r in enumerate(buf.rows):
        if hasattr(r, "extra"):
            arr["ee_pos_cam"][i], arr["ee_quat_cam"][i] = r.extra
    out = Path(out_root) / time.strftime("qt_%Y%m%d_%H%M%S")
    save_episode(out, arr, {"source": {"quicktest": True}, "intrinsics": {"K": intr.K.tolist(), **intr.meta},
                            "hand_model": {"method": model.method, "wrist_middle_m": model.wrist_middle},
                            "config": cfg.to_plain()})
    print(f"■ 저장: {out}  (카메라 기준 {np.sum(~np.isnan(arr['ee_pos_cam'][:, 0]))}/{T} 프레임)")


if __name__ == "__main__":
    main()
