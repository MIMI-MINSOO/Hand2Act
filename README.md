# hand-pose-data

머리에 단 카메라(iPhone) 영상에서 맨손을 추적해 **로봇 base 기준 EE 위치·자세(6DoF) + 그리퍼 개폐(1)** 를
metric으로 뽑아 로봇 데이터로 저장하는 파이프라인. [hand-control-robot](https://github.com/advaithdasu/hand-control-robot)
구조를 기준으로 하되, teleop용 상대 매핑을 **PnP lifting 기반 절대 pose**로 바꿈.

```
iPhone(머리) ─▶ MediaPipe 2D 21kp ─▶ 손바닥 PnP (IPPE + LM) ─▶ T_cam_hand
             └▶ ChArUco PnP ─────────────────────────────────▶ T_cam_board
   T_base_ee = T_base_board · T_cam_board⁻¹ · T_cam_hand · (손가락 3D → pinch point, 축 대응)
   gripper   = 엄지–검지 끝 3D 거리, 캘리브레이션 범위로 [0,1] 정규화
```

| | 레퍼런스 (teleop) | 이 프로젝트 (데이터) |
|---|---|---|
| 깊이 | 손 크기 비율 heuristic | 실측 손 모델 + 손바닥 PnP (metric) |
| 손 크기 | 평균 손 | ChArUco 평면 역투영으로 사용자 손 실측 (또는 자) |
| 기준 좌표 | calibration 대비 상대, clutch | 보드 = 로봇 base, 절대 좌표 |
| 카메라 | 고정 | 머리 착용 → 매 프레임 보드 PnP로 ego-motion 상쇄 |
| 필터 | One Euro (causal) | 실시간 One Euro + 오프라인 양방향 Savitzky–Golay |

## 설치

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12 권장
pip install -r requirements.txt
bash scripts/download_model.sh                          # MediaPipe 모델
python -m pytest tests -q                               # 합성 데이터 테스트 18개
```

## 사용 순서

모든 명령은 프로젝트 루트에서 실행. 설정은 `configs/default.yaml`, 일회성 변경은 `--set key=value`.

**0) 보드 출력**
```bash
python -m handdata.calib.make_board --out calib/charuco_A4.png --hand-marker-id 40
```
A4에 100% 배율로 인쇄 → 폼보드 등 평평한 판에 부착 → 사각형 한 칸을 자로 재서
`board.square_length`, `board.marker_length` 수정 (이 값이 모든 metric의 기준).

**1) 카메라 intrinsic** — 수집 때와 같은 해상도·방향·렌즈로, 보드를 화면 곳곳·여러 각도로 20~30초 촬영
```bash
python -m handdata.calib.intrinsics --video calib_raw/intrinsic.mov      # → calib/intrinsics.json
```

**2) 손 모델** — 손을 보드 옆 책상에 쫙 펴서 올리고, 보드와 손이 함께 보이게 위에서 3~5초 (카메라 각도 조금씩 바꿔가며)
```bash
python -m handdata.calib.hand_model --video calib_raw/hand_flat.mov      # → calib/hand_model.json
# 비교용 자 방식
python -m handdata.calib.hand_model --method ruler --ruler-wrist-middle 0.088 \
    --video calib_raw/hand_flat.mov --out calib/hand_model_ruler.json
```
출력되는 wrist–middle MCP 길이를 자로 잰 값과 비교해서 sanity check. `calib/hand_model_check.jpg`로 검출 확인.

**3) 그리퍼 범위** — 손을 공중에 두고 엄지·검지를 최대로 벌렸다 붙였다 5~10회
```bash
python -m handdata.calib.gripper_range --video calib_raw/pinch.mov       # → calib/gripper_range.json
```

**4) base frame** — 보드를 로봇 기준 고정 위치에 두고 `board.board_in_base`에 보드 원점(좌상단 코너)의 base 기준
위치·자세 입력. 기본값(`rpy_deg: [180,0,0]`)은 책상에 눕힌 보드의 z(아래)를 base z(위)로 뒤집는 것.

**5) 수집**
```bash
# 오프라인 (권장): 녹화 → 데이터
python -m handdata.apps.offline --video data/raw/ep001.mov --out data/ep001 --preview
# 실시간: 카메라/스트림 → MuJoCo 시뮬 teleop + 기록 (r: 기록, c: 리셋, q: 종료)
python -m handdata.apps.live --camera 0 --sim
```

**6) 후처리** — 짧은 결측 보간 → 양방향 smoothing → IK feasibility
```bash
python -m handdata.apps.postprocess data/ep001                          # → episode_proc.npz
```

**7) 평가**
```bash
# 책상 위 손 GT (2D→3D 복원 오차만 분리 측정)
python -m handdata.apps.eval_board_gt --video eval/hand_table.mov --out results/pnp.json
# 손등 ArUco 마커 (공중 일관성)
python -m handdata.apps.eval_marker_gt --video eval/marker_hand.mov --marker-id 40 --marker-size 0.030
```

## Ablation 예시

검출은 한 번만 돌리고(`detections.npz` 캐시) lifting 설정만 바꿔 재처리:
```bash
python -m handdata.apps.offline --video data/raw/ep001.mov --out data/ep001 --reuse-detections \
    --set lifting.finger_method=ray_bone --name episode_raybone.npz
```

| 항목 | 설정 키 | 값 |
|---|---|---|
| lifting 방법 | `lifting.method` | `pnp` / `size_ratio` (레퍼런스 방식 baseline) |
| PnP 해법 | `lifting.pnp_method` | `ippe` / `sqpnp` / `iterative` |
| 손가락 복원 | `lifting.finger_method` | `umeyama` / `ray_bone` |
| 손 모델 | `paths.hand_model` | board 방식 / ruler 방식 json |
| 손 두께 보정 | `hand_model.plane_offset` | 0 ~ 0.015 m |

## 구조

```
handdata/
  core/      landmarks, geometry(SE3·palm frame·Umeyama), camera, board(ChArUco·광선-평면 교차),
             tracker(MediaPipe), lifting(PnP·손가락·baseline), filters, video
  calib/     make_board, intrinsics, hand_model, gripper_range
  robot/     retarget(손→EE·그리퍼), kinematics(MuJoCo 범용), ik(DLS)
  dataio/    episode (npz + json 저장/로드)
  apps/      offline, live, postprocess, eval_board_gt, eval_marker_gt
  pipeline.py  프레임 단위 처리 (이미지와 분리 → live/offline/테스트 공용)
  viz.py       오버레이
configs/default.yaml   assets/demo_arm.xml   tests/
```

## 저장 포맷

`episode.npz` (프레임 T개): `t, valid, flags, ee_pos(T,3), ee_quat(T,4 xyzw), gripper(T,), pinch_dist,
reproj_px, T_base_ee, T_base_hand, T_cam_hand, T_cam_board (T,4,4), kp3d_base, kp3d_cam (T,21,3), uv (T,21,2)`
`episode_proc.npz`: 보간·smoothing된 `ee_pos, ee_quat, gripper` + IK `q, ik_ok, ik_pos_err`.
`meta.json`: 좌표계·단위 규약, 설정 스냅샷, 플래그 통계. 결측은 NaN.

flags (bitmask): 1 no_hand · 2 no_board · 4 pnp_fail · 8 high_reproj · 16 interpolated · 32 ik_fail · 64 side_mismatch

## 로봇 교체 (PiPER / SO-101 등)

`robot.mjcf`, `robot.joints`, `robot.ee_site`, `robot.home_q`, `robot.gripper_joints`만 바꾸면 됨
(MJCF는 mujoco_menagerie 등에서). EE site 축 규약이 다르면 `retarget.tool_axes_in_palm` / `extra_rpy_deg`로 맞춤.
MuJoCo world = 로봇 base로 가정.

## 촬영 체크리스트 (iPhone)

- 해상도·fps·렌즈(1x) 고정, **영상 안정화와 자동 렌즈 전환 끄기** (켜져 있으면 K가 변함)
- 카메라 설정 → 포맷 → "높은 호환성"(H.264), HDR 비디오 끄기 (OpenCV 호환)
- 캘리브레이션과 수집 영상의 방향(세로/가로)이 같아야 함 — 다르면 `check_frame`이 에러를 냄
- 수집 중 보드가 항상 일부라도 보이게 (ChArUco는 부분 가림 허용, 코너 6개 이상)

## 알려진 한계

- 손바닥 강체·평면 가정: 주먹 쥐면 MCP 아치가 생겨 오차 증가 → `eval_marker_gt`로 정량화
- 정면 근처에서 IPPE flip ambiguity: 이전 프레임 연속성 → MediaPipe 상대 깊이 법선 순으로 해결
  (합성 테스트에서 이전 프레임 정보 없이 200회 중 10회 오선택)
- 깊이 오차 ∝ 거리²: 손을 멀리 뻗을수록 나빠짐 (`eval_board_gt`의 거리 구간별 결과 참고)
- 물체를 쥐면 손가락·손바닥 가림 → 재투영 오차 플래그로 거르고 짧은 결측만 보간
