# Hand2Act — fixed stereo RGB extension

This file supplements the repository's existing `README.md`; do not replace the original monocular instructions.

**Goal:** Two fixed RealSense color cameras (D435i + D415) → calibrated 3D hand/EE pose → MuJoCo/real robot teleoperation → synchronized robot wrist/base RGB + measured joints + commanded actions. ChArUco is used for one-time calibration and validation, not necessarily during normal runtime. The initial milestone is **camera/intrinsic validation**, not full teleoperation.

## Files
- `AGENTS.md`: compact Codex operating rules; automatically read at task start.
- `docs/PROJECT_GUIDE.md`: architecture, interface conventions, calibration, milestones.
- `docs/PROGRESS.md`: current verified status, open issues and next task; Codex edits this each session.
- `prompts/FIRST_CODEX_PROMPT.md`: initial M0–M1 task prompt.

## Windows PowerShell quickstart
```powershell
git clone https://github.com/MIMI-MINSOO/Hand2Act.git
cd Hand2Act
git switch -c feat/stereo-calibration
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
codex
```
Use an available Python version compatible with the installed packages if 3.11 is unavailable. Copy this starter pack **into the repo root** before starting Codex. Record failed commands as failures, not successful checks.

## First physical checks
1. Verify printed ChArUco pattern and **measured** square/marker lengths (do not trust the repo default blindly).
2. Probe both RealSense device serials, supported color streams, factory RGB intrinsics and distortion models.
3. Validate factory intrinsics against several board views, preserve images/results.
4. After review, acquire many **stationary-board** pairs and calibrate fixed stereo extrinsics using common ChArUco IDs.

## M0/M1 commands

Create the ignored local config before running stereo commands:

```powershell
Copy-Item configs/stereo.example.yaml configs/stereo.yaml
```

Skip this copy when `configs/stereo.yaml` already exists. Fill in camera serials
and the matching factory intrinsic file paths. The example leaves physical board
verification disabled; confirm the installed board and enter its measured values
in the local config. Raw captures, calibration JSON and local config stay out of Git.

The probe is serial-bound (never numeric camera-index-bound). It lists color
profiles, selects a common profile (preferring 1280x720@30), exports one factory
RGB intrinsic JSON per serial, and records frame number, sensor timestamp/domain,
host `monotonic_ns`, measured rate/jitter and frame-number gaps.

```powershell
.\.venv\Scripts\python.exe -m handdata.apps.realsense_probe --list-only
.\.venv\Scripts\python.exe -m handdata.apps.realsense_probe `
  --serial 841512070365 --serial 816612060658 `
  --duration 10 --capture-count 12 --capture-every 20 `
  --out-dir calib/realsense_probe
```

`configs/stereo.yaml` extends the unchanged `default.yaml` and carries the board
values from `configs/charuco_A4_5x7_40mm.txt`: 5x7 squares, 40 mm square,
24 mm marker and `DICT_6X6_250`. Before validation, measure the printed board
width/height and square/marker lengths, enter them under `board.verification`,
confirm every verification flag, and set `board.physical_verified: true`.
Measured square/marker lengths override the TXT reference when they differ.
Measure the printed board rectangle (expected 200x280 mm), preferably measure
several adjacent squares and divide, then measure marker side length. Do not use
the A4 paper edge (210x297 mm) as the board rectangle.

```powershell
.\.venv\Scripts\python.exe -m handdata.calib.intrinsics `
  --config configs/stereo.yaml `
  --images "calib_raw/cam0/*.png" `
  --validate-factory calib/realsense_probe/intrinsics_color_SERIAL_640x480_30.json `
  --report calib/cam0_factory_validation.json
```

Add `--save-refit calib/cam0_charuco_refit.json` only for an independent
comparison; it never overwrites the factory file. Both cameras were verified at
640x480@30 after fully closing RealSense Viewer; a residual Viewer process keeps
the UVC interface busy.

## M2/M3 commands

The installed board is verified as 5x7, 40 mm square, 24 mm marker and
200x280 mm. Keep the paper flat. Capture about 30 unobstructed calibration
pairs, then 8-10 independent holdout pairs in a different directory
(`s`/space saves, `q` exits):

```powershell
.\.venv\Scripts\python.exe -m handdata.apps.stereo_capture `
  --config configs/stereo.yaml --out calib_raw/stereo_calibration
.\.venv\Scripts\python.exe -m handdata.apps.stereo_capture `
  --config configs/stereo.yaml --out calib_raw/stereo_holdout
.\.venv\Scripts\python.exe -m handdata.calib.stereo_extrinsics `
  --config configs/stereo.yaml `
  --pairs calib_raw/stereo_calibration `
  --holdout-pairs calib_raw/stereo_holdout `
  --out calib/stereo_extrinsics.json --report calib/stereo_validation.json
.\.venv\Scripts\python.exe -m handdata.apps.stereo_live `
  --config configs/stereo.yaml --record-dir data/stereo_live_001
```

Live output is in the D435i/cam0 frame, never robot base. Nearest timestamp
pairing invalidates gaps above `stereo.max_pair_gap_ms`; equal `system_time`
domains do not prove exposure synchronization.

XY translation and in-plane rotation on a flat table are acceptable for an
initial calibration, but weakly constrain out-of-plane behavior. They do not
establish equal accuracy throughout the 3D workspace. If possible, add flat,
rigidly supported captures at other heights and tilts without bending the paper.

`stereo.pair_lookahead_frames=0` is the real-time default. Setting it to `1`
selects over one future frame and can reduce timestamp gap, but adds latency and
may skip many input frames. Keep the setting and selected frame numbers in each
evaluation report.

## Mono vs stereo evaluation

Record each static/dynamic/pinch/occlusion trial with `stereo_live`, then run all
three methods from the same saved images and shared MediaPipe detection cache:

```powershell
.\.venv\Scripts\python.exe -m handdata.apps.stereo_eval `
  --config configs/stereo.yaml `
  --recording data/stereo_live_001 --out results/stereo_live_001 `
  --experiment static --video
```

Use `--segments segments.csv` (`start_frame,end_frame,experiment`) for a mixed
recording and `--reuse-detections` for repeatable reruns. Outputs are
`frames.csv`, `summary.json`, `trajectories.npz`, `comparison.png`, and optional
`comparison.mp4`. Without an independently measured `--gt` NPZ, the report
labels 3D comparisons as stability/consistency/disagreement, never accuracy.
