# Hand2Act local progress (living state)

**Status date:** 2026-11-19. Two-camera RGB probing, factory export and physical board specification are verified; diverse-view calibration capture remains pending.

| Milestone | Status | Verified evidence / artifact |
|---|---|---|
| M0 Repo/camera/board audit | Complete | Both serial-bound RGB devices stream via SDK; firmware/USB/profiles and 5x7/40 mm/24 mm/200x280 mm board verified |
| M1 RGB intrinsics + validation | Partial | Both factory RGB intrinsics exported and 24 real frames captured/detected; reprojection metrics blocked by unverified board and non-diverse views |
| M2 Fixed stereo extrinsic | Capture ready | Explicit calibration/holdout directories, common-ID/outlier/pose deviation, reprojection and metric corner-distance reports implemented; diverse pairs not yet captured |
| M3 Capture and time sync | Partial | Nearest baseline, reference-ordered lookahead and guarded 2D interpolation implemented; same-raw replay and bounded live timing measured; exposure synchronization remains unverified |
| M4 Stereo hand pose/evaluation | Software complete, hardware blocked | Shared-cache B1-A/B1-B/B2 evaluation, trajectories/CSV/JSON/plot/video implemented; no real metric comparison without extrinsic and hand model |
| M5 Teleoperation | Not started | — |
| M6 Demonstration dataset | Not started | — |

## Current known setup
- Locally installed: fixed RealSense D435i and D415; physical ChArUco board.
- Verified devices: D435i `841512070365`, D415 `816612060658`; firmware `5.17.0.10`; both USB `2.1`.
- Verified board: `DICT_6X6_250`, 5x7, square 0.040 m, marker 0.024 m, 0.200x0.280 m, legacy pattern false; paper must remain flat.
- **Not yet verified**: multi-pose intrinsic/stereo reprojection accuracy and workspace-wide out-of-plane accuracy.
- Verified software: Python 3.11 virtual environment has `pyrealsense2 2.59.2`; branch is `feat/stereo-calibration`.
- Existing monocular code must remain compatible.

## Latest session (Codex updates in place)
- Date / task: 2026-11-19, stereo RGB time alignment
- Branch / commit: `feat/stereo-calibration`; implementation snapshot prepared for GitHub on 2026-11-20
- Files changed: RealSense timing/metadata, stereo time aligner, live/offline time alignment and recording, sync diagnostic/replay CLI, synthetic tests. Triangulation, MediaPipe model, IK and filters unchanged.
- Actual same-raw replay: `data/sync_20261119_02.npz`, 2.86 s, 86/75 frames, queue drops 0/0, report `results/sync_20261119_02.json`. Nearest: 26.22 pairs/s, gap median/p95/max 15.29/23.43/24.33 ms, unselected 11/0. Improved lookahead: 25.87 pairs/s, 12.06/19.01/19.76 ms, unselected 12/1, no reuse. Interpolate: 25.87 brackets/s, maximum endpoint distance 21.41/26.05/26.75 ms, unselected 12/0, 73 intentional endpoint reuses. Replay buffer-delay median/p95 16/31 ms for all modes; excludes actual processing.
- Actual separate 3 s headless live runs: nearest/lookahead/interpolate output 24.74/24.54/24.67 frame/s; host receive-to-output median/p95 31/47, 31/46.35, 32/47 ms. Artifacts: `data/sync_live_{nearest,lookahead,interpolate}_01`. These separate runs are not a same-motion comparison.
- Actual hand results: 0 detections in both views; all modes valid pose/joint ratios and valid pose FPS 0. Interpolation success 0%; static jitter and dynamic hand continuity unmeasured. The 25 valid-pose FPS target is not verified.
- Actual timestamp audit: common `system_time`; only backend timestamp and time-of-arrival metadata available in sampled RGB frames. Sensor timestamp/frame timestamp/actual exposure metadata unavailable. No LED event test or exposure offset/drift estimate performed. Short sensor-host residuals include delivery jitter and do not establish clock drift.
- Tests: full suite `50 passed, 8 skipped`; synthetic alignment tests include no reuse, explicit endpoint reuse, startup recovery, device-local clock fallback, missing/side-change/gap rejection, linear-motion geometry and known LED offset. CLI help, compilation and whitespace checks passed. Initial interrupted raw archive `_01` is incomplete and excluded; `_02` is the valid replay input.
- GitHub snapshot: source/config template/tests/docs included; `.venv`, raw data, calibration artifacts and local `configs/stereo.yaml` excluded. Tests use `stereo.example.yaml` so a fresh checkout does not depend on local serial settings.
- Risks: landmark visibility/persistent hand IDs are unavailable; hidden inferred joints cannot be identified directly. Host receive latency excludes unknown exposure/USB delay. Current default remains nearest pending hand validation; lookahead is the next candidate.
- **Next single action:** Record static and dynamic visible-hand trials using `stereo_sync_eval --capture-seconds 10 --data <new.npz> --experiment static|dynamic`, then replay the same raw data for all modes.

## Update rule
Rewrite only the **Latest session** section and milestone table after each task; avoid appending an ever-growing conversation log. If a decision affects architecture or calibration, update the relevant section in `PROJECT_GUIDE.md`. If an executable command changes, update `../README_STEREO.md`. Do not invent metrics or mark skipped hardware checks as passed.
