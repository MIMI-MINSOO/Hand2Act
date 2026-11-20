# Hand2Act — Codex instructions

## Goal and environment
Extend the existing Hand2Act repo rather than rewriting it. Initial hardware: **two fixed Intel RealSense RGB cameras (D435i, D415)** and a physically installed ChArUco board. Start with camera validation/calibration, then stereo hand pose, low-speed robot teleoperation, and synchronized robot-view demonstration data. All implementation runs locally.

## Context and token budget
- At the start of each session, read **only** `AGENTS.md`, `docs/PROGRESS.md`, and code/config/tests directly relevant to the current task. Read **relevant sections** of `docs/PROJECT_GUIDE.md` when needed, not the entire docs tree or all Markdown files automatically.
- `README_STEREO.md` is the concise user quickstart. `docs/PROJECT_GUIDE.md` holds architecture, coordinate conventions, milestone scope and calibration protocol. `docs/PROGRESS.md` holds the latest verified state and next step. Avoid duplicating long explanations across files.
- Before implementation: `git status --short`, inspect existing code, give a brief file-level plan. Implement only the requested milestone. After implementation: run appropriate tests and summarize actual results.
- **Markdown files are editable project artifacts, not read-only instructions.** Update `docs/PROGRESS.md` after every meaningful task. Update `docs/PROJECT_GUIDE.md` when the design/contracts change, `README_STEREO.md` when setup/CLI changes, and this file only if stable agent rules change. Make precise incremental edits, not wholesale rewrites. Keep progress concise: current status, evidence, blockers, next action; archive older detail in commits rather than accumulating a large session log.

## Non-negotiable technical rules
- Preserve the original monocular/B1 path, public APIs and existing tests. Add stereo separately; avoid broad refactors.
- Both tracking cameras are **fixed**. Calibrate stereo extrinsics once after mounting and redo if moved; **no runtime head-camera ChArUco pose estimation**. ChArUco need not appear in normal hand-tracking frames.
- Use **per-device RGB/color** intrinsics for the active stream, not depth/IR calibration. Get factory K, distortion coefficients and **distortion model** with `pyrealsense2`, then validate with actual ChArUco captures; refit only if warranted. Never fabricate factory RMS.
- Do not use RealSense depth/IMU to estimate hand pose. Depth may optionally serve as independently labeled evaluation-only ground truth.
- Do not assume physical board measurements match repo defaults. Confirm dictionary, squares X/Y, square size, marker size, legacy pattern and actual printed scaling. Earlier mention of a 40 mm board does not establish all properties.
- Stereo ChArUco correspondence must match the **same IDs in both views**. Preserve existing `CharucoTarget.detect()` compatibility if adding a detector that returns IDs.
- Raw distorted image coordinates, undistorted pixel coordinates and normalized rays are different; use the matching projection model. Fail clearly on unsupported distortion models instead of silently misinterpreting them.
- Coordinate convention: `T_A_B` maps points in B to A. `T_cam1_cam0` maps cam0 to cam1. OpenCV camera axes x right, y down, z forward. All metric 3D in meters, robot angles in radians, host monotonic timestamps in nanoseconds. Explicitly bind cam0/cam1 to serial numbers.
- Two RealSense sensor timestamps might use different time domains. Preserve sensor time/domain, frame number and host receive time separately. A host `read()` timestamp is **not** proven exposure time. For stereo calibration, capture the board stationary in each image pair until inter-camera synchronization is validated.
- Do not overwrite original observations, calibrations or uncommitted changes. Keep local camera serial settings and large video/calibration samples out of Git.
- Hardware commands only after simulation, limits, timeouts and physical E-stop are verified. Never trigger a real robot movement during camera calibration.

## Scope and reporting
M0 repo/device/board audit → M1 RGB factory intrinsics + validation → M2 fixed stereo R,t + holdout evaluation → M3 capture/time sync → M4 MediaPipe×2/triangulation → M5 simulation/robot teleop → M6 robot-view dataset.

Every Codex result must state: changed files, commands/tests actually run, measured results versus untested items, risks, and **one concrete next action**. Record the same in a compact form in `docs/PROGRESS.md`.
