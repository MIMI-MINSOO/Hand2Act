# First local Codex prompt — M0 + M1

You are in my local checkout of `MIMI-MINSOO/Hand2Act`. Two **fixed** RealSense cameras (D435i and D415) and a physical ChArUco target are already installed. All later capture, stereo tracking, teleoperation and dataset collection will be developed locally. Use **both cameras' RGB streams** to estimate hand geometry, not depth/IR or IMU.

**Task: implement only M0 and M1** (repo/devices audit, RGB factory intrinsic export, physical target verification and measured ChArUco reprojection validation). Do not implement stereo extrinsics, MediaPipe stereo, robot SDK, IK or data export yet.

Read `AGENTS.md`, `docs/PROGRESS.md`, the **M0–M2 calibration and contracts sections** of `docs/PROJECT_GUIDE.md`, and only the relevant existing source/config/tests. Check `git status --short` and current test baseline; respect all uncommitted changes. First give a short file-level plan, then implement.

1. Inspect `handdata/core/camera.py`, `handdata/core/board.py`, `handdata/calib/intrinsics.py`, `handdata/core/video.py`, `handdata/config.py`, `configs/default.yaml`, requirements and existing tests. Preserve monocular APIs/tests.
2. Add a small documented RealSense camera-probe tool: report name, serial, firmware, color stream profiles, active color K/dist/**model**, frame number, sensor timestamp/domain, host receive timestamp. Test simultaneous streaming at a common supported resolution/FPS (prefer 1280×720@30 if both support it) and report **measured** rates/jitter/drops, not just configured FPS. Do not hardcode numeric camera indices.
3. Export a **per-device factory RGB intrinsic** compatible with existing `Intrinsics` (extra provenance in metadata/sidecar). No made-up factory RMS. Explicitly reject unknown or unsupported distortion conversion.
4. Verify physical ChArUco specs against config before any calibration. If dictionary/actual printed dimensions are unknown, produce an explicit input checklist and block calibration, **not** invented values. Validate intrinsic accuracy using multiple board positions/angles, with counts, reprojection median/p95, optional independent refit saved separately, and captured evidence.
5. Add focused deterministic tests for serialization/config checks and source compatibility; separate optional live-camera tests. Use `configs/stereo.example.yaml` with existing `extends` support if appropriate, without changing old defaults.
6. **You may edit Markdown files.** Update `docs/PROGRESS.md` with true results, `README_STEREO.md` with *tested* CLI usage, `docs/PROJECT_GUIDE.md` only for verified design changes, and `AGENTS.md` only if stable rules need correction. Do not indiscriminately load/rewrite every `.md` file.

Conclude with files changed, exact PowerShell commands, tests executed/passed/failed/skipped, real hardware measurements vs pending validation, and **one next physical action**. Leave M2 for a separate session and do not move the real robot.
