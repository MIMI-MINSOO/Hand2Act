# Project guide — architecture, calibration and interfaces

This is a **living** technical guide. Codex may update only the relevant sections as implementation is verified. For actual progress and test evidence see `PROGRESS.md`.

## Architecture / milestones
Two fixed RGB cameras (D435i and D415, camera roles determined by serial) → per-camera timestamped frames → stereo pairing/sync → MediaPipe 2D×2 → 3D triangulation → palm frame + pinch → retargeting/filter/IK → MuJoCo → optional real robot. Robot wrist/base RGB plus actual robot feedback and command timestamps form the demonstration dataset. Stereo camera-to-camera calibration **does not** provide robot-base calibration; `T_base_cam0` must be measured independently.

| Stage | Deliverable / acceptance evidence |
|---|---|
| M0 | Existing tests, Git state, detected serials/firmware/RGB modes, physical board specification |
| M1 | Separate factory RGB K/dist/model files, ChArUco validation median/p95 and saved samples |
| M2 | Fixed stereo `T_cam1_cam0`, common-ID matched pairs, held-out epipolar/reprojection errors, baseline sanity |
| M3 | Continuous two-camera capture, raw frames/timestamps, drops/jitter, independently measured clock offset |
| M4 | Stereo 3D hand pose, cheirality/reprojection/angle/bone-length checks, monocular comparison |
| M5 | Simulation first; robot driver, safety and measured end-to-end control delay |
| M6 | Synchronized robot-view RGB, actual state and commanded action, re-loadable dataset |

Later three-person ownership: **A** capture/sync/recording/export, **B** stereo geometry/tracking/evaluation, **C** retargeting/IK/robot/integration. Shared contracts need review. Only implement the current milestone.

## Calibration protocol (M0–M2)
**Board:** Confirm actual printed dictionary, `squares_x`, `squares_y`, `square_length_m`, `marker_length_m`, OpenCV/legacy pattern, print scale and physical flatness. Existing default may refer to 5×7 squares, 35 mm squares, 26 mm markers and `DICT_4X4_50`: **not verified** for this rig. Never infer unspecified marker length/dictionary from a stated 40 mm square size.

For stereo calibration, use `--config configs/stereo.yaml`; it extends the unchanged monocular default and records the TXT reference as 5×7, 40 mm squares, 24 mm markers and `DICT_6X6_250`. Validation remains blocked until board width/height and square/marker lengths are physically measured. Consistent physical measurements are authoritative when they differ from the reference.

**RGB device probe:** Identify both cameras by serial, firmware and supported RGB color modes. Prefer a common resolution/FPS such as 1280×720@30 **only if supported by both**. Run simultaneous color streams, save K, `dist`, model and active resolution/FPS, sensor timestamps/domain, frame numbers and host `monotonic_ns` receive times; measure FPS/intervals/drops. Verify manual exposure/focus/white-balance controls if supported. Do not use depth/IR intrinsic parameters.

**Intrinsic validation:** Capture board views at center/edges and several distances/tilts. Compare observed corners with factory RGB K/dist using a **compatible** distortion model. Report per-view, median and p95 reprojection errors with valid sample counts. Factory calibration RMS is unknown; validation reprojection error is a distinct measurement. If needed, separately compare ChArUco-refitted intrinsics without overwriting factory values.

**Stereo extrinsic:** Board must be stationary for each image pair unless synchronization is independently demonstrated. Capture many diverse paired images and keep calibration (~30) and holdout (8-10) in separate directories. Match **identical ChArUco corner IDs** in both cameras and sorted order, express board points in meters. For supported distortion models, use fixed intrinsics and appropriate OpenCV stereo calibration (`CALIB_FIX_INTRINSIC`). Record transform `T_cam1_cam0`, inverse, serial assignment, board specs, fitted/held-out errors, counts, baseline and raw images. Verify proper R, sensible metric baseline, positive depth and independent held-out images. Do not use raw distorted pixels with normalized-ray projectors. Flat-table XY translation/in-plane rotation is an initial paper-board protocol, not evidence of uniform accuracy throughout the 3D workspace; add rigidly supported height/tilt diversity when possible without bending the board.

**Mono/stereo evaluation:** Save one immutable paired RGB recording, run MediaPipe once per camera, and cache its 2D/world outputs. B1-A and B1-B reuse the existing PnP/hand model; transform B1-B from cam1 to cam0 with `T_cam0_cam1`. B2 triangulates in cam0 from the same 2D cache. Report per-method validity and all-method common-valid subsets separately. Nearest pairing is the online result; offline landmark interpolation is a separate result. Sensor `system_time` is not described as exposure time. Without independent GT, report jitter, bone consistency and method disagreement rather than accuracy.

**Rig freeze:** After mounting, changes in lens focus, cropping, image resolution or physical position may invalidate calibration. Preserve raw calibration samples and verification reports. Large raw data, local hardware overrides and serial-specific configuration should be ignored by Git as appropriate. Existing config loader supports `extends: default.yaml`; keep old monocular defaults stable.

## Data contracts (draft; refine without breaking B1)
- `T_A_B`: transforms **B → A** using homogeneous column vectors. `T_cam1_cam0` transforms cam0 points to cam1. OpenCV camera axes x right / y down / z forward. 3D meters, joints radians, host time monotonic nanoseconds.
- `CameraFrame`: `camera_id`, `serial`, `frame_idx`, `image_bgr[H,W,3]`, `host_received_ns`, `sensor_timestamp` plus `timestamp_domain`, active resolution and capture metadata. No assumption that different sensors share a clock.
- `IntrinsicCalibration`: `serial`, color stream profile, `K`, distortion coefficients **and model**, source (`factory`/`refit`), measured validation statistics. Reuse existing `handdata.core.camera.Intrinsics` format where compatible, store additional provenance in metadata/sidecar. The OpenCV adapter accepts `none` and unmodified Brown-Conrady directly; another RealSense model is accepted only when every distortion coefficient is exactly zero, making it a no-distortion projection while preserving the original model in metadata. Nonzero unsupported models fail explicitly.
- `StereoCalibration`: cam0/cam1 serials, active intrinsic references, `T_cam1_cam0`, inverse, board spec, matched-ID sample counts and fit/held-out statistics. The implementation fixes factory intrinsics, rejects per-pair relative-pose outliers, and refuses metric output while physical board verification is false.
- `StereoPoseEstimate`: 21x3 cam0-frame landmarks with per-joint validity/reason, palm/EE pose, quaternion xyzw, pinch/gripper validity, pair gap and quality flags. Raw pixels are undistorted to normalized rays and triangulated with `[I|0]`, `[R|t]`; `K` is not applied again. **Camera coordinate position is not robot-base position** until `T_base_cam0` is obtained.

## Common failure cases
Wrong printed square length silently changes metric scale; mismatched ChArUco IDs corrupt correspondences; unsupported RealSense distortion model corrupts reprojection; moving board with unsynced cameras corrupts stereo extrinsics; resize/crop invalidates K; low corner coverage and blurry views weaken calibration; sensor timestamps across cameras may be incomparable.
