"""Calibrate fixed ``T_cam1_cam0`` from saved common-ID ChArUco pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ..config import add_config_args, load_config
from ..core.board import CharucoTarget
from ..core.camera import Intrinsics
from ..core.stereo import StereoCalibration, calibrate_stereo, match_common_charuco
from .intrinsics import require_verified_board


def load_observations(root, target, min_common, name_prefix=""):
    observations, pair_metadata, skipped = [], [], []
    for pair_dir in sorted(Path(root).glob("pair_*")):
        meta_path = pair_dir / "metadata.json"
        image0, image1 = cv2.imread(str(pair_dir / "cam0.png")), cv2.imread(str(pair_dir / "cam1.png"))
        if image0 is None or image1 is None or not meta_path.exists():
            skipped.append({"pair": pair_dir.name, "reason": "incomplete_pair"})
            continue
        found0, found1 = target.detect_with_ids(image0), target.detect_with_ids(image1)
        if found0 is None or found1 is None:
            skipped.append({"pair": pair_dir.name, "reason": "charuco_not_detected"})
            continue
        matched = match_common_charuco(found0, found1, min_common)
        if matched is None:
            skipped.append({"pair": pair_dir.name, "reason": "too_few_common_ids"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        observations.append((*matched[:3], matched[3], name_prefix + pair_dir.name))
        pair_metadata.append(meta)
    return observations, pair_metadata, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--pairs", default="calib_raw/stereo")
    parser.add_argument("--holdout-pairs",
                        help="independent holdout directory; never used to fit R,t")
    parser.add_argument("--out", default="calib/stereo_extrinsics.json")
    parser.add_argument("--report", default="calib/stereo_validation.json")
    parser.add_argument("--min-common", type=int, default=8)
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--holdout-every", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    try:
        board_spec = require_verified_board(cfg.board)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stereo = cfg.stereo
    intr0_path, intr1_path = cfg.resolve(stereo.intrinsics_cam0), cfg.resolve(stereo.intrinsics_cam1)
    intr0, intr1 = Intrinsics.load(intr0_path), Intrinsics.load(intr1_path)
    if intr0.meta.get("serial") != str(stereo.serial_cam0):
        raise SystemExit("cam0 intrinsic serial does not match configs/stereo.yaml")
    if intr1.meta.get("serial") != str(stereo.serial_cam1):
        raise SystemExit("cam1 intrinsic serial does not match configs/stereo.yaml")
    target = CharucoTarget(cfg.board)
    observations, pair_meta, skipped = load_observations(
        cfg.resolve(args.pairs), target, args.min_common, "calibration/")
    n_calibration = len(observations)
    holdout_indices = None
    holdout_meta, holdout_skipped = [], []
    if args.holdout_pairs:
        holdout_obs, holdout_meta, holdout_skipped = load_observations(
            cfg.resolve(args.holdout_pairs), target, args.min_common, "holdout/")
        if not holdout_obs:
            raise SystemExit("no usable pairs in the independent holdout directory")
        observations.extend(holdout_obs)
        holdout_indices = list(range(n_calibration, len(observations)))
    result = calibrate_stereo(
        observations, intr0, intr1, args.holdout_every, args.min_pairs,
        holdout_indices=holdout_indices)
    metadata = {
        "coordinate_convention": "T_cam1_cam0 maps cam0 points to cam1; OpenCV axes",
        "units": "meters", "board": board_spec,
        "n_calibration_pairs": n_calibration,
        "n_holdout_pairs": len(result["holdout_indices"]),
        "calibration_skipped_pairs": skipped,
        "holdout_skipped_pairs": holdout_skipped,
        "pair_names": [o[4] for o in observations],
        "calibration_pair_timestamp_gaps_ms": [m.get("pair_gap_ms") for m in pair_meta],
        "holdout_pair_timestamp_gaps_ms": [m.get("pair_gap_ms") for m in holdout_meta],
        "holdout_source": (str(cfg.resolve(args.holdout_pairs))
                           if args.holdout_pairs else f"every {args.holdout_every} pairs"),
        "stereo_rms_px": result["stereo_rms_px"],
        "initial_rms_px": result["initial_rms_px"],
        "inlier_indices": result["inlier_indices"],
        "rejected_indices": result["rejected_indices"],
        "holdout_indices": result["holdout_indices"],
        "pose_deviations": result["pose_deviations"],
        "fit_evaluation": result["fit_evaluation"],
        "holdout_evaluation": result["holdout_evaluation"],
        "rig_warning": "Recalibrate after either fixed camera is moved or its active profile changes.",
    }
    calibration = StereoCalibration(
        str(stereo.serial_cam0), str(stereo.serial_cam1), result["R"], result["t"],
        intr0.width, intr0.height, str(intr0_path), str(intr1_path), metadata)
    calibration.save(cfg.resolve(args.out))
    report = {
        "extrinsics": str(cfg.resolve(args.out)),
        "R_cam1_cam0": result["R"].tolist(), "t_cam1_cam0_m": result["t"].tolist(),
        "baseline_m": result["baseline_m"], "inlier_indices": result["inlier_indices"],
        "outlier_indices": result["rejected_indices"],
        "holdout_indices": result["holdout_indices"],
        "calibration_pair_names": [observations[i][4] for i in result["train_indices"]],
        "inlier_pair_names": [observations[i][4] for i in result["inlier_indices"]],
        "outlier_pair_names": [observations[i][4] for i in result["rejected_indices"]],
        "holdout_pair_names": [observations[i][4] for i in result["holdout_indices"]],
        "pose_deviations": result["pose_deviations"],
        "fit": result["fit_evaluation"], "holdout": result["holdout_evaluation"],
        "metric_note": "corner-distance error uses neighbouring ChArUco corners",
    }
    report_path = cfg.resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {cfg.resolve(args.out)}")
    print(f"saved {report_path}")
    print(f"baseline={calibration.baseline_m:.4f}m rms={result['stereo_rms_px']:.3f}px "
          f"inliers={len(result['inlier_indices'])}/{len(result['train_indices'])}")
    held = result["holdout_evaluation"]
    print(f"holdout pairs={held['n_pairs']} median={held['median_px']}px p95={held['p95_px']}px")


if __name__ == "__main__":
    main()
