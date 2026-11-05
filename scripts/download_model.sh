#!/usr/bin/env bash
# MediaPipe HandLandmarker 모델 다운로드 (Apache-2.0, ~7.8MB)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets
curl -fL -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
echo "saved: assets/hand_landmarker.task"
