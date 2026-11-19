"""WeGo NERO 양팔 장면을 홈 자세(키프레임 'home')로 시작하는 MuJoCo 뷰어.

    python scripts/view_nero.py                 # 홈 자세로 시작 (물리 켜짐, 액추에이터가 자세 유지)
    python scripts/view_nero.py --key none      # 관절 0 (팔 똑바로 선 자세)

뷰어에서 Backspace(리셋)는 관절 0 자세로 돌아감 → 홈으로 다시 가려면 오른쪽 패널 Simulation > Key 에서 'home' 로드.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mjcf", default=str(ROOT / "assets/nero/nero_dual.xml"))
p.add_argument("--key", default="home")
a = p.parse_args()

model = mujoco.MjModel.from_xml_path(a.mjcf)
data = mujoco.MjData(model)
if a.key != "none":
    mujoco.mj_resetDataKeyframe(model, data, model.key(a.key).id)
mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)
