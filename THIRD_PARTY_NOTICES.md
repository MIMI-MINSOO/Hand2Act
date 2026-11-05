# Third-party notices

다음 파일은 [advaithdasu/hand-control-robot](https://github.com/advaithdasu/hand-control-robot) (MIT License)의 코드를 가져와 수정했습니다.

- `handdata/core/filters.py` — One Euro filter (실시간 부분)
- `handdata/robot/ik.py` — damped least squares IK
- `handdata/robot/kinematics.py` — MuJoCo FK/Jacobian
- `assets/demo_arm.xml` — 6-DOF 데모 arm MJCF (원본 `assets/scene.xml`)

```
MIT License

Copyright (c) 2026 Advaith Dasu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MediaPipe HandLandmarker 모델(`assets/hand_landmarker.task`, `scripts/download_model.sh`로 받음)은 Google의 Apache-2.0 라이선스를 따릅니다.
