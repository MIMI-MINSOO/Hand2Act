import os

# Windows MSMF 웹캠: 하드웨어 변환 초기화 때문에 VideoCapture 열기에 1분 넘게 걸리는 장치가 있음 (끄면 ~1.5초).
# cv2가 MSMF 백엔드를 처음 초기화하기 전에 설정돼야 하므로 패키지 import 시점에 지정.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
