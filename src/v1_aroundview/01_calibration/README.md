# 01_calibration — 카메라 왜곡계수

체커보드를 촬영해 왜곡계수를 뽑고, 그 값을 Left·Rear 코드에 적용한다.

| 파일 | 하는 일 |
|---|---|
| `calibration.py` | 캘리브레이션을 돌려 왜곡계수를 반환 |
| `angle_left.py` · `angle_rear.py` | 왜곡계수를 적용해 ArUco 마커 각도를 반환 |
