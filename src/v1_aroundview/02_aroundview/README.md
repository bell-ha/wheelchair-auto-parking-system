# 02_aroundview — 탑뷰 합성

여기의 캘리브레이션은 어라운드뷰 전용이다. `01_calibration/` 의 것과 별개다.

| 파일 | 하는 일 |
|---|---|
| `calibration_left.py` · `calibration_rear.py` | 일반 카메라 캘리브레이션 |
| `aroundview_left.py` · `aroundview_rear.py` | 캘리브레이션 결과에 점을 찍어 어라운드뷰 생성 |
| `pickpoint.py` | `point_*.py` 가 쓸 점만 골라 저장 |
| `point_left.py` · `point_rear.py` | 저장한 점으로 어라운드뷰 생성 |
| `main.py` | Left·Rear 를 합쳐 하나의 캔버스로 |

`Data/` — 캘리브레이션 `.npz` 와 중간에 캡처한 데이터.
