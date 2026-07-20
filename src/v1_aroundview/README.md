# V1 — 차량 어라운드뷰 기반 (개발 중단)

차량에 장착한 어라운드뷰 카메라 2대(Left, Rear)로 휠체어에 부착된 ArUco 마커를
검출해 휠체어 위치를 추정하고, A* 경로계획으로 제어 신호(F/B/L/R)를 만들어
휠체어를 차량 옆 주차 위치까지 유도하는 파이프라인.

> **중단 사유**: 어안(fisheye) 카메라 캘리브레이션의 누적 오차로 맵 좌표가 튀는
> 문제를 07_map_calibration에서 끝까지 잡지 못함. 카메라를 차량이 아닌
> **휠체어에 탑재**하는 방식([V2](../v2_vision/))으로 전환.

## 환경 설정

```bash
source ../venv/bin/activate   # 가상환경 (repo에는 미포함)
pip install -r requirements.txt
```

## 폴더 구성 (파이프라인 순서)

| 폴더 | 내용 |
|---|---|
| `01_calibration/` | 체커보드 촬영 → 카메라 왜곡계수 추출 |
| `02_aroundview/` | Left/Rear 영상 정합 → 어라운드뷰(탑뷰) 합성 |
| `03_localization/` | ArUco 마커 검출로 휠체어 위치/방향 추정, 맵 생성 |
| `04_planning/` | A* 기반 경로계획 |
| `05_hardware/` | 구동부: TRACER 섀시 CAN 통신 제어 절차 |
| `06_angle_tunning/` | 각도 보정(트랙바 UI로 파라미터 튜닝) |
| `07_map_calibration/` | 최종 맵 캘리브레이션 — 오차 가중 보정, outlier 제거 시도 |
| `08_parking/` | 통합 실행: 위치추정 + 경로계획 + 제어 (수동조종 포함) |
| `command/` | 03/06 계열 통합 실험 버전 |
| `utils/` | 카메라 탐색, 녹화, ArUco 검출 등 보조 도구 |

## 기술 노트

캘리브레이션 수치, 카메라 설치 위치/각도, 좌표계 정의 등 상세 기록은
[NOTES.md](NOTES.md) 참고.
