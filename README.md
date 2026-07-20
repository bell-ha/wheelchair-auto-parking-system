# 휠체어 자동주차 시스템 (Wheelchair Auto-Parking System)

전동휠체어가 차량(기아 레이)을 인식하고 스스로 승하차 위치까지 접근·도킹하는
시스템. 교통약자의 승하차 과정을 자동화하는 것이 목표다.
단국대학교 배리어프리 ICT기술 연구센터(ITRC) 과제로 진행 중.

## 프로젝트 히스토리 — 왜 두 버전인가

| | V1 (중단) | V2 (진행 중) |
|---|---|---|
| 카메라 위치 | **차량**에 어라운드뷰 카메라 2대 (Left/Rear) | **휠체어**에 카메라 탑재 |
| 휠체어 인식 | 휠체어에 부착한 ArUco 마커 검출 | 딥러닝으로 차량 부위 직접 탐지 (YOLOv8) |
| 위치 추정 | 어라운드뷰 탑뷰 맵 위 좌표 | 번호판 실측 규격 기반 단안 거리·각도 추정 |
| 경로/제어 | A* 경로계획 → CAN 통신 (TRACER 섀시) | 팬 모터 트래킹 + 초음파 결합 (설계 중) |
| 코드 | [`src/v1_aroundview/`](src/v1_aroundview/) | [`src/v2_vision/`](src/v2_vision/) |

**전환 이유**: V1은 카메라 캘리브레이션 → 어라운드뷰 합성 → 위치추정 → A*
경로계획 → CAN 제어까지 파이프라인을 완성했지만, 어안(fisheye) 카메라
캘리브레이션의 누적 오차로 맵 좌표가 튀는 문제를 최종 단계(map calibration)에서
해결하지 못했다. 오차의 근원이 "차량에 고정된 카메라로 넓은 영역을 왜곡 보정해서
보는 구조" 자체에 있다고 판단, **카메라를 휠체어에 싣고 차량을 직접 인식하는
구조(V2)** 로 방향을 전환했다.

## V2 현재 성과

**차량 부위 6종 탐지 모델 (YOLOv8, v4 기준)** — 자체 촬영·라벨링한 레이 데이터로
공공 데이터셋의 도메인 불일치를 해결. val mAP50:

| license_plate | car_emblem | door_handle | fuel_cap | tail_light | side_mirror |
|---|---|---|---|---|---|
| .995 | .974 | .917 | .878 | .811 | .754 |

**번호판 기반 단안 거리·자세 추정** — 번호판 실측 규격(335×155mm)과 핀홀 모델로
거리(m), yaw(차량이 비스듬한 정도), bearing(시야 내 방향)을 실시간 추정.
실영상 육안 검증에서 거리·부호 모두 장면과 일치 (예: 실측 1.2m 정면 → 추정
1.20m, yaw ≈ 0).

상세 기록: [모델 학습 이력](src/v2_vision/README.md) ·
[거리추정 정리](src/v2_vision/거리추정_정리.md) ·
[세그멘테이션 실험](src/v2_vision/seg_experiment/RESULTS.md)

## 저장소 구조

```
├── src/
│   ├── v1_aroundview/     # V1: 어라운드뷰 + ArUco 파이프라인 (01~08 단계별)
│   └── v2_vision/         # V2: YOLOv8 차량 부위 탐지 + 거리추정 (진행 중)
│       ├── models/        #   학습된 모델 v1~v4 (.pt)
│       ├── scripts/       #   추론·거리추정·파인튜닝 스크립트
│       ├── seg_experiment/#   번호판 세그멘테이션 실험
│       ├── my_data/       #   자체 촬영 레이 라벨 데이터
│       └── outputs/       #   탐지/거리추정 결과 영상
├── files/                 # 발표자료, 참고문헌, 3D 모델(전시용), 포스터 이미지
└── README.md
```

## 기술 스택

- **Vision**: OpenCV (fisheye calibration, ArUco, 원근 변환), YOLOv8 (ultralytics)
- **학습**: Kaggle T4 GPU, Roboflow 라벨링, 도메인 특화 fine-tuning / oversampling
- **경로계획**: A* + 재계획(replanning)
- **하드웨어**: AgileX TRACER 섀시 (CAN 통신, python-can), 어안 카메라 2대(V1) → 휠체어 탑재 카메라(V2)

## 대용량 파일 안내

100MB를 초과하는 결과 영상·데이터셋 원본 18개(약 3.8GB)는 GitHub 용량 제한으로
저장소에서 제외했다(`.gitignore`에 목록 명시). 학습 데이터 원본은 Roboflow 공공
데이터셋 + 자체 촬영 영상이며, 재현에 필요한 스크립트와 학습 노트북은 모두
포함되어 있다.
