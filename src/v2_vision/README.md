# 휠체어프로젝트 — 자동차 부위 탐지 (YOLOv8)

전동휠체어가 차량에 접근/인식하기 위한 자동차 부위 6종 탐지 모델.

## 전역 클래스 표준 (자연어 언더바, nc=6)
0:car_emblem  1:door_handle  2:fuel_cap  3:license_plate  4:side_mirror  5:tail_light

## 폴더 구조
- raw_datasets/ : 원본 Roboflow zip (건드리지 않음). 공공 6종 + ray_roboflow_v1.zip
- models/ : best_v1_merged / best_v2_ft / best_v3_ray / best_v4_6cls(최신)
- raw_videos/ : 레이 원본 영상 (1, 2, 3)
- my_data/labeled/ray_all/ : 레이 라벨 데이터 (6종, 117장, 후미등 포함) ★현재 데이터
- my_data/frames/ : 프레임 추출용 (현재 비어있음)
- scripts/ : test_video.py, finetune_local.py, remap_ray.sh,
             kaggle_finetune_v3.ipynb, kaggle_finetune_v4.ipynb
- outputs/ : 결과 영상 (v3, v4 테스트)

## 모델 버전 이력
- v1_merged : 5종 개별→통합. 번호판 배경충돌로 억제됨
- v2_ft     : 신형 번호판 fine-tune. 레이영상 plate 13→80프레임
- v3_ray    : 레이 도메인 데이터(5종) 추가 fine-tune. 번호판 실영상 개선
- v4_6cls   : 후미등(tail_light) 추가, 6클래스 새 학습 (2026.07.06) ★최신
              val mAP50: plate .995 / emblem .974 / handle .917 /
              fuel_cap .878 / tail_light .811 / mirror .754

## 학습 방법 (요약)
- Kaggle T4에서 학습. 노트북: scripts/kaggle_finetune_v4.ipynb
- Add Input: 공공 6종 bin + ray_all.bin (한 데이터셋에 다 넣어도 됨)
- 자세한 준비/업로드 순서: kaggle_학습방법.md 참고
- 클래스 5→6 바뀌어 v3 이어받기 불가 → yolov8n부터 새 학습
- 레이는 후미등 유일 출처라 oversampling(RAY_REPEAT)으로 비중 확보

## 다음 할 일
- v4 실영상 추가 검증 (후미등 오인 줄었는지, 미러 성능 회복되는지)
- 필요시: 미러 데이터 보강, 레이 영상 추가 촬영/라벨링, RAY_REPEAT 튜닝

## 핵심 교훈
1. 부분 라벨=배경충돌: 병합 시 라벨 안 된 대상이 배경으로 학습돼 검출이 죽음
   (번호판 v1, 후미등도 공공데이터엔 없어 레이로만 배움)
2. 리허설 fine-tune: 새 데이터+기존 클래스 일부를 섞으면 망각 없이 특정 클래스 보강
3. val≠실전: val 지표 높아도 진짜 판정은 "내 영상에서 몇 프레임 잡히나"
4. 도메인 불일치가 진짜 원인: 해결은 학습량이 아니라 내 영상 도메인 데이터
5. 클래스 추가 시 재매핑: Roboflow 알파벳순 → 전역 표준 순서로 맞춰야 함
   (auto-label은 세그멘테이션 라벨 뱉음 → 박스 변환 필요. v4 노트북이 자동 처리)

## 환경 메모
- 로컬 맥: python3(3.14)는 MPS 안 됨. python3.11 사용 (opencv 등은 3.11에 설치)
- 추론: python3.11 scripts/test_video.py <영상> <모델.pt> <conf>
- 학습은 Kaggle T4 권장 (무료, 환경문제 없음)
- Kaggle zip data.yaml 충돌 → .bin으로 확장자 바꿔 회피

## 다음 단계: 실배치 레이 특화 (계획)
- 실제 테스트/배치할 레이 1대를 확정
- 그 차를 다양한 상황(각도·거리·조명·낮밤)으로 충분히 촬영·라벨링
- 일반화보다 "그 차 확실히 잡기" 목표 → 해당 레이에 의도적으로 특화(overfit 허용)
- 주의: "한 대"여도 다양한 상황이 필요 (같은 장면만 과적합하면 실주행서 약함)
- RAY_REPEAT 높이고, 그 차 데이터 비중 크게 잡아 학습
