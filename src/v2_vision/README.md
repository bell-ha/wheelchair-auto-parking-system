# 휠체어프로젝트 — 자동차 부위 탐지 (YOLOv8)

전동휠체어가 차량에 접근/인식하기 위한 자동차 부위 6종 탐지 모델.

## 전역 클래스 표준 (자연어 언더바, nc=6)
0:car_emblem  1:door_handle  2:fuel_cap  3:license_plate  4:side_mirror  5:tail_light

## 폴더 구조 (2026-07-23 정리)
- raw_datasets/ : 원본 Roboflow zip (건드리지 않음). 공공 6종 + ray_roboflow_v1.zip
- models/ : best_v1_merged / best_v2_ft / best_v3_ray / best_v4_6cls(최신)
- raw_videos/ : 원본 영상 — 레이영상 1·2·3(실차) + 포스터레이.MOV(실물크기 포스터, 포스터 라벨의 원본)
- my_data/labeled/ray_all/ : 실차 라벨 데이터 (6종, 121장) ★
- my_data/labeled/ray_poster/ : 포스터 라벨 데이터 (6종, 60장, 2026-07 추가) ★
- scripts/ : test_video.py, distance_video.py, live_camera.py, finetune_local.py,
             kaggle_finetune_v3.ipynb, kaggle_finetune_v4.ipynb (remap_ray.sh는 5클래스 시절 유물 — 사용 금지)
- outputs/ : 결과 영상 (v4 검출 3개 + distance HUD 3개만 보존. v3·tail 실험본은 삭제 — 스크립트로 재생성 가능)
- seg_experiment/ : 세그멘테이션 실험 기록 (RESULTS.md + 스크립트만 보존, 대용량 삭제)
- kaggle_upload/ : Kaggle 업로드용 공공 6종 .bin 로컬 백업 (캐글 데이터셋 유실 대비.
  raw_datasets zip과 내용 동일, 이름만 노트북 SOURCES 기준)
- my_data/labeled/ray_all.bin·ray_poster.bin : 라벨 폴더에서 재생성되는 업로드용 압축본 (git 미추적)

## 모델 버전 이력
- v1_merged : 5종 개별→통합. 번호판 배경충돌로 억제됨
- v2_ft     : 신형 번호판 fine-tune. 레이영상 plate 13→80프레임
- v3_ray    : 레이 도메인 데이터(5종) 추가 fine-tune. 번호판 실영상 개선
- v4_6cls   : 후미등(tail_light) 추가, 6클래스 새 학습 (2026.07.06)
              val mAP50: plate .995 / emblem .974 / handle .917 /
              fuel_cap .878 / tail_light .811 / mirror .754
- v5_poster : 포스터(실물크기 목업) 60장 추가, POSTER_REPEAT=12 독립 소스 (2026.07.24) ★최신
              포스터 환경 대폭 개선 (plate 검출 프레임 6배, conf 0.26→0.89)
              ※ 실차 성능 재검증 미완 — 자세한 내용은 포스터_위치추정_정리.md

## 학습 방법 (요약)
- Kaggle T4에서 학습. 노트북: scripts/kaggle_finetune_v4.ipynb
- Add Input: 공공 6종 bin + ray_all.bin (한 데이터셋에 다 넣어도 됨)
- 자세한 준비/업로드 순서: kaggle_학습방법.md 참고
- 클래스 5→6 바뀌어 v3 이어받기 불가 → yolov8n부터 새 학습
- 레이는 후미등 유일 출처라 oversampling(RAY_REPEAT)으로 비중 확보

## 다음 할 일 (2026-07-24 갱신 — 상세는 포스터_위치추정_정리.md '남은 일')
- 포스터 줄자 실측 4개 → 위치추정 랜드마크 좌표 확정
- f 캘리브레이션 (--calib), v5 실차 성능 확인 (레이영상 1~3 재실행)
- 도킹 제어 개발 (진입점 계산 → 2단계 접근 → IBVS 정밀 정렬)

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
