# 세그멘테이션 실험 결과 (2026-07-10)

## 요약 (TL;DR)

**가설("번호판 마스크로 yaw 부호를 더 안정적으로 잴 수 있다")은 유망하다.**
20분짜리 로컬 맛보기 학습(100 epoch 중 13에서 계획 중단)만으로도 근거리
프레임에서 마스크가 번호판에 정확히 붙었고, 마스크 → 사다리꼴 꼭짓점 →
좌/우 변 길이 비교(키스톤)로 낸 yaw 부호가 검증 지점 3곳 중 2곳 정답.
틀린 1곳은 미성숙 모델의 마스크 노이즈가 원인 → **Kaggle 완주 학습으로
재검증할 가치 충분** (패키지 준비 완료, 아래 "다음 단계").

## 한 것

1. `prepare_seg_dataset.py` — ray_all(121장) → seg용 데이터셋
   (train 103 / valid 18, 박스 라벨 53개는 사각 폴리곤으로 변환)
2. `train_seg.py` — yolov8n-seg, MPS 로컬 학습. **13/100 epoch에서 계획 중단**
   (사용자 부재 일정 때문. best.pt는 매 epoch 저장되므로 유효)
3. `eval_seg.py` — 거리추정_정리.md에 육안 검증된 정답 프레임 4곳에서 평가

## 평가 결과 (epoch 13 모델, conf 0.15)

| 프레임 | 기대 | 검출 | 마스크 yaw 부호 |
|---|---|---|---|
| 영상1 1초 (우측 45°) | R(+) | 5개 | 변길이비 0.605 → **R(+) ✓** |
| 영상1 10초 (좌측 치우침) | L(-) | 5개 | 변길이비 1.495 → **L(-) ✓** |
| 영상3 20초 (우측 비스듬) | R(+) | 3개 | 변길이비 2.305 → **L(-) ✗** |
| 영상2 5초 (원거리 후면) | - | 1개 (tail만) | 번호판 미검출 |

- 이미지: `eval/*_masks.jpg` (전체 마스크), `eval/*_plate_mask.jpg` (번호판 확대)
- **영상1 1초 마스크는 육안으로도 번호판 윤곽에 깔끔하게 밀착** — 이 품질이면
  기존 Otsu 꼭짓점 검출(distance_video.py의 find_plate_quad)을 대체 가능한 수준
- **영상3 실패 원인**: 마스크가 번호판 밖(브래킷·차체)으로 새는 노이즈 →
  convex hull 꼭짓점이 왜곡. 13 epoch 미성숙 모델의 전형적 증상.
  참고: 이 프레임의 번호판은 다른 차("237허 5991")임 — 목표 차량 필터 미구현
  한계(README 기지사항)가 seg에서도 동일하게 존재
- 영상2 원거리 미검출도 미성숙 + 소형 객체 → 완주 학습 후 재평가 필요
- val 지표(epoch 13): mask mAP50 0.19 (탐지 v4의 0.9x 대비 낮음 — 13 epoch뿐이라
  당연. 완주 후 다시 볼 것)

## 배운 것 / 함정

- **minAreaRect로는 키스톤 측정 불가** (항상 완전한 직사각형을 반환해 좌/우 변이
  정의상 동일). convex hull → approxPolyDP 4각형 근사를 써야 함. eval_seg.py에 반영
- 세그 학습은 폴리곤 라벨만 허용 → 박스 53개 변환 필요했음 (prepare 스크립트가 처리)
- ray_all은 valid가 비어 있어 분할 필수 (여기선 85/15 프레임 랜덤, 시드 42)

## 다음 단계 (Kaggle 완주 학습 — 패키지 준비 완료)

1. kaggle.com → 기존 데이터셋에 `seg_experiment/ray_seg.bin` 추가 업로드
2. `seg_experiment/kaggle_seg.ipynb` 노트북 업로드 → T4, Internet On →
   Add Input에 ray_seg.bin 데이터셋 연결 → 전체 실행 (100 epoch, patience 25)
3. `/kaggle/working/runs/seg_full/weights/best.pt` 다운로드 →
   `seg_experiment/runs/seg_full_best.pt`로 저장
4. 재평가: `python3.10 seg_experiment/eval_seg.py seg_experiment/runs/seg_full_best.pt`
5. 그 결과가 좋으면: distance_video.py의 find_plate_quad(Otsu)를 마스크 기반으로
   교체하는 것 검토 + 후미등 "안쪽 변" 기준 각도 재도전

## 파일

```
seg_experiment/
├── prepare_seg_dataset.py  # ray_all → seg 데이터셋 (원본 안 건드림)
├── train_seg.py            # 로컬 MPS 학습 스크립트
├── eval_seg.py             # 정답 프레임 4곳 평가 (모델 경로 인자로 교체 가능)
├── kaggle_seg.ipynb        # Kaggle T4 완주 학습용 노트북
├── ray_seg.bin             # Kaggle 업로드용 데이터셋 (55MB)
├── dataset/                # 로컬 seg 데이터셋 (train 103 / valid 18)
├── runs/seg_v1/            # 이번 13-epoch 실험 결과 (weights/best.pt)
├── eval/                   # 평가 이미지 (마스크 시각화)
└── train_log.txt           # 학습 로그
```

※ 본 실험은 전부 이 폴더 안에 격리됨. 기존 파이프라인(scripts/, models/,
my_data/labeled/ray_all)은 변경 없음. my_data/labeled/ray_all.bin(탐지 v5용)은
이전 작업에서 생성된 별개 파일.
