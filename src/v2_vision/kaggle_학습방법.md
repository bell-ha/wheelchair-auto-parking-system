# Kaggle 재학습 가이드 (다음에 데이터 추가해서 학습할 때)

## 큰 그림
공공 6종 데이터(bin) + 내 레이 데이터(ray_all) + 포스터 데이터(ray_poster)를 합쳐 Kaggle T4에서 학습.
노트북: scripts/kaggle_finetune_v4.ipynb (6클래스 기준)
- ray_all.bin: 실차 촬영 121장 (RAY_REPEAT로 비중 조절)
- ray_poster.bin: 실물크기 포스터 촬영 60장, 2026-07 추가 (POSTER_REPEAT로 독립 조절.
  실전 테스트 환경이 포스터라 기본 12로 강하게 시작)

## 1. 로컬에서 레이 데이터 .bin 만들기
새로 라벨링/재매핑 끝난 ray_all 폴더를 .bin으로:
    cd ~/Desktop/휠체어프로젝트/my_data/labeled
    cd ray_all && zip -r -q ../ray_all.zip . && cd ..
    cp ray_all.zip ray_all.bin
(공공 6종 bin은 이미 Kaggle에 있으면 재활용. 없으면 raw_datasets에서 다시 .bin 생성)

## 2. Kaggle 업로드
- ray_all.bin (+ 새로 생겼으면 ray_poster.bin)을 데이터셋에 추가 (기존 데이터셋 있으면 거기에)
- .bin이라 data.yaml 자동압축해제 충돌 안 남

## 3. 노트북 실행
- Settings → GPU T4, Internet On
- Add Input: 공공 6종 bin 있는 데이터셋 + ray_all.bin
- scripts/kaggle_finetune_v4.ipynb 올리고 0번 셀부터 실행
- 확인: 0번(경로 3개 잘 찾나) / 3번(매핑·asis) / 4번(클래스 분포, tail_light 수, 레이 비중)

## 4. 학습 후
- best.pt 다운로드 → models/best_v5_xxx.pt (버전 올려서 저장)
- 검증: python3.11 scripts/test_video.py raw_videos/레이영상_3.mov <모델> 0.15
- README 성능 이력에 새 버전 추가

## 튜닝 손잡이 (노트북 1번 셀)
- RAY_REPEAT : 레이(실차)/후미등 비중. 약하면 올리고, 기존 클래스 무너지면 낮춤
- POSTER_REPEAT : 포스터 비중 (실차와 독립). 포스터 검출 약하면 올림, 실차·공공 무너지면 낮춤
- REHEARSAL_PER_CLASS : 공공데이터 양. 기존 클래스 흔들리면 올림
- EPOCHS : 새 클래스 추가 시 넉넉히(40), 이어받기면 적게(20)

## 함정 메모 (겪은 것)
- 클래스 추가 시 Roboflow는 알파벳순 → 전역 표준 순서로 재매핑 필요
- Roboflow auto-label(SAM3)은 세그멘테이션(폴리곤) 라벨 뱉음 → 박스 변환 필요
  (v4 노트북 3번 셀이 자동 변환함)
- v4 노트북은 여러줄 명령 복붙 시 zsh에서 # 주석 깨질 수 있음 → 한 줄씩 실행
