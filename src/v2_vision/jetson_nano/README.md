# jetson_nano — 젯슨 나노 배포용 폴더

이 폴더만 통째로 젯슨 나노에 복사하면 카메라 실시간 추론 테스트와
하드웨어(초음파/수동조종) 코드를 바로 쓸 수 있게 구성한 배포 묶음.
학습 관련 코드·데이터는 포함하지 않음 (추론 전용).

## 구성

```
jetson_nano/
├── models/
│   └── best_v5_poster.pt    # v5: 장형 번호판 포함 포스터 목업용 최신 모델
├── scripts/
│   └── live_camera.py       # 카메라 실시간 YOLO 추론 + FPS 부하 테스트
│                            #   (USB 웹캠 C920 계열 select() 타임아웃 대응:
│                            #    MJPG 강제 + 워밍업 + cv2.error 방어 포함)
├── hardware/
│   └── ultrasound/
│       ├── ultrasound.ino   # 아두이노: 초음파 3개로 벽 각도/거리 계산
│       └── keyboard.py      # 키보드 방향키 → MQTT 수동 조종 명령 전송
└── requirements.txt
```

## 젯슨에서 다운로드 (GitHub)

리포 전체는 데이터셋·영상 때문에 수 GB라 통째로 clone하지 말고,
이 폴더만 sparse checkout으로 받는다.

```bash
# 젯슨 터미널에서 — GitHub 폴더 안에 받는다
# (clone --sparse 옵션은 구버전 git에서 버그가 있어 단계를 나눠 실행)
cd ~/GitHub
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/bell-ha/wheelchair-auto-parking-system.git
cd wheelchair-auto-parking-system
git sparse-checkout init --cone
git sparse-checkout set src/v2_vision/jetson_nano
git checkout main
cd src/v2_vision/jetson_nano
```

이후 맥에서 수정해 push한 내용을 다시 받을 때는:

```bash
cd ~/GitHub/wheelchair-auto-parking-system
git pull
```

## 젯슨에서 수정 → push

sparse checkout이어도 일반 git 저장소라 젯슨에서 테스트하며 고친 것도
바로 commit·push 가능.

처음 한 번만 설정:

```bash
git config --global user.name "LeeBellHa"
git config --global user.email "studioseiha@gmail.com"
git config --global credential.helper store   # 토큰 한 번 입력 후 저장
```

push할 때 비밀번호 자리에는 GitHub **Personal Access Token(PAT)** 입력
(GitHub → Settings → Developer settings → Personal access tokens →
Tokens (classic) → repo 권한 체크. 계정 비밀번호로는 push 안 됨).

```bash
git add -A
git commit -m "젯슨: 테스트 중 수정"
git push
```

**충돌 방지 규칙**: 맥이든 젯슨이든 작업 시작 전에 `git pull` 먼저,
끝나면 바로 push. 양쪽에서 같은 파일을 동시에 고치면 충돌 남.

**git 버전 주의**: 젯슨 나노(JetPack, Ubuntu 18.04)의 기본 git 2.17은
`sparse-checkout` 명령이 없음(2.25+ 필요). 먼저 git을 업데이트할 것:

```bash
sudo add-apt-repository ppa:git-core/ppa
sudo apt update && sudo apt install -y git
```

## 젯슨에서 설치

```bash
pip3 install -r requirements.txt
```

- **OpenCV 주의**: CSI 카메라(`--csi`)를 쓰려면 GStreamer 지원 OpenCV가 필요.
  JetPack 기본 OpenCV를 쓰면 됨 (pip로 opencv-python을 덮어쓰지 말 것).
  USB 웹캠만 쓸 거면 상관없음.
- `ultrasound.ino`는 아두이노 IDE로 아두이노 보드에 업로드 (젯슨 아님).

## 실행

```bash
cd ~/GitHub/wheelchair-auto-parking-system/src/v2_vision/jetson_nano

# USB 웹캠
python3 scripts/live_camera.py models/best_v5_poster.pt

# CSI 카메라
python3 scripts/live_camera.py models/best_v5_poster.pt --csi

# SSH(화면 없음) — 콘솔에 FPS만 출력
python3 scripts/live_camera.py models/best_v5_poster.pt --no-show
```

부하 확인은 다른 터미널에서 `sudo tegrastats` 병행 권장
(GPU 사용률·온도·스로틀링 확인). 자세한 결과는
[../jetson_부하테스트_보고서.md](../jetson_부하테스트_보고서.md) 참고.

## 수동 조종 (keyboard.py)

방향키 → MQTT(`wheelchair/command`)로 각도(0/90/180/270) 전송, `s`로 정지.
브로커는 `broker.emqx.io` 공용 브로커 사용 중 — 필요 시 파일 상단 설정 수정.

```bash
python3 hardware/ultrasound/keyboard.py
```
