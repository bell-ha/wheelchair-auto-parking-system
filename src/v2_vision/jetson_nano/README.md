# jetson_nano — 젯슨 나노 배포용 폴더

카메라 실시간 추론 + 하드웨어(초음파/서보) 통합 코드 모음.
학습 코드·데이터셋은 없음 (추론 전용).

```
jetson_nano/
├── main.py                        # 통합 실행: 카메라(YOLO) + 초음파/서보(ESP32) 한 프로세스
├── camera/
│   ├── live_camera.py             # 카메라 단독 실행/부하테스트용 (USB 웹캠, C920)
│   └── best_v5_poster.pt          # 최신 모델
├── ultrasound/
│   ├── sample.ino                 # ESP32 펌웨어: 초음파 5개 + 서보 2축(팬/틸트), JSON 프로토콜
│   └── test_sample.py             # 수동 조작용 curses 툴 (A/D/W/S로 서보, 텔레메트리 화면표시)
├── guidance/
│   ├── teach.py                   # 목표 상태 저장 툴 (카메라+초음파 보면서 지도 클릭, s로 저장)
│   ├── guide.py                   # 사람-가이드 정렬: TURN LEFT 등 지시 표시 (모터 출력 없음)
│   └── common.py                  # guidance 공용 (카메라/시리얼 파서/앵커 추출/좌표 추정)
├── unused/                        # 지금 안 쓰는 코드 (이유는 unused/README.md 참고)
│   └── joystick/                  # 휠체어 주행 방향 제어 — 받아줄 펌웨어가 아직 없음
├── requirements.txt
└── README.md
```

`main.py`가 카메라(비전)와 초음파/서보(ESP32)를 한 프로세스에서 같이 돌리는
통합 실행 파일 — 검출 결과와 초음파 텔레메트리를 매 프레임 터미널에 같이 출력함.
`camera/live_camera.py`와 `ultrasound/test_sample.py`는 각 부품을 단독으로 빠르게
점검하고 싶을 때 쓰는 개별 진입점으로 계속 남겨둠. 카메라/초음파 값을 엮어서 서보를
자동으로 움직이는 판단 로직은 아직 없음 (지금은 순수 통합만).

## 젯슨 접속 (SSH)

맥과 젯슨이 같은 네트워크(같은 공유기)에 있어야 함. 계정 `jetson`, 호스트명 `nano`.

```bash
# 맥에서 — 호스트명으로 접속 (mDNS)
ssh jetson@nano.local

# 안 되면 IP로 (192.168.0.254)
ssh jetson@<JETSON_IP>
```

- IP 확인: 젯슨에서 `hostname -I`, 또는 공유기 관리 페이지 접속기기 목록.
- IP가 자꾸 바뀌면 공유기에서 DHCP 고정(예약) 해두기.
- 비밀번호 생략하려면 맥에서 한 번만: `ssh-copy-id jetson@nano.local`

## GitHub에서 받기

리포 전체는 수 GB라 이 폴더만 sparse checkout으로 받는다.

```bash
cd ~/GitHub
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/bell-ha/wheelchair-auto-parking-system.git
cd wheelchair-auto-parking-system
git sparse-checkout init --cone
git sparse-checkout set src/v2_vision/jetson_nano
git checkout main
cd src/v2_vision/jetson_nano
```

## Pull / Push

```bash
# 받기 (맥 등 다른 곳에서 push된 내용)
cd ~/GitHub/wheelchair-auto-parking-system
git pull

# 젯슨에서 수정한 거 올리기 (인증은 이미 설정돼있음)
git add -A
git commit -m "젯슨: 테스트 중 수정"
git push
```

작업 시작 전 `git pull`, 끝나면 바로 `push`. 같은 파일을 양쪽에서 동시에 고치면 충돌 남.

## 설치

```bash
pip3 install -r requirements.txt

# 주의: ultralytics가 opencv-python을 의존성으로 깔기 때문에 위 명령 직후
# GStreamer 미지원 pip opencv가 다시 설치돼있음. 반드시 지워서 JetPack
# 기본 OpenCV(dist-packages, GStreamer 지원)가 쓰이게 할 것 —
# 안 지우면 USB 웹캠(C920)에서 카메라 열기가 무한 대기함.
pip3 uninstall -y opencv-python
```

`ultrasound/sample.ino`는 아두이노 IDE로 ESP32 보드에 업로드 (젯슨 아님).
Jetson과 ESP32는 USB 케이블로 직결, 포트는 `/dev/ttyUSB0`, 115200bps.

## 실행 (통합 — main.py)

```bash
cd ~/GitHub/wheelchair-auto-parking-system/src/v2_vision/jetson_nano
python3 main.py camera/best_v5_poster.pt --cam-width 1280 --cam-height 720
```

카메라(YOLO 검출)와 ESP32(초음파 텔레메트리 수신)를 한 프로세스에서 같이 돌림.
터미널에 `test_sample.py`와 같은 스타일의 고정 화면(curses)이 뜨고, 검출 개수·FPS·
초음파 5개 값·서보 현재각이 실시간으로 갱신됨. `q`로 종료.

- HDMI 모니터에는 검출 박스가 그려진 카메라 창이 같이 뜸 (`--no-show`로 끄면 터미널 화면만)
- ESP32가 안 붙어있으면 `--no-servo`로 카메라만 테스트 가능
- 그 외 옵션은 `python3 main.py --help`

시작 직후 "모델 워밍업 중..."이 30초~1분 정도 뜨는 건 정상 (젯슨 첫 추론 특성).
워밍업이 끝나야 터미널 화면으로 전환됨.

`--cam-width`/`--cam-height`는 16:9 조합(1280x720, 1920x1080 등)으로 맞출 것 — C920은
4:3 해상도(640x480, 800x600, 1280x960)를 요청하면 카메라 자체가 좌우를 크롭해서
화각이 좁아짐. 1920x1080은 화면은 더 선명하지만 이 보드(RAM 4GB) 기준 메모리 여유가
빠듯해서(스왑 발생 확인됨) 1280x720 권장.

## 카메라만 단독 실행 (live_camera.py)

부하 테스트나 카메라 단독 점검용. `main.py`와 옵션 동일.

```bash
python3 camera/live_camera.py camera/best_v5_poster.pt --cam-width 1280 --cam-height 720
```

SSH/VSCode 원격 터미널로 실행해도 젯슨 본체 HDMI 모니터(:0)에 알아서 창이 뜸.
시작하고 "모델 워밍업 중..."이 최대 1분 정도 뜨는 건 정상 (멈춘 게 아님).

부하 확인은 다른 터미널에서 `sudo tegrastats` 병행 권장 (GPU 사용률·온도·스로틀링 확인).
자세한 결과는 [../jetson_부하테스트_보고서.md](../jetson_부하테스트_보고서.md) 참고.

## 초음파+서보 수동 조작/점검 (test_sample.py)

A/D = 서보 X, W/S = 서보 Y, Q = 종료. 초음파 5개(정면/좌전/좌후/우전/우후) 텔레메트리도
화면에 같이 표시됨. ESP32 연결 상태를 빠르게 점검하고 싶을 때 이걸로.

```bash
python3 ultrasound/test_sample.py
```

포트는 파일 상단 `SERIAL_PORT`(`/dev/ttyUSB0`) 수정.

## 사람-가이드 정렬 (guidance/)

teach & repeat 방식: `teach.py`로 목표 상태(사이드미러 위치·크기 + 측면 초음파 쌍 +
지도에 클릭한 목표 좌표)를 JSON으로 저장해두고, `guide.py`가 현재 상태와 비교해
"TURN LEFT / MOVE FORWARD / ALIGNED - STOP" 지시를 화면에 표시함 (모터 출력 없음).
①VISION 단계(사이드미러 정렬) → ②SONAR 단계(측면 초음파 앞/뒤 차이로 평행 정렬) 순서.

```bash
python3 guidance/teach.py    # 지도 패널 클릭으로 목표 지정, s 저장, q 종료
python3 guidance/guide.py    # r로 VISION 단계 재시작, q 종료
```

- 카메라·시리얼을 main.py/test_sample.py와 공유하므로 **동시 실행 불가**
- 초음파 기준 거리(SONAR_BASELINE=0.315m)는 `sample.ino`의 `SIDE_SENSOR_SPACING_CM=31.5`와
  맞춰둔 값 — 센서 간격을 바꾸면 양쪽 다 수정할 것



## 젯슨나노 종료
```bash
sudo shutdown -h now
```