# jetson_nano — 젯슨 나노 배포용 폴더

카메라 실시간 추론 + 하드웨어(초음파/수동조종) 코드 모음.
학습 코드·데이터셋은 없음 (추론 전용).

```
jetson_nano/
├── models/
│   └── best_v5_poster.pt          # 최신 모델
├── scripts/
│   └── live_camera.py             # 카메라 실시간 YOLO 추론 (USB 웹캠, C920)
├── hardware/
│   ├── serial_link.py             # 공용 시리얼 연결 헬퍼 (초음파/조이스틱 공용, 연결 하나 공유 가능)
│   ├── joystick/
│   │   └── joystick_controller.py # 방향 명령(각도/stop) 송신, keyboard.py가 사용
│   └── ultrasound/
│       ├── ultrasound.ino         # 아두이노/ESP32: 초음파 8개+IMU 골격 (WIP, 아직 더미값)
│       ├── ultrasound_reader.py   # 시리얼 JSON 수신 확인용 스크립트 (WIP, 콘솔 출력만 함)
│       └── keyboard.py            # 방향키 → JoystickController → ESP32 수동 조종
└── requirements.txt
```

카메라(비전) · 초음파+IMU · 조이스틱 송신은 각자 독립적으로 동작하는
부품 상태이고, 셋을 엮어서 자동 주행 명령을 만드는 오케스트레이션
코드(`main.py`)와 경로 계획 로직은 아직 없음 — 설계 방향 정해지는 대로 추가 예정.

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

`ultrasound.ino`는 아두이노 IDE로 아두이노/ESP32 보드에 업로드 (젯슨 아님).
지금은 실제 센서/IMU 없이 더미값을 JSON으로 내보내는 골격만 있는 상태
(하드웨어 도착하면 `readUltrasonicDummy`/`readYawDummy`를 실측 코드로 교체).

## 실행

```bash
cd ~/GitHub/wheelchair-auto-parking-system/src/v2_vision/jetson_nano
python3 scripts/live_camera.py models/best_v5_poster.pt --cam-width 640 --cam-height 480
```

SSH/VSCode 원격 터미널로 실행해도 젯슨 본체 HDMI 모니터(:0)에 알아서 창이 뜸.
시작하고 "모델 워밍업 중..."이 최대 1분 정도 뜨는 건 정상 (멈춘 게 아님).

다른 옵션(콘솔 모드, CSI 카메라 등)은 `python3 scripts/live_camera.py --help`.

부하 확인은 다른 터미널에서 `sudo tegrastats` 병행 권장 (GPU 사용률·온도·스로틀링 확인).
자세한 결과는 [../jetson_부하테스트_보고서.md](../jetson_부하테스트_보고서.md) 참고.

## 수동 조종 (keyboard.py)

방향키 → ESP32에 USB 시리얼로 각도(0/90/180/270) 전송, `s`로 정지.
Jetson과 ESP32는 USB 케이블로 직결, 포트는 `/dev/ttyUSB0`, 115200bps
(필요 시 파일 상단 `SERIAL_PORT`/`BAUD_RATE` 수정).

```bash
python3 hardware/ultrasound/keyboard.py
```

## 초음파 수신 확인 (ultrasound_reader.py)

`ultrasound.ino`가 보내는 JSON 한 줄(`{"us":[...], "side_angle":.., "side_dist":.., "yaw":.., ...}`)을
그대로 읽어서 콘솔에 찍어보는 연결 확인용 스크립트. 아직 값을 다른
코드가 가져다 쓸 수 있는 형태는 아니고(백그라운드 스레드/조회 함수 없음),
단독 실행만 됨. 포트는 파일 상단 `PORT`(`/dev/ttyUSB0`) 수정.

```bash
python3 hardware/ultrasound/ultrasound_reader.py
```



## 젯슨나노 종료
```bash
sudo shutdown -h now
```