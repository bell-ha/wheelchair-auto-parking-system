# jetson_nano — 젯슨 나노 배포용 폴더

카메라 실시간 추론 + 하드웨어(초음파/수동조종) 코드 모음.
학습 코드·데이터셋은 없음 (추론 전용).

```
jetson_nano/
├── models/
│   └── best_v5_poster.pt    # 최신 모델
├── scripts/
│   └── live_camera.py       # 카메라 실시간 YOLO 추론 (USB 웹캠, C920)
├── hardware/
│   └── ultrasound/
│       ├── ultrasound.ino   # 아두이노: 초음파 3개로 벽 각도/거리 계산
│       └── keyboard.py      # 방향키 → ESP32(USB 시리얼) 수동 조종
└── requirements.txt
```

## 젯슨 접속 (SSH)

맥과 젯슨이 같은 네트워크(같은 공유기)에 있어야 함. 계정 `jetson`, 호스트명 `nano`.

```bash
# 맥에서 — 호스트명으로 접속 (mDNS)
ssh jetson@nano.local

# 안 되면 IP로
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
```

`ultrasound.ino`는 아두이노 IDE로 아두이노 보드에 업로드 (젯슨 아님).

## 실행

```bash
cd ~/GitHub/wheelchair-auto-parking-system/src/v2_vision/jetson_nano
python3 scripts/live_camera.py models/best_v5_poster.pt
```

부하 확인은 다른 터미널에서 `sudo tegrastats` 병행 권장 (GPU 사용률·온도·스로틀링 확인).
자세한 결과는 [../jetson_부하테스트_보고서.md](../jetson_부하테스트_보고서.md) 참고.

## 수동 조종 (keyboard.py)

방향키 → ESP32에 USB 시리얼로 각도(0/90/180/270) 전송, `s`로 정지.
Jetson과 ESP32는 USB 케이블로 직결, 포트는 `/dev/ttyUSB0`, 115200bps
(필요 시 파일 상단 `SERIAL_PORT`/`BAUD_RATE` 수정).

```bash
python3 hardware/ultrasound/keyboard.py
```
