좋아. 지금까지 한 과정을 **“다른 리눅스 PC / 다른 TRACER에서도 그대로 되는 표준 절차”**로 정리해줄게. (너처럼 현장에서 바로 따라 할 수 있게)

아래는 **PC 준비 → 하드웨어 연결 → 모드 전환(리모컨 권한 반납) → 주행 테스트 → WASD 실행** 순서야.

---

## A. PC(리눅스) 준비 체크리스트

### A-1) USB-CAN 인식 확인

```bash
lsusb
```

* 네 어댑터처럼 `OpenMoko ... CAN adapter`면 보통 `gs_usb` 계열이라 SocketCAN 바로 가능.

### A-2) 필수 패키지

```bash
sudo apt update
sudo apt install -y can-utils
```

(파이썬 WASD를 쓸 거면)

```bash
pip3 install python-can
```

### A-3) 드라이버 로드(필요시)

```bash
sudo modprobe gs_usb
```

### A-4) can 인터페이스 생겼는지 확인

```bash
ip link | grep -E "can[0-9]"
```

* `can0` 또는 `can1`이 보여야 정상

> **다른 PC에서 can0 이름이 바뀌는 경우**가 흔해. 항상 `ip link`로 확인하고 그 이름을 쓰면 됨.

---

## B. TRACER ↔ USB-CAN 배선 표준

1. 로봇 전원 **OFF 권장** 상태에서 배선

* **CAN_H ↔ CAN_H**
* **CAN_L ↔ CAN_L**
* 가능하면 **GND도 공통**(노이즈/불안정 줄어듦)

2. 로봇 전원 ON

* **E-stop 해제**
* 주변 안전 확보

> 통신이 들쭉날쭉하면 종단저항(120Ω) 문제일 수도 있어. 하지만 너처럼 프레임이 잘 들어오면 일단 OK.

---

## C. CAN 통신 “켜기” (비트레이트 포함)

TRACER는 **500kbps**.

```bash
sudo ip link set can0 up type can bitrate 500000
```

(인터페이스가 can1이면 can1로)

---

## D. “로봇이 말하는지(피드백)” 먼저 확인

이게 되면 배선/비트레이트/물리계층은 통과야.

```bash
candump can0
```

너처럼 `0x211`, `0x221`, `0x231` 등이 주기적으로 보이면 정상.

---

## E. 핵심: 리모컨 권한 반납(= CAN 제어 모드로 전환)

### E-1) 현재 모드 확인 (0x211)

```bash
candump can0,211:7FF
```

출력 예시:

* `00 00 ...` → **RC 모드**
* `00 01 ...` → **CAN 모드**
* `00 02 ...` → **Serial 모드**

### E-2) CAN 모드로 전환 명령 (ID 0x421)

RC를 꺼둬도 내부 모드가 RC로 남아있을 수 있어서, 이걸 한 번 보내서 **CAN 모드로 바꿔야** 했지.

```bash
cansend can0 421#01
```

다시 확인:

```bash
candump can0,211:7FF
```

`00 01 ...`로 바뀌면 **모드 전환 성공**.

> 이 단계가 바로 네가 말한 “리모컨 권한 반납”에 해당.

---

## F. 전진 테스트(주기 전송이 필수)

TRACER는 보호 때문에 **500ms 이상 명령이 끊기면 0속도**로 떨어져.
그래서 “한 번 보내기”가 아니라 **20ms 주기 전송**이 필요.

아주 느리게 전진(100mm/s):

```bash
while true; do cansend can0 111#0064000000000000; usleep 20000; done
```

멈춤: `Ctrl+C`

속도 피드백 확인(선택):

```bash
candump can0,221:7FF
```

---

## G. WASD 코드가 “다른 PC/다른 로봇”에서도 잘 되게 만드는 팁

### G-1) CAN 인터페이스 이름을 코드에서 바꾸기 쉽게

코드 맨 위 `CAN_IFACE="can0"`를 쓰고 있었지?
이걸 “환경변수로 바꾸게” 해두면 다른 PC에서 can1이어도 편해.

예:

```bash
CAN_IFACE=can1 python3 teleop_wasd_can.py
```

(코드에서는 `os.environ.get("CAN_IFACE","can0")`로 받으면 됨)

### G-2) 실행 전에 “모드 전환 자동화” (추천)

실행 스크립트 하나로 묶어두면 실수 줄어들어.

예: `start_tracer.sh`

```bash
#!/usr/bin/env bash
set -e

IFACE="${1:-can0}"

sudo modprobe gs_usb || true
sudo ip link set "$IFACE" down 2>/dev/null || true
sudo ip link set "$IFACE" up type can bitrate 500000

# CAN 모드로 전환
cansend "$IFACE" 421#01

# 모드 확인(1초만)
timeout 1 candump "$IFACE",211:7FF || true

# WASD 실행
python3 teleop_wasd_can.py
```

실행:

```bash
chmod +x start_tracer.sh
./start_tracer.sh can0
```

### G-3) 권한 이슈 줄이기

매번 `sudo ip link ...` 치기 귀찮으면, 네트워크 권한을 주거나 udev 규칙으로 접근성 개선도 가능해.
하지만 지금은 **실전 안정성**은 `sudo` 쓰는 게 오히려 깔끔함.

---

## H. “안 될 때” 즉시 진단 루틴 (현장용 30초)

1. `ip link`에 can0 있나?
2. `candump can0`에 프레임 뜨나?
3. `candump can0,211:7FF`에서 byte[1]이 01인가?

   * 아니면 `cansend can0 421#01` 다시
4. 그래도 안 움직이면 E-stop/하중/바퀴 들림/바닥저항/보호 플래그 확인

---

원하면 내가 네가 쓰는 현재 `teleop_wasd_can.py`를 **“완전 휴대용 버전”**으로 바꿔줄게:

* `--iface can1` 옵션 지원
* 시작할 때 자동으로 `421#01` 보내고, `0x211`이 `01` 될 때까지 재시도
* 종료 시 정지 프레임 여러 번 송신
* 속도 파라미터를 CLI로 조절 (`--v 200 --w 300`)

말만 해줘.
