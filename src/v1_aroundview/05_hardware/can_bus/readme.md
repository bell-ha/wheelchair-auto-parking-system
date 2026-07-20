## A. PC(리눅스) 준비 체크리스트

### A-1) USB-CAN 인식 확인

```bash
lsusb
```

*  `OpenMoko ... CAN adapter`면 보통 `gs_usb` 계열이라 SocketCAN 바로 가능.

### A-2) 필수 패키지

```bash
sudo apt update
sudo apt install -y can-utils
```

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

---

## B. TRACER ↔ USB-CAN 배선 표준

1. 로봇 전원 **OFF 권장** 상태에서 배선

* **CAN_H ↔ CAN_H**
* **CAN_L ↔ CAN_L**

2. 로봇 전원 ON
* **E-stop 해제**
* 주변 안전 확보

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

RC를 꺼둬도 내부 모드가 RC로 남아있을 수 있어서, 이걸로 CAN 통신 전환

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

