# unused — 지금 안 쓰는 코드

## joystick/
휠체어 주행 방향(0/90/180/270 각도 + stop) 제어용 코드. `keyboard.py`로 방향키 →
`joystick_controller.py` → `serial_link.py`로 시리얼 전송까지는 동작하지만,
**이 프로토콜을 받아줄 펌웨어가 아직 없음** (지금 연결된 ESP32는 `ultrasound/sample.ino`이고,
이건 초음파+서보 프로토콜만 처리함 — 주행 모터 명령은 무시됨).

실제 "조이스틱" 역할(방향 조종)은 지금 `ultrasound/test_sample.py`의 서보 제어(A/D/W/S)가
대신하고 있어서, 이 폴더의 프로토콜은 당장 필요 없음. 실제 주행 모터/보드가 준비되면
그때 다시 꺼내 쓰거나, 새 펌웨어에 맞게 프로토콜을 고쳐서 씀.
