"""연결된 UWB 보드 점검 — 재부팅시켜 부팅 메시지를 읽고, 어떤 펌웨어가 올라가 있는지 보여준다.

  python3 tools/check_board.py              # 연결된 보드 전부
  python3 tools/check_board.py --port /dev/cu.usbserial-030ADFCA --secs 5

보는 것:
  - DW3000 칩 ID: deca0302 이면 정상 (DW3000 C0, PDoA 없는 단일 안테나 칩)
  - 펌웨어: "# uwb_tag" / "# uwb_anchor" / 공장 테스트("SIMPLE TX" 등)
  - "XTRIM OTP READ FAIL"은 호환 모듈이라 수정발진기 보정값이 비어 있다는 경고. 기본값으로 동작한다

보드가 USB 장치로는 잡히는데 여기 안 나오면 (시스템 정보에는 CP2104가 보이는데 /dev/cu.usbserial-* 가 없음)
드라이버가 붙지 않은 것이다. 포트는 있는데 열 때 'Invalid argument'가 나는 것도 같은 부류다
(2026-10-29 DFBB·DFCA 둘 다 겪음 — 맥 기본 CP2104 드라이버 쪽 문제로 봄). 그 보드를 뽑았다 다시 꽂는다.
"""
import argparse
import termios
import time

import serial

from uwb_proto import find_ports, open_and_reset

# 부팅 로그에서 보여줄 줄 (ESP32 ROM 로그는 생략)
KEYS = ("#", "DEVICE ID", "SIMPLE", "Range", "XTRIM", "FAILED", "ERROR", "R,", "E,", "DIST")


def peek(port, secs):
    s = open_and_reset(port)
    end = time.time() + secs
    buf = b""
    while time.time() < end:
        buf += s.read(4096)
    s.close()
    lines = buf.decode("utf8", "replace").splitlines()
    return [l for l in lines if l.strip().startswith(KEYS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="생략하면 연결된 CP2104 보드 전부")
    ap.add_argument("--secs", type=float, default=3.0, help="보드당 읽는 시간 (초)")
    args = ap.parse_args()

    ports = [(args.port, "?")] if args.port else find_ports()
    if not ports:
        print("연결된 보드 없음 (CP2104). 케이블이 충전 전용인지, 드라이버가 붙었는지 확인")
        return
    for dev, sn in ports:
        sn = sn if sn != "?" else dev.rsplit("-", 1)[-1]
        print(f"=== {dev}  (고유번호 {sn}, 라벨 '{sn[-4:]}')")
        try:
            lines = peek(dev, args.secs)
        except (serial.SerialException, termios.error, OSError) as e:
            print(f"  포트를 열 수 없음: {e}")
            print("  → 다시 꽂거나 케이블·USB 포트를 바꿔 볼 것. 시리얼 모니터가 열려 있으면 닫을 것")
            continue
        if not lines:
            print("  (출력 없음 — 펌웨어가 없거나 통신 속도가 다름)")
        for l in lines[:12]:
            print("  " + l)
        if len(lines) > 12:
            print(f"  ... ({len(lines) - 12}줄 더)")


if __name__ == "__main__":
    main()
