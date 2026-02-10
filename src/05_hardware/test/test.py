#!/usr/bin/env python3
import time
import threading
import curses
import can

CAN_IFACE = "can0"
CMD_ID = 0x111

# ====== 튜닝 파라미터 ======
FWD_MM_S = 200        # W 전진 속도 (mm/s)
REV_MM_S = -200       # S 후진 속도 (mm/s)
YAW_MRAD_S = 300      # A/D 회전 속도 (0.001 rad/s 단위). 300 => 0.300 rad/s
PERIOD_S = 0.02       # 20ms
DEADMAN_S = 0.25      # 이 시간동안 키 입력 없으면 자동 정지
# ==========================

def int16_be(x: int) -> bytes:
    return int(x).to_bytes(2, byteorder="big", signed=True)

class CanTeleop:
    def __init__(self, bus: can.Bus):
        self.bus = bus
        self._lock = threading.Lock()
        self.v_mm_s = 0
        self.w_mrad_s = 0
        self.last_input_t = time.time()
        self.running = True

    def set_cmd(self, v_mm_s: int, w_mrad_s: int):
        with self._lock:
            self.v_mm_s = v_mm_s
            self.w_mrad_s = w_mrad_s
            self.last_input_t = time.time()

    def stop(self):
        self.set_cmd(0, 0)

    def _send_once(self, v_mm_s: int, w_mrad_s: int):
        data = bytearray(8)
        data[0:2] = int16_be(v_mm_s)     # mm/s
        data[2:4] = int16_be(w_mrad_s)   # 0.001 rad/s
        msg = can.Message(arbitration_id=CMD_ID, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except can.CanError:
            # 버스 오류가 나도 루프는 유지 (안전상 자동정지 로직이 있음)
            pass

    def tx_loop(self):
        """20ms 주기로 계속 전송 + deadman 정지"""
        while self.running:
            now = time.time()
            with self._lock:
                v = self.v_mm_s
                w = self.w_mrad_s
                dt = now - self.last_input_t

            # 키 입력 끊기면 자동 정지
            if dt > DEADMAN_S:
                v, w = 0, 0

            self._send_once(v, w)
            time.sleep(PERIOD_S)

def ui_loop(stdscr, teleop: CanTeleop):
    curses.curs_set(0)
    stdscr.nodelay(True)   # getch() non-blocking
    stdscr.keypad(True)

    while teleop.running:
        stdscr.erase()
        stdscr.addstr(0, 0, "TRACER CAN Teleop (W/A/S/D)  Space:STOP  Q:QUIT")
        stdscr.addstr(2, 0, f"W: forward  ({FWD_MM_S} mm/s)")
        stdscr.addstr(3, 0, f"S: reverse  ({REV_MM_S} mm/s)")
        stdscr.addstr(4, 0, f"A: left yaw ({YAW_MRAD_S} mrad/s)")
        stdscr.addstr(5, 0, f"D: right yaw({-YAW_MRAD_S} mrad/s)")
        stdscr.addstr(7, 0, f"Deadman: {DEADMAN_S:.2f}s  Send period: {PERIOD_S*1000:.0f}ms")
        stdscr.addstr(9, 0, "Tip: 키를 계속 누르면 그 방향 유지, 손 떼면 잠깐 후 자동정지")

        ch = stdscr.getch()
        if ch != -1:
            c = chr(ch).lower() if 0 <= ch < 256 else ""
            if c == "w":
                teleop.set_cmd(FWD_MM_S, 0)
            elif c == "s":
                teleop.set_cmd(REV_MM_S, 0)
            elif c == "a":
                teleop.set_cmd(0, YAW_MRAD_S)   # +면 왼쪽/오른쪽은 로봇 좌표계 따라 다를 수 있음
            elif c == "d":
                teleop.set_cmd(0, -YAW_MRAD_S)
            elif ch == ord(" "):
                teleop.stop()
            elif c == "q":
                teleop.stop()
                teleop.running = False

        time.sleep(0.03)

def main():
    bus = can.interface.Bus(channel=CAN_IFACE, bustype="socketcan")
    teleop = CanTeleop(bus)

    th = threading.Thread(target=teleop.tx_loop, daemon=True)
    th.start()

    try:
        curses.wrapper(ui_loop, teleop)
    finally:
        teleop.running = False
        # 마지막으로 정지 프레임 몇 번 보내기 (안전)
        for _ in range(10):
            teleop._send_once(0, 0)
            time.sleep(0.02)

if __name__ == "__main__":
    main()
