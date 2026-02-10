#!/usr/bin/env python3
import curses
import json
import socket
import time

# ===== 사용자 설정 =====
SERVER_IP = "172.25.244.144"   # 리눅스 PC IP로 바꾸기
SERVER_PORT = 25001

SEND_HZ = 30                   # 20~50 추천
# 속도 (원하면 여기만 조절)
FWD_MM_S = 200
REV_MM_S = -200
YAW_MRAD_S = 300               # 0.300 rad/s
# =======================

def send(sock, addr, payload: dict):
    sock.sendto(json.dumps(payload).encode("utf-8"), addr)

def ui(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (SERVER_IP, SERVER_PORT)

    v, w = 0, 0
    last_send = 0.0
    dt = 1.0 / float(SEND_HZ)

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, "Mac WASD Client → Linux CAN Server (UDP)")
        stdscr.addstr(1, 0, f"Target: {SERVER_IP}:{SERVER_PORT}   Send: {SEND_HZ} Hz")
        stdscr.addstr(3, 0, "W: forward   S: reverse   A: left yaw   D: right yaw")
        stdscr.addstr(4, 0, "X / Space: STOP   Q: QUIT")
        stdscr.addstr(6, 0, f"Current cmd: v={v} mm/s, w={w} mrad/s")
        stdscr.addstr(8, 0, "Tip: 키를 눌러 방향 바꾸고, X/Space로 정지")

        ch = stdscr.getch()
        if ch != -1:
            c = chr(ch).lower() if 0 <= ch < 256 else ""
            if c == "w":
                v, w = FWD_MM_S, 0
            elif c == "s":
                v, w = REV_MM_S, 0
            elif c == "a":
                v, w = 0, YAW_MRAD_S
            elif c == "d":
                v, w = 0, -YAW_MRAD_S
            elif c == "x" or ch == ord(" "):
                v, w = 0, 0
                send(sock, addr, {"stop": True})
            elif c == "q":
                send(sock, addr, {"stop": True})
                return

        now = time.time()
        if (now - last_send) >= dt:
            send(sock, addr, {"v": v, "w": w})
            last_send = now

        time.sleep(0.005)

def main():
    curses.wrapper(ui)

if __name__ == "__main__":
    main()
