#!/usr/bin/env python3
import json
import socket
import threading
import time
from dataclasses import dataclass

import can

# ===== 사용자 설정 =====
CAN_IFACE = "can0"
UDP_BIND_IP = "0.0.0.0"
UDP_PORT = 25001

CMD_ID = 0x111
SEND_PERIOD_S = 0.02      # 20ms
DEADMAN_S = 0.25          # 이 시간 동안 명령 못 받으면 자동정지

# 속도 단위:
#  - v_mm_s: mm/s (int16)
#  - w_mrad_s: 0.001 rad/s 단위 (int16)
# =======================

def int16_be(x: int) -> bytes:
    return int(x).to_bytes(2, byteorder="big", signed=True)

@dataclass
class Cmd:
    v_mm_s: int = 0
    w_mrad_s: int = 0
    t: float = 0.0

class CanTeleopServer:
    def __init__(self, bus: can.Bus):
        self.bus = bus
        self.cmd = Cmd(0, 0, 0.0)
        self.lock = threading.Lock()
        self.running = True

    def set_cmd(self, v_mm_s: int, w_mrad_s: int):
        with self.lock:
            self.cmd.v_mm_s = int(v_mm_s)
            self.cmd.w_mrad_s = int(w_mrad_s)
            self.cmd.t = time.time()

    def get_cmd(self):
        with self.lock:
            return self.cmd.v_mm_s, self.cmd.w_mrad_s, self.cmd.t

    def send_once(self, v_mm_s: int, w_mrad_s: int):
        data = bytearray(8)
        data[0:2] = int16_be(v_mm_s)
        data[2:4] = int16_be(w_mrad_s)
        msg = can.Message(arbitration_id=CMD_ID, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except can.CanError:
            # 송신 실패해도 루프는 유지 (deadman이 안전정지 담당)
            pass

    def can_tx_loop(self):
        while self.running:
            now = time.time()
            v, w, t_last = self.get_cmd()
            if (now - t_last) > DEADMAN_S:
                v, w = 0, 0
            self.send_once(v, w)
            time.sleep(SEND_PERIOD_S)

def udp_rx_loop(server: CanTeleopServer, bind_ip: str, port: int):
    """
    UDP payload (JSON):
      {"v": 200, "w": 0}  # v: mm/s, w: mrad/s(0.001rad/s)
      {"stop": true}
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, port))
    sock.settimeout(0.5)

    while server.running:
        try:
            data, _addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            continue

        if msg.get("stop") is True:
            server.set_cmd(0, 0)
            continue

        if "v" in msg and "w" in msg:
            try:
                v = int(msg["v"])
                w = int(msg["w"])
                print("[UDP RX]", msg)
                server.set_cmd(v, w)
            except Exception:
                pass

def main():
    # python-can 최신 권장: interface="socketcan"
    bus = can.Bus(channel=CAN_IFACE, interface="socketcan")

    srv = CanTeleopServer(bus)

    th_tx = threading.Thread(target=srv.can_tx_loop, daemon=True)
    th_rx = threading.Thread(target=udp_rx_loop, args=(srv, UDP_BIND_IP, UDP_PORT), daemon=True)

    th_tx.start()
    th_rx.start()

    print(f"[SERVER] Listening UDP {UDP_BIND_IP}:{UDP_PORT}")
    print(f"[SERVER] Sending CAN {CAN_IFACE} id=0x{CMD_ID:X} every {SEND_PERIOD_S*1000:.0f}ms, deadman={DEADMAN_S:.2f}s")
    print("[SERVER] Ctrl+C to stop (will send stop frames)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.running = False
        # 정지 프레임 여러 번
        for _ in range(10):
            srv.send_once(0, 0)
            time.sleep(0.02)
        try:
            bus.shutdown()
        except Exception:
            pass
        print("\n[SERVER] stopped")

if __name__ == "__main__":
    main()
