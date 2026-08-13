"""공용 ESP32 계층 — ultrasound/sample.ino와의 시리얼 프로토콜 "한 벌".

main.py(통합 실행)와 guidance/(teach & repeat)가 이 모듈을 같이 쓴다.
프로토콜(=시스템의 계약)이 바뀌면 sample.ino와 여기 두 곳만 맞추면 됨.

프로토콜 (USB 시리얼, 115200bps, 한 줄 = JSON + 개행):
  수신 (~4~10Hz):
    {"type":"telemetry","front":62.3,"left_front":-1,...,"right_front":45.5,
     "right_rear":46.6,...,"servo_x":90,"servo_y":90,"t":123456}
    거리 단위는 cm, 측정 실패는 -1.
  송신:
    {"cmd":"servo","x":0~180,"y":0~180}
"""
import json
import math
import time

import serial

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
SONAR_MAX_M = 3.0   # 이 범위(m)를 벗어나면 무효 처리


class ESP32Link:
    """텔레메트리 수신 + 서보 명령 송신을 하나의 연결로 처리."""

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 reset_wait=2.0, stale_seconds=0.7):
        self.serial = serial.Serial(port, baud, timeout=0)
        if reset_wait:
            time.sleep(reset_wait)  # 포트를 열면 ESP32가 리셋되므로 부팅 대기
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.stale_seconds = stale_seconds
        self.telemetry = {}         # 최신 텔레메트리 (원본 cm 단위 그대로)
        self._buffer = bytearray()
        self._last_valid_pair = {}  # side별 마지막 유효 측면 쌍 캐시 (깜빡임 완화)

    # ---------------- 수신 ----------------

    def poll(self):
        """수신 버퍼에 쌓인 줄을 전부 소화해 telemetry를 최신으로 갱신하고 반환.

        논블로킹 — 새 데이터가 없으면 기존 telemetry를 그대로 반환.
        표시용 보조 키: _last_raw(마지막 원문), _last_rx(monotonic 수신 시각).
        """
        waiting = self.serial.in_waiting
        if waiting:
            self._buffer.extend(self.serial.read(waiting))
        while b"\n" in self._buffer:
            raw, _, self._buffer = self._buffer.partition(b"\n")
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.telemetry["_last_raw"] = f"JSON parse error: {text}"
                continue
            if not isinstance(data, dict) or data.get("type") != "telemetry":
                self.telemetry["_last_raw"] = text
                continue
            self.telemetry.update(data)
            self.telemetry["_last_raw"] = text
            self.telemetry["_last_rx"] = time.monotonic()
        return self.telemetry

    def rx_age(self):
        """마지막 유효 텔레메트리 이후 경과 초. 아직 못 받았으면 None."""
        last = self.telemetry.get("_last_rx")
        return None if last is None else time.monotonic() - last

    def side_pair_m(self, side="right"):
        """측면 앞/뒤 초음파 쌍을 m 단위로 반환. 무효면 None. (guidance 정렬용)

        초음파는 표면 각도/재질에 따라 개별 측정이 간헐적으로 실패(-1)하는 게
        정상이라, 이번 측정이 무효여도 stale_seconds 안의 마지막 유효 쌍을
        대신 반환한다 (깜빡임 한두 번에 판단이 INVALID로 튀지 않게).
        진짜로 stale_seconds 이상 유효값이 없을 때만 None.
        """
        now = time.monotonic()
        age = self.rx_age()
        pair = None
        if age is not None and age <= self.stale_seconds:
            try:
                front = float(self.telemetry[f"{side}_front"]) / 100.0
                rear = float(self.telemetry[f"{side}_rear"]) / 100.0
                if all(math.isfinite(v) and 0.0 < v < SONAR_MAX_M
                       for v in (front, rear)):
                    pair = {"front": front, "rear": rear}
            except (KeyError, TypeError, ValueError):
                pass

        if pair is not None:
            self._last_valid_pair[side] = (pair, now)
            return dict(pair)

        cached = self._last_valid_pair.get(side)
        if cached is not None and now - cached[1] <= self.stale_seconds:
            return dict(cached[0])
        return None

    # ---------------- 송신 ----------------

    def send_servo(self, x_deg, y_deg):
        """서보(조이스틱 액추에이터) 2축 목표각 전송. 90/90 = 중앙 = 정지."""
        msg = json.dumps(
            {"cmd": "servo", "x": round(float(x_deg), 1),
             "y": round(float(y_deg), 1)},
            separators=(",", ":")) + "\n"
        self.serial.write(msg.encode("utf-8"))

    def close(self):
        self.serial.close()
