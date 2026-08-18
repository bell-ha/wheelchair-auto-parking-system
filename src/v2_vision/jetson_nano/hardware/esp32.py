"""공용 ESP32 계층 — ultrasound/sample.ino와의 시리얼 프로토콜 "한 벌".

main.py(통합 실행)와 guidance/(teach & repeat)가 이 모듈을 같이 쓴다.
프로토콜(=시스템의 계약)이 바뀌면 sample.ino와 여기 두 곳만 맞추면 됨.

프로토콜 (USB 시리얼, 115200bps, 한 줄 = JSON + 개행):
  수신 (~4~10Hz):
    {"type":"telemetry","front":62.3,"left_front":-1,...,"right_front":45.5,
     "right_rear":46.6,...,"servo_x":90,"servo_y":90,
     "imu_ready":true,"yaw_active":false,"yaw_calibrating":false,
     "yaw":12.3,"gyro_z":0.02,"gyro_z_bias":-0.11,"t":123456}
    거리 단위는 cm, 측정 실패는 -1. yaw는 deg, gyro는 deg/s.
  송신:
    {"cmd":"servo","x":0~180,"y":0~180}
    {"cmd":"yaw_toggle"}  — yaw 측정 토글. 첫 토글은 바이어스 캘리브레이션
                            (휠체어 정지 필요) → 0점 → 측정 시작. 재토글 시 정지.
"""
import json
import math
import time

import serial

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
SONAR_MAX_M = 3.0   # 이 범위(m)를 벗어나면 무효 처리

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

# ------------------------------------------------------
# 조이스틱 프리셋 (실측 캘리브레이션)
# 서보가 휠체어 조이스틱을 물리적으로 밀기 때문에 중립은 90/90이 아니라
# 실측값 87/85임에 주의 — "정지"는 반드시 NEUTRAL_*로 보내야 한다.
# 좌/우, 전/후의 편향량이 비대칭인 것도 실측 결과 그대로.
# ------------------------------------------------------

NEUTRAL_X_DEG, NEUTRAL_Y_DEG = 87.0, 85.0
# 아래 4개는 "조이스틱을 끝까지 꺾었을 때"의 실측 풀 편향 원본 (참고·보존용)
FORWARD_X_DEG, FORWARD_Y_DEG = 87.0, 35.0
LEFT_X_DEG, LEFT_Y_DEG = 100.0, 78.0
RIGHT_X_DEG, RIGHT_Y_DEG = 72.0, 78.0
REVERSE_X_DEG, REVERSE_Y_DEG = 87.0, 124.0

# 편향 강도 — 풀 편향은 실차에서 너무 빨라서(2026-08-18 테스트 피드백) 각 프리셋을
# "중립 기준 최대 성분이 이 각도"가 되도록 비율 유지한 채 축소해서 사용.
# 실차 튜닝은 아래 방향별 값만 바꾸면 됨 (풀 편향 복원 = 50 이상으로).
# 주의: 조이스틱 데드존보다 작으면 휠체어가 아예 안 움직일 수 있음.
DEFLECT_LEFT_DEG = 10.0
DEFLECT_RIGHT_DEG = 13.0      # 실차: 우회전이 약해서 좌보다 크게 (2026-08-18)
DEFLECT_FORWARD_DEG = 13.0
DEFLECT_BACKWARD_DEG = 13.0


def _scaled_preset(x_deg, y_deg, max_deflect):
    dx, dy = x_deg - NEUTRAL_X_DEG, y_deg - NEUTRAL_Y_DEG
    biggest = max(abs(dx), abs(dy))
    k = 1.0 if biggest <= max_deflect else max_deflect / biggest
    return (round(NEUTRAL_X_DEG + dx * k, 1), round(NEUTRAL_Y_DEG + dy * k, 1))


# 지시 문구 규약은 guidance.common.servo_targets와 동일:
# 이동 지시 4종만 편향, 그 외 문구는 전부 중립=정지로 해석.
_PRESETS = {
    "TURN LEFT": _scaled_preset(LEFT_X_DEG, LEFT_Y_DEG, DEFLECT_LEFT_DEG),
    "TURN RIGHT": _scaled_preset(RIGHT_X_DEG, RIGHT_Y_DEG, DEFLECT_RIGHT_DEG),
    "MOVE FORWARD": _scaled_preset(FORWARD_X_DEG, FORWARD_Y_DEG, DEFLECT_FORWARD_DEG),
    "MOVE BACKWARD": _scaled_preset(REVERSE_X_DEG, REVERSE_Y_DEG, DEFLECT_BACKWARD_DEG),
}
_SWAP_X = {"TURN LEFT": "TURN RIGHT", "TURN RIGHT": "TURN LEFT"}
_SWAP_Y = {"MOVE FORWARD": "MOVE BACKWARD", "MOVE BACKWARD": "MOVE FORWARD"}


def servo_preset(text, invert_x=False, invert_y=False):
    """지시 문구 → 실측 조이스틱 프리셋 (x, y). 이동 지시 4개 외에는 중립=정지.

    어느 방향이 물리적으로 좌/전진인지는 서보-조이스틱 장착 방향에 달렸으므로
    invert_x/invert_y로 현장에서 맞출 것 (좌↔우, 전↔후 프리셋을 맞바꿈).
    """
    if invert_x:
        text = _SWAP_X.get(text, text)
    if invert_y:
        text = _SWAP_Y.get(text, text)
    return _PRESETS.get(text, (NEUTRAL_X_DEG, NEUTRAL_Y_DEG))


def clamp(value, lo=SERVO_MIN_DEG, hi=SERVO_MAX_DEG):
    return max(lo, min(hi, value))


def approach(current, target, step):
    """current를 target 쪽으로 step만큼만 이동 (지나치지 않게)."""
    if current < target:
        return min(current + step, target)
    if current > target:
        return max(current - step, target)
    return current


class ESP32Link:
    """텔레메트리 수신 + 서보/yaw 명령 송신을 하나의 연결로 처리."""

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 reset_wait=2.0, stale_seconds=0.7):
        self.serial = serial.Serial(port, baud, timeout=0, write_timeout=1)
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
        IMU 키(imu_ready/yaw_active/yaw_calibrating/yaw/gyro_z/gyro_z_bias)도
        telemetry 필드에 같은 방식으로 병합된다.
        표시용 보조 키: _last_raw(마지막 원문), _last_rx(monotonic 수신 시각).
        """
        waiting = self.serial.in_waiting
        if waiting:
            self._buffer.extend(self.serial.read(waiting))
        if len(self._buffer) > 10000:
            # 개행이 오지 않는 노이즈 유입 등으로 버퍼가 비정상적으로 커지면 리셋
            self.telemetry["_last_raw"] = "RX buffer overflow - reset"
            self._buffer.clear()
            return self.telemetry
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

    def _send_json(self, msg):
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        self.serial.write(line.encode("utf-8"))

    def send_servo(self, x_deg, y_deg):
        """서보(조이스틱 액추에이터) 2축 목표각 전송. 정지 = NEUTRAL_X/Y_DEG."""
        self._send_json({"cmd": "servo", "x": round(float(x_deg), 1),
                         "y": round(float(y_deg), 1)})

    def send_yaw_toggle(self):
        """MPU6050 yaw 측정 시작/정지 토글 (첫 토글: 캘리브레이션→0점→시작)."""
        self._send_json({"cmd": "yaw_toggle"})

    def close(self):
        self.serial.close()


# ======================================================
# 서보 램프 — 목표각까지 한 번에 점프하지 않고 일정 속도로 접근
# ======================================================

# 1.5 deg / 0.03 s ≈ 50 deg/s.
# 급격한 서보 이동이 전원 브라운아웃(꺼짐/재부팅)을 일으키면 step을 1.0으로 낮출 것.
SERVO_RAMP_STEP_DEG = 1.5
SERVO_RAMP_INTERVAL = 0.03


class ServoRamp:
    """set_target()으로 목표각만 바꾸고, 루프에서 tick()을 계속 불러주면
    interval마다 step_deg씩 목표로 접근하며 send_servo한다.

    keepalive를 주면 목표에 도달해 조용해진 뒤에도 그 주기마다 마지막
    명령을 재전송한다 (수신 측 갱신 유지용).
    시리얼 예외는 삼키지 않고 호출자에게 그대로 전파한다.
    """

    def __init__(self, link, start_x=NEUTRAL_X_DEG, start_y=NEUTRAL_Y_DEG,
                 step_deg=SERVO_RAMP_STEP_DEG, interval=SERVO_RAMP_INTERVAL,
                 keepalive=None):
        self.link = link
        self.step_deg = step_deg
        self.interval = interval
        self.keepalive = keepalive
        self.command_x = start_x    # 실제 마지막 송신값
        self.command_y = start_y
        self.target_x = start_x
        self.target_y = start_y
        self._last_step = 0.0
        self._last_send = 0.0

    def set_target(self, x_deg, y_deg):
        self.target_x = clamp(float(x_deg))
        self.target_y = clamp(float(y_deg))

    @property
    def settled(self):
        return (self.command_x == self.target_x and
                self.command_y == self.target_y)

    def tick(self):
        """한 스텝 진행. 실제로 송신했으면 True."""
        now = time.monotonic()
        if now - self._last_step >= self.interval:
            next_x = approach(self.command_x, self.target_x, self.step_deg)
            next_y = approach(self.command_y, self.target_y, self.step_deg)
            if next_x != self.command_x or next_y != self.command_y:
                self.command_x, self.command_y = next_x, next_y
                self.link.send_servo(next_x, next_y)
                self._last_step = now
                self._last_send = now
                return True
        if (self.keepalive is not None and
                now - self._last_send >= self.keepalive):
            self.link.send_servo(self.command_x, self.command_y)
            self._last_send = now
            return True
        return False
