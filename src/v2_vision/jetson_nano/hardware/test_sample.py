#!/usr/bin/env python3

import curses
import json
import time
from dataclasses import dataclass
from typing import Optional

import serial


# ======================================================
# Config
# ======================================================

SERIAL_PORT = "/dev/ttyUSB0"   # 필요하면 /dev/ttyACM0 로 변경
SERIAL_BAUD = 115200

SERVO_STEP_DEG = 0.5  # 초당 회전속도를 기존 대비 5배로 (0.1 -> 0.5deg / COMMAND_INTERVAL)

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

INITIAL_SERVO_X_DEG = 90.0
INITIAL_SERVO_Y_DEG = 90.0

# 키 입력 체크 주기
LOOP_DT = 0.03  # 약 33 Hz

# 같은 키가 눌려있을 때 너무 빠르게 변하지 않도록 제한
COMMAND_INTERVAL = 0.05  # 50 ms마다 0.1도 변화


# ======================================================
# State
# ======================================================

@dataclass
class Telemetry:
    front_cm: Optional[float] = None

    left_front_cm: Optional[float] = None
    left_rear_cm: Optional[float] = None
    left_dist_cm: Optional[float] = None
    left_angle_deg: Optional[float] = None

    right_front_cm: Optional[float] = None
    right_rear_cm: Optional[float] = None
    right_dist_cm: Optional[float] = None
    right_angle_deg: Optional[float] = None

    servo_x_deg: float = INITIAL_SERVO_X_DEG
    servo_y_deg: float = INITIAL_SERVO_Y_DEG

    imu_ready: Optional[bool] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    last_rx_time: float = 0.0
    last_raw: str = ""


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def send_servo_command(ser: serial.Serial, servo_x_deg: float, servo_y_deg: float) -> None:
    msg = {
        "cmd": "servo",
        "x": round(servo_x_deg, 1),
        "y": round(servo_y_deg, 1),
    }

    line = json.dumps(msg, separators=(",", ":")) + "\n"
    ser.write(line.encode("utf-8"))


def update_telemetry_from_json(data, tel: Telemetry, raw: str) -> None:
    # JSON 파싱은 됐지만 {"type":"telemetry", ...} 형태가 아닌 경우 방어
    if not isinstance(data, dict):
        tel.last_raw = f"ignored non-object JSON: {raw}"
        return

    msg_type = data.get("type", "")

    if msg_type != "telemetry":
        tel.last_raw = raw
        return

    tel.front_cm = data.get("front")

    tel.left_front_cm = data.get("left_front")
    tel.left_rear_cm = data.get("left_rear")
    tel.left_dist_cm = data.get("left_dist")
    tel.left_angle_deg = data.get("left_angle")

    tel.right_front_cm = data.get("right_front")
    tel.right_rear_cm = data.get("right_rear")
    tel.right_dist_cm = data.get("right_dist")
    tel.right_angle_deg = data.get("right_angle")

    tel.servo_x_deg = data.get("servo_x", tel.servo_x_deg)
    tel.servo_y_deg = data.get("servo_y", tel.servo_y_deg)

    tel.imu_ready = data.get("imu_ready", tel.imu_ready)
    tel.yaw_deg = data.get("yaw", tel.yaw_deg)
    tel.pitch_deg = data.get("pitch", tel.pitch_deg)
    tel.roll_deg = data.get("roll", tel.roll_deg)

    tel.last_rx_time = time.time()
    tel.last_raw = raw

def read_serial_nonblocking(ser: serial.Serial, tel: Telemetry) -> None:
    while ser.in_waiting > 0:
        raw_bytes = ser.readline()

        if not raw_bytes:
            break

        raw = raw_bytes.decode("utf-8", errors="ignore").strip()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            tel.last_raw = f"JSON parse error: {raw}"
            continue

        # dict가 아닌 JSON은 무시
        if not isinstance(data, dict):
            tel.last_raw = f"ignored non-object JSON: {raw}"
            continue

        update_telemetry_from_json(data, tel, raw)


def fmt(value, unit="", width=8) -> str:
    if value is None:
        return f"{'---':>{width}}{unit}"

    try:
        return f"{float(value):>{width}.1f}{unit}"
    except (TypeError, ValueError):
        return f"{'ERR':>{width}}{unit}"


def safe_addstr(stdscr, y: int, x: int, text: str) -> None:
    max_y, max_x = stdscr.getmaxyx()

    if y < 0 or y >= max_y or x < 0 or x >= max_x:
        return

    try:
        stdscr.addstr(y, x, text[: max_x - x])
    except curses.error:
        # 커서가 화면 맨 끝(우하단)에 닿으면 ncurses가 ERR을 반환하는 경우가 있어 무시
        pass


def draw_screen(stdscr, tel: Telemetry, target_x: float, target_y: float, port: str) -> None:
    stdscr.erase()

    safe_addstr(stdscr, 0, 0, "Jetson Nano USB Serial Keyboard Servo Controller")
    safe_addstr(stdscr, 1, 0, "Keys: A/D = Servo X, W/S = Servo Y, Q = quit")
    safe_addstr(stdscr, 2, 0, f"Port: {port} | Baud: {SERIAL_BAUD} | Step: {SERVO_STEP_DEG:.1f} deg")

    safe_addstr(stdscr, 4, 0, "< Command Target >")
    safe_addstr(stdscr, 5, 0, f"Servo X target: {target_x:8.1f} deg")
    safe_addstr(stdscr, 6, 0, f"Servo Y target: {target_y:8.1f} deg")

    safe_addstr(stdscr, 8, 0, "< ESP32 Servo State >")
    safe_addstr(stdscr, 9, 0, f"Servo X current: {tel.servo_x_deg:8.1f} deg")
    safe_addstr(stdscr, 10, 0, f"Servo Y current: {tel.servo_y_deg:8.1f} deg")

    safe_addstr(stdscr, 12, 0, "< Ultrasonic >")
    safe_addstr(stdscr, 13, 0, f"Front distance: {fmt(tel.front_cm, ' cm')}")

    safe_addstr(stdscr, 15, 0, "< Left Side >")
    safe_addstr(stdscr, 16, 0, f"L_FRONT: {fmt(tel.left_front_cm, ' cm')}")
    safe_addstr(stdscr, 17, 0, f"L_REAR : {fmt(tel.left_rear_cm, ' cm')}")
    safe_addstr(stdscr, 18, 0, f"L_DIST : {fmt(tel.left_dist_cm, ' cm')}")
    safe_addstr(stdscr, 19, 0, f"L_ANGLE: {fmt(tel.left_angle_deg, ' deg')}")

    safe_addstr(stdscr, 21, 0, "< Right Side >")
    safe_addstr(stdscr, 22, 0, f"R_FRONT: {fmt(tel.right_front_cm, ' cm')}")
    safe_addstr(stdscr, 23, 0, f"R_REAR : {fmt(tel.right_rear_cm, ' cm')}")
    safe_addstr(stdscr, 24, 0, f"R_DIST : {fmt(tel.right_dist_cm, ' cm')}")
    safe_addstr(stdscr, 25, 0, f"R_ANGLE: {fmt(tel.right_angle_deg, ' deg')}")

    age = time.time() - tel.last_rx_time if tel.last_rx_time > 0 else None
    safe_addstr(stdscr, 27, 0, f"Last ESP32 RX: {fmt(age, ' s')} ago")

    safe_addstr(stdscr, 29, 0, "< Last raw line >")
    raw_line = tel.last_raw[:110] if tel.last_raw else ""
    safe_addstr(stdscr, 30, 0, raw_line)

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    tel = Telemetry()

    target_x = INITIAL_SERVO_X_DEG
    target_y = INITIAL_SERVO_Y_DEG

    last_cmd_time = 0.0

    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
    time.sleep(2.0)

    # ESP32 입력 버퍼 정리
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 초기 각도 전송
    send_servo_command(ser, target_x, target_y)

    running = True

    while running:
        now = time.time()

        # ==================================================
        # 1. Keyboard input
        # ==================================================
        key = stdscr.getch()
        changed = False

        if key != -1 and now - last_cmd_time >= COMMAND_INTERVAL:
            if key in (ord("q"), ord("Q")):
                running = False

            elif key in (ord("a"), ord("A")):
                target_x -= SERVO_STEP_DEG
                changed = True

            elif key in (ord("d"), ord("D")):
                target_x += SERVO_STEP_DEG
                changed = True

            elif key in (ord("w"), ord("W")):
                target_y += SERVO_STEP_DEG
                changed = True

            elif key in (ord("s"), ord("S")):
                target_y -= SERVO_STEP_DEG
                changed = True

            if changed:
                target_x = clamp(target_x, SERVO_MIN_DEG, SERVO_MAX_DEG)
                target_y = clamp(target_y, SERVO_MIN_DEG, SERVO_MAX_DEG)

                send_servo_command(ser, target_x, target_y)
                last_cmd_time = now

        # ==================================================
        # 2. Serial receive
        # ==================================================
        read_serial_nonblocking(ser, tel)

        # ==================================================
        # 3. Draw terminal log
        # ==================================================
        draw_screen(stdscr, tel, target_x, target_y, SERIAL_PORT)

        time.sleep(LOOP_DT)

    ser.close()


if __name__ == "__main__":
    curses.wrapper(main)