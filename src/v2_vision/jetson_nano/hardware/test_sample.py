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

SERVO_STEP_DEG = 0.1

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


def update_telemetry_from_json(data: dict, tel: Telemetry, raw: str) -> None:
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
            update_telemetry_from_json(data, tel, raw)
        except json.JSONDecodeError:
            tel.last_raw = f"JSON parse error: {raw}"


def fmt(value, unit="", width=8) -> str:
    if value is None:
        return f"{'---':>{width}}{unit}"

    try:
        return f"{float(value):>{width}.1f}{unit}"
    except (TypeError, ValueError):
        return f"{'ERR':>{width}}{unit}"


def draw_screen(stdscr, tel: Telemetry, target_x: float, target_y: float, port: str) -> None:
    stdscr.erase()

    stdscr.addstr(0, 0, "Jetson Nano USB Serial Keyboard Servo Controller")
    stdscr.addstr(1, 0, "Keys: A/D = Servo X, W/S = Servo Y, Q = quit")
    stdscr.addstr(2, 0, f"Port: {port} | Baud: {SERIAL_BAUD} | Step: {SERVO_STEP_DEG:.1f} deg")

    stdscr.addstr(4, 0, "< Command Target >")
    stdscr.addstr(5, 0, f"Servo X target: {target_x:8.1f} deg")
    stdscr.addstr(6, 0, f"Servo Y target: {target_y:8.1f} deg")

    stdscr.addstr(8, 0, "< ESP32 Servo State >")
    stdscr.addstr(9, 0, f"Servo X current: {tel.servo_x_deg:8.1f} deg")
    stdscr.addstr(10, 0, f"Servo Y current: {tel.servo_y_deg:8.1f} deg")

    stdscr.addstr(12, 0, "< Ultrasonic >")
    stdscr.addstr(13, 0, f"Front distance: {fmt(tel.front_cm, ' cm')}")

    stdscr.addstr(15, 0, "< Left Side >")
    stdscr.addstr(16, 0, f"L_FRONT: {fmt(tel.left_front_cm, ' cm')}")
    stdscr.addstr(17, 0, f"L_REAR : {fmt(tel.left_rear_cm, ' cm')}")
    stdscr.addstr(18, 0, f"L_DIST : {fmt(tel.left_dist_cm, ' cm')}")
    stdscr.addstr(19, 0, f"L_ANGLE: {fmt(tel.left_angle_deg, ' deg')}")

    stdscr.addstr(21, 0, "< Right Side >")
    stdscr.addstr(22, 0, f"R_FRONT: {fmt(tel.right_front_cm, ' cm')}")
    stdscr.addstr(23, 0, f"R_REAR : {fmt(tel.right_rear_cm, ' cm')}")
    stdscr.addstr(24, 0, f"R_DIST : {fmt(tel.right_dist_cm, ' cm')}")
    stdscr.addstr(25, 0, f"R_ANGLE: {fmt(tel.right_angle_deg, ' deg')}")

    age = time.time() - tel.last_rx_time if tel.last_rx_time > 0 else None
    stdscr.addstr(27, 0, f"Last ESP32 RX: {fmt(age, ' s')} ago")

    stdscr.addstr(29, 0, "< Last raw line >")
    raw_line = tel.last_raw[:110] if tel.last_raw else ""
    stdscr.addstr(30, 0, raw_line)

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