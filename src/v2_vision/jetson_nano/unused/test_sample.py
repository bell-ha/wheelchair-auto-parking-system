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

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200

SERVO_STEP_DEG = 0.5

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

INITIAL_SERVO_X_DEG = 90.0
INITIAL_SERVO_Y_DEG = 90.0

# 메인 루프 주기
LOOP_DT = 0.03

# Servo command 제한
COMMAND_INTERVAL = 0.05

# E키 연속 입력 방지
YAW_TOGGLE_INTERVAL = 0.3


# ======================================================
# State
# ======================================================

@dataclass
class Telemetry:

    # Ultrasonic
    front_cm: Optional[float] = None

    left_front_cm: Optional[float] = None
    left_rear_cm: Optional[float] = None
    left_dist_cm: Optional[float] = None
    left_angle_deg: Optional[float] = None

    right_front_cm: Optional[float] = None
    right_rear_cm: Optional[float] = None
    right_dist_cm: Optional[float] = None
    right_angle_deg: Optional[float] = None

    # Servo
    servo_x_deg: float = INITIAL_SERVO_X_DEG
    servo_y_deg: float = INITIAL_SERVO_Y_DEG

    # MPU6050 IMU
    imu_ready: Optional[bool] = None

    yaw_active: bool = False
    yaw_calibrating: bool = False

    yaw_deg: Optional[float] = None

    gyro_z_dps: Optional[float] = None
    gyro_z_bias_dps: Optional[float] = None

    # Serial
    last_rx_time: float = 0.0
    last_raw: str = ""

    # Parse error count
    parse_error_count: int = 0


# ======================================================
# Utility
# ======================================================

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(
        min_value,
        min(max_value, value)
    )


def fmt(value, unit="", width=8, precision=2) -> str:

    if value is None:
        return f"{'---':>{width}}{unit}"

    try:
        return f"{float(value):>{width}.{precision}f}{unit}"

    except (TypeError, ValueError):
        return f"{'ERR':>{width}}{unit}"


# ======================================================
# Serial TX
# ======================================================

def send_json_command(
    ser: serial.Serial,
    msg: dict
) -> None:

    line = (
        json.dumps(
            msg,
            separators=(",", ":")
        )
        + "\n"
    )

    ser.write(
        line.encode("utf-8")
    )

    ser.flush()


def send_servo_command(
    ser: serial.Serial,
    servo_x_deg: float,
    servo_y_deg: float
) -> None:

    msg = {
        "cmd": "servo",
        "x": round(servo_x_deg, 1),
        "y": round(servo_y_deg, 1),
    }

    send_json_command(
        ser,
        msg
    )


def send_yaw_toggle(
    ser: serial.Serial
) -> None:

    msg = {
        "cmd": "yaw_toggle"
    }

    send_json_command(
        ser,
        msg
    )


# ======================================================
# Telemetry JSON
# ======================================================

def update_telemetry_from_json(
    data,
    tel: Telemetry,
    raw: str
) -> None:

    if not isinstance(data, dict):

        tel.last_raw = (
            f"ignored non-object JSON: {raw}"
        )

        return

    msg_type = data.get(
        "type",
        ""
    )

    # ==================================================
    # Status / Error
    # ==================================================

    if msg_type != "telemetry":

        tel.last_raw = raw

        return

    # ==================================================
    # Ultrasonic
    # ==================================================

    tel.front_cm = data.get(
        "front",
        tel.front_cm
    )

    tel.left_front_cm = data.get(
        "left_front",
        tel.left_front_cm
    )

    tel.left_rear_cm = data.get(
        "left_rear",
        tel.left_rear_cm
    )

    tel.left_dist_cm = data.get(
        "left_dist",
        tel.left_dist_cm
    )

    tel.left_angle_deg = data.get(
        "left_angle",
        tel.left_angle_deg
    )

    tel.right_front_cm = data.get(
        "right_front",
        tel.right_front_cm
    )

    tel.right_rear_cm = data.get(
        "right_rear",
        tel.right_rear_cm
    )

    tel.right_dist_cm = data.get(
        "right_dist",
        tel.right_dist_cm
    )

    tel.right_angle_deg = data.get(
        "right_angle",
        tel.right_angle_deg
    )

    # ==================================================
    # Servo
    # ==================================================

    tel.servo_x_deg = data.get(
        "servo_x",
        tel.servo_x_deg
    )

    tel.servo_y_deg = data.get(
        "servo_y",
        tel.servo_y_deg
    )

    # ==================================================
    # MPU6050 IMU
    # ==================================================

    tel.imu_ready = data.get(
        "imu_ready",
        tel.imu_ready
    )

    tel.yaw_active = data.get(
        "yaw_active",
        tel.yaw_active
    )

    tel.yaw_calibrating = data.get(
        "yaw_calibrating",
        tel.yaw_calibrating
    )

    tel.yaw_deg = data.get(
        "yaw",
        tel.yaw_deg
    )

    tel.gyro_z_dps = data.get(
        "gyro_z",
        tel.gyro_z_dps
    )

    tel.gyro_z_bias_dps = data.get(
        "gyro_z_bias",
        tel.gyro_z_bias_dps
    )

    # ==================================================
    # RX timestamp
    # ==================================================

    tel.last_rx_time = time.time()
    tel.last_raw = raw


# ======================================================
# Serial RX
# ======================================================

def read_serial_nonblocking(
    ser: serial.Serial,
    tel: Telemetry
) -> None:

    """
    timeout=0 환경에서 readline()을 직접 사용하면
    JSON 한 줄이 완전히 도착하기 전에 일부만 반환될 수 있음.

    따라서 직접 문자열 버퍼를 유지하고
    '\n'까지 도착한 완전한 줄만 JSON parsing 한다.
    """

    if not hasattr(
        read_serial_nonblocking,
        "rx_buffer"
    ):

        read_serial_nonblocking.rx_buffer = ""

    waiting = ser.in_waiting

    if waiting <= 0:
        return

    raw_bytes = ser.read(
        waiting
    )

    if not raw_bytes:
        return

    chunk = raw_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    read_serial_nonblocking.rx_buffer += chunk

    # 안전장치:
    # 비정상적으로 버퍼가 커지는 경우 초기화
    if len(
        read_serial_nonblocking.rx_buffer
    ) > 10000:

        tel.last_raw = (
            "RX buffer overflow - reset"
        )

        read_serial_nonblocking.rx_buffer = ""

        return

    # ==================================================
    # 완전한 한 줄만 처리
    # ==================================================

    while "\n" in read_serial_nonblocking.rx_buffer:

        line, read_serial_nonblocking.rx_buffer = (
            read_serial_nonblocking.rx_buffer.split(
                "\n",
                1
            )
        )

        raw = line.strip()

        if not raw:
            continue

        try:

            data = json.loads(
                raw
            )

        except json.JSONDecodeError as e:

            tel.parse_error_count += 1

            tel.last_raw = (
                f"JSON parse error #{tel.parse_error_count}: "
                f"{e} | RAW: {raw}"
            )

            continue

        if not isinstance(
            data,
            dict
        ):

            tel.last_raw = (
                f"ignored non-object JSON: "
                f"{raw}"
            )

            continue

        update_telemetry_from_json(
            data,
            tel,
            raw
        )


# ======================================================
# Screen Utility
# ======================================================

def safe_addstr(
    stdscr,
    y: int,
    x: int,
    text: str
) -> None:

    max_y, max_x = (
        stdscr.getmaxyx()
    )

    if (
        y < 0
        or y >= max_y
        or x < 0
        or x >= max_x
    ):

        return

    try:

        stdscr.addstr(
            y,
            x,
            text[
                : max_x - x
            ]
        )

    except curses.error:

        pass


# ======================================================
# Draw screen
# ======================================================

def draw_screen(
    stdscr,
    tel: Telemetry,
    target_x: float,
    target_y: float,
    port: str
) -> None:

    stdscr.erase()

    # ==================================================
    # Header
    # ==================================================

    safe_addstr(
        stdscr,
        0,
        0,
        "Jetson Nano Wheelchair Controller"
    )

    safe_addstr(
        stdscr,
        1,
        0,
        (
            "A/D: Servo X | "
            "W/S: Servo Y | "
            "E: Yaw Start/Stop | "
            "Q: Quit"
        )
    )

    safe_addstr(
        stdscr,
        2,
        0,
        (
            f"Port: {port} | "
            f"Baud: {SERIAL_BAUD} | "
            f"Servo Step: {SERVO_STEP_DEG:.1f} deg"
        )
    )

    # ==================================================
    # Servo
    # ==================================================

    safe_addstr(
        stdscr,
        4,
        0,
        "< Servo Command >"
    )

    safe_addstr(
        stdscr,
        5,
        0,
        (
            f"Servo X target : "
            f"{target_x:8.1f} deg"
        )
    )

    safe_addstr(
        stdscr,
        6,
        0,
        (
            f"Servo Y target : "
            f"{target_y:8.1f} deg"
        )
    )

    safe_addstr(
        stdscr,
        7,
        0,
        (
            f"Servo X current: "
            f"{tel.servo_x_deg:8.1f} deg"
        )
    )

    safe_addstr(
        stdscr,
        8,
        0,
        (
            f"Servo Y current: "
            f"{tel.servo_y_deg:8.1f} deg"
        )
    )

    # ==================================================
    # MPU6050 IMU
    # ==================================================

    safe_addstr(
        stdscr,
        10,
        0,
        "< MPU6050 Relative Rotation >"
    )

    if tel.imu_ready is True:

        imu_status = "READY"

    elif tel.imu_ready is False:

        imu_status = "NOT READY"

    else:

        imu_status = "---"

    safe_addstr(
        stdscr,
        11,
        0,
        f"MPU6050 status : {imu_status}"
    )

    if tel.yaw_calibrating:

        yaw_state = (
            "CALIBRATING - KEEP WHEELCHAIR STILL"
        )

    elif tel.yaw_active:

        yaw_state = "MEASURING"

    else:

        yaw_state = "STOPPED"

    safe_addstr(
        stdscr,
        12,
        0,
        f"Yaw state  : {yaw_state}"
    )

    safe_addstr(
        stdscr,
        13,
        0,
        (
            "Yaw angle  : "
            f"{fmt(tel.yaw_deg, ' deg', 9, 2)}"
        )
    )

    safe_addstr(
        stdscr,
        14,
        0,
        (
            "Gyro Z     : "
            f"{fmt(tel.gyro_z_dps, ' dps', 9, 3)}"
        )
    )

    safe_addstr(
        stdscr,
        15,
        0,
        (
            "Gyro bias  : "
            f"{fmt(tel.gyro_z_bias_dps, ' dps', 9, 4)}"
        )
    )

    safe_addstr(
        stdscr,
        16,
        0,
        (
            "E first: calibrate -> zero -> start | "
            "E again: stop"
        )
    )

    # ==================================================
    # Ultrasonic
    # ==================================================

    safe_addstr(
        stdscr,
        18,
        0,
        "< Ultrasonic >"
    )

    safe_addstr(
        stdscr,
        19,
        0,
        (
            "Front  : "
            f"{fmt(tel.front_cm, ' cm')}"
        )
    )

    # Left

    safe_addstr(
        stdscr,
        21,
        0,
        "< Left Side >"
    )

    safe_addstr(
        stdscr,
        22,
        0,
        (
            "L_FRONT: "
            f"{fmt(tel.left_front_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        23,
        0,
        (
            "L_REAR : "
            f"{fmt(tel.left_rear_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        24,
        0,
        (
            "L_DIST : "
            f"{fmt(tel.left_dist_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        25,
        0,
        (
            "L_ANGLE: "
            f"{fmt(tel.left_angle_deg, ' deg')}"
        )
    )

    # Right

    safe_addstr(
        stdscr,
        27,
        0,
        "< Right Side >"
    )

    safe_addstr(
        stdscr,
        28,
        0,
        (
            "R_FRONT: "
            f"{fmt(tel.right_front_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        29,
        0,
        (
            "R_REAR : "
            f"{fmt(tel.right_rear_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        30,
        0,
        (
            "R_DIST : "
            f"{fmt(tel.right_dist_cm, ' cm')}"
        )
    )

    safe_addstr(
        stdscr,
        31,
        0,
        (
            "R_ANGLE: "
            f"{fmt(tel.right_angle_deg, ' deg')}"
        )
    )

    # ==================================================
    # Serial status
    # ==================================================

    if tel.last_rx_time > 0:

        age = (
            time.time()
            - tel.last_rx_time
        )

    else:

        age = None

    safe_addstr(
        stdscr,
        33,
        0,
        (
            "Last ESP32 RX: "
            f"{fmt(age, ' s', 8, 2)} ago"
        )
    )

    safe_addstr(
        stdscr,
        34,
        0,
        (
            f"JSON parse errors: "
            f"{tel.parse_error_count}"
        )
    )

    safe_addstr(
        stdscr,
        36,
        0,
        "< Last raw / status >"
    )

    raw_line = (
        tel.last_raw[:150]
        if tel.last_raw
        else ""
    )

    safe_addstr(
        stdscr,
        37,
        0,
        raw_line
    )

    stdscr.refresh()


# ======================================================
# Main
# ======================================================

def main(stdscr):

    curses.curs_set(0)

    stdscr.nodelay(True)
    stdscr.timeout(1)

    tel = Telemetry()

    target_x = (
        INITIAL_SERVO_X_DEG
    )

    target_y = (
        INITIAL_SERVO_Y_DEG
    )

    last_servo_cmd_time = 0.0
    last_yaw_toggle_time = 0.0

    # ==================================================
    # Serial open
    # ==================================================

    ser = serial.Serial(
        SERIAL_PORT,
        SERIAL_BAUD,
        timeout=0,
        write_timeout=1
    )

    # ESP32 USB serial 연결 시 reset 대기
    time.sleep(2.0)

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 초기 Servo 90 / 90
    send_servo_command(
        ser,
        target_x,
        target_y
    )

    running = True

    try:

        while running:

            now = time.time()

            # ==========================================
            # Keyboard
            # ==========================================

            key = stdscr.getch()

            changed = False

            if key != -1:

                # --------------------------------------
                # Q = Quit
                # --------------------------------------

                if key in (
                    ord("q"),
                    ord("Q")
                ):

                    running = False

                # --------------------------------------
                # E = Yaw start / stop
                # --------------------------------------

                elif key in (
                    ord("e"),
                    ord("E")
                ):

                    if (
                        now
                        - last_yaw_toggle_time
                        >= YAW_TOGGLE_INTERVAL
                    ):

                        send_yaw_toggle(
                            ser
                        )

                        last_yaw_toggle_time = now

                # --------------------------------------
                # Servo control
                # --------------------------------------

                elif (
                    now
                    - last_servo_cmd_time
                    >= COMMAND_INTERVAL
                ):

                    if key in (
                        ord("a"),
                        ord("A")
                    ):

                        target_x -= (
                            SERVO_STEP_DEG
                        )

                        changed = True

                    elif key in (
                        ord("d"),
                        ord("D")
                    ):

                        target_x += (
                            SERVO_STEP_DEG
                        )

                        changed = True

                    elif key in (
                        ord("w"),
                        ord("W")
                    ):

                        target_y += (
                            SERVO_STEP_DEG
                        )

                        changed = True

                    elif key in (
                        ord("s"),
                        ord("S")
                    ):

                        target_y -= (
                            SERVO_STEP_DEG
                        )

                        changed = True

                    if changed:

                        target_x = clamp(
                            target_x,
                            SERVO_MIN_DEG,
                            SERVO_MAX_DEG
                        )

                        target_y = clamp(
                            target_y,
                            SERVO_MIN_DEG,
                            SERVO_MAX_DEG
                        )

                        send_servo_command(
                            ser,
                            target_x,
                            target_y
                        )

                        last_servo_cmd_time = now

            # ==========================================
            # Serial RX
            # ==========================================

            read_serial_nonblocking(
                ser,
                tel
            )

            # ==========================================
            # Screen
            # ==========================================

            draw_screen(
                stdscr,
                tel,
                target_x,
                target_y,
                SERIAL_PORT
            )

            time.sleep(
                LOOP_DT
            )

    finally:

        ser.close()


# ======================================================
# Entry
# ======================================================

if __name__ == "__main__":

    curses.wrapper(
        main
    )