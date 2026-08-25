"""전/후진 드리프트 계측 도구 — 카메라/YOLO 없이 시리얼만 사용 (시작 ~3초).

앞바퀴(캐스터)가 어느 쪽으로 틀어져 있는지 모르는 상태에서 전/후진 명령이
실제로 얼마나·어느 쪽으로 휘는지를 초음파(+IMU yaw)로 정량 기록한다.
여기서 얻은 데이터가 2단계(센서 피드백 직진 보정 루프)의 설계 근거가 됨.

키:
  W = 전진   S = 후진   SPACE = 정지
  E = yaw 측정 토글 (첫 토글은 캘리브레이션 — 반드시 정지 상태에서!)
  Q = 종료 (중립 복귀 후 종료)

안전:
  - 전진 중 전방 초음파가 30cm 미만이면 자동 정지
  - 어떤 경로로 끝나든 조이스틱 중립 복귀

기록: experiments/logs/drive_<시각>.csv — 0.1초마다 상태+센서 전부.
      해석 예: 전진 중 right_front-right_rear 차이가 커지면 = 우측 벽 대비
      기울어지는 중 = 휘고 있음. yaw가 있으면 그게 가장 직접적인 지표.
"""
import csv
import curses
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hardware.esp32 import (ESP32Link, ServoRamp, NEUTRAL_X_DEG, NEUTRAL_Y_DEG,
                            servo_preset)

LOOP_DT = 0.03            # 키 폴링 ~33Hz
LOG_INTERVAL = 0.1        # CSV 기록 10Hz
DRAW_INTERVAL = 0.1
FRONT_STOP_CM = 30.0      # 전진 중 이 거리 미만이면 자동 정지
YAW_TOGGLE_COOLDOWN = 0.5

CSV_FIELDS = ["t", "state", "front", "left_front", "left_rear",
              "right_front", "right_rear", "yaw", "gyro_z",
              "cmd_x", "cmd_y", "echo_x", "echo_y"]


def fmt(v, width=7):
    if v is None:
        return f"{'---':>{width}}"
    try:
        return f"{float(v):>{width}.1f}"
    except (TypeError, ValueError):
        return f"{'ERR':>{width}}"


def put(stdscr, y, text):
    try:
        stdscr.addstr(y, 0, "".join(c for c in text if c.isprintable() or c == " "))
    except curses.error:
        pass


def run(stdscr, link, writer, fp, log_path):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    ramp = ServoRamp(link, keepalive=0.5)
    state = "STOP"
    note = ""
    n_samples = 0
    t0 = time.monotonic()
    last_log = 0.0
    last_draw = 0.0
    last_yaw_toggle = 0.0

    while True:
        now = time.monotonic()

        # 1. 키 입력
        key = stdscr.getch()
        if key != -1:
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("w"), ord("W")):
                state, note = "MOVE FORWARD", ""
            elif key in (ord("s"), ord("S")):
                state, note = "MOVE BACKWARD", ""
            elif key == ord(" "):
                state, note = "STOP", ""
            elif key in (ord("e"), ord("E")):
                if now - last_yaw_toggle > YAW_TOGGLE_COOLDOWN:
                    link.send_yaw_toggle()
                    last_yaw_toggle = now

        # 2. 수신 + 전방 자동 정지
        tel = link.poll()
        front = tel.get("front")
        try:
            front_v = float(front)
        except (TypeError, ValueError):
            front_v = -1.0
        if state == "MOVE FORWARD" and 0.0 < front_v < FRONT_STOP_CM:
            state = "STOP"
            note = f"!! 전방 {front_v:.0f}cm — 자동 정지"

        # 3. 서보 (프리셋 → 램프로 접근)
        ramp.set_target(*servo_preset(state))
        ramp.tick()

        # 4. CSV 기록 (10Hz)
        if now - last_log >= LOG_INTERVAL:
            writer.writerow({
                "t": round(now - t0, 2), "state": state,
                "front": tel.get("front"),
                "left_front": tel.get("left_front"),
                "left_rear": tel.get("left_rear"),
                "right_front": tel.get("right_front"),
                "right_rear": tel.get("right_rear"),
                "yaw": tel.get("yaw"), "gyro_z": tel.get("gyro_z"),
                "cmd_x": ramp.command_x, "cmd_y": ramp.command_y,
                "echo_x": tel.get("servo_x"), "echo_y": tel.get("servo_y"),
            })
            n_samples += 1
            last_log = now
            if n_samples % 10 == 0:
                fp.flush()   # 강제종료돼도 기록이 날아가지 않게 1초마다 디스크 반영

        # 5. 화면 (10Hz)
        if now - last_draw >= DRAW_INTERVAL:
            stdscr.erase()
            put(stdscr, 0, "전/후진 드리프트 계측 — W=전진 S=후진 SPACE=정지 E=yaw토글 Q=종료")
            put(stdscr, 2, f"상태: {state}   {note}")
            put(stdscr, 4, f"서보 명령 x={ramp.command_x:5.1f} y={ramp.command_y:5.1f}"
                            f"   에코 x={fmt(tel.get('servo_x'), 5)} y={fmt(tel.get('servo_y'), 5)}")
            put(stdscr, 6, f"전방(cm): {fmt(front)}   (전진 자동정지: {FRONT_STOP_CM:.0f}cm)")
            put(stdscr, 7, f"우측(cm): 앞 {fmt(tel.get('right_front'))}  뒤 {fmt(tel.get('right_rear'))}"
                            f"   차이 = 기울어짐 지표")
            put(stdscr, 8, f"좌측(cm): 앞 {fmt(tel.get('left_front'))}  뒤 {fmt(tel.get('left_rear'))}")
            yaw_state = ("측정중" if tel.get("yaw_active") else
                         "캘리브레이션중" if tel.get("yaw_calibrating") else "꺼짐(E로 시작)")
            put(stdscr, 10, f"IMU yaw: {fmt(tel.get('yaw'))} deg  [{yaw_state}]"
                             f"   gyro_z {fmt(tel.get('gyro_z'))}")
            put(stdscr, 12, f"기록: {n_samples}줄 → {log_path.name}")
            rx = link.rx_age()
            put(stdscr, 13, f"ESP32 수신: {fmt(rx, 5)}s 전" + ("  !! 수신 끊김" if (rx or 99) > 1 else ""))
            stdscr.refresh()
            last_draw = now

        time.sleep(LOOP_DT)


def main():
    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"drive_{datetime.now():%Y%m%d_%H%M%S}.csv"

    print("ESP32 연결 중... (2초 리셋 대기)")
    try:
        link = ESP32Link()
    except serial.SerialException as e:
        sys.exit(f"ESP32 연결 실패: {e}")
    print(f"연결됨 — 기록 파일: {log_path}")

    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            curses.wrapper(run, link, writer, fp, log_path)
        except KeyboardInterrupt:
            pass
        finally:
            # 어떤 경로로 끝나든 조이스틱 중립 복귀
            try:
                for _ in range(3):
                    link.send_servo(NEUTRAL_X_DEG, NEUTRAL_Y_DEG)
                    time.sleep(0.05)
            except Exception:
                pass
            link.close()

    print(f"종료 — 기록: {log_path}")
    print("분석하려면 이 파일을 알려주세요 (드리프트 방향/크기 정리해드림)")


if __name__ == "__main__":
    main()
