"""전/후진 직진 유지 실험 도구 — IMU(yaw) 피드백 보정 내장. 카메라/YOLO 불필요.

배경: 캐스터(앞바퀴)가 어디로 틀어져 있는지에 따라 전/후진이 매번 다르게 휨
(실측: 전진 -0.9~+1.7°/s, 후진 +2.1°/s — 방향도 상황마다 바뀜). 회전 권한도
상황마다 달라서 상수로 예측 불가 → 그래서 "정확한 상수"가 필요 없는
피드백 방식 사용: 이동 시작 순간의 yaw를 0점 잡고, 벗어난 만큼 서보 X를
반대로 미세 편향. 오차가 줄 때까지 계속 보정하므로 권한이 변해도 수렴한다.

키:
  W = 전진   S = 후진   A/D = 좌/우회전(보정 없음)   SPACE = 정지
  C = 직진 보정 ON/OFF (기본 OFF — 먼저 OFF로 휘는 것 확인 후 ON과 비교)
  I = 보정 방향 반전 (보정 켰는데 더 휘면 = 부호 반대 → I 한 번)
  E = yaw 측정 토글 (꺼져 있으면 정지 상태에서 켤 것 — 캘리브레이션 몇 초)
  Q = 종료 (중립 복귀 후 종료)

안전:
  - 전진 중 전방 초음파 30cm 미만 → 자동 정지 (센서 전원 꺼져 있으면 무력!)
  - 어떤 경로로 끝나든 조이스틱 중립 복귀

기록: experiments/logs/drive_<시각>.csv (10Hz, 1초마다 flush)
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
                            clamp, servo_preset)

LOOP_DT = 0.03            # 키 폴링 ~33Hz
LOG_INTERVAL = 0.1        # CSV 기록 10Hz
DRAW_INTERVAL = 0.1
FRONT_STOP_CM = 30.0      # 전진 중 이 거리 미만이면 자동 정지
YAW_TOGGLE_COOLDOWN = 0.5

# ---- 직진 보정 파라미터 (2026-08-25 실측 기반) ----
# 실측 부호 규약: 좌회전(X+) 명령 → yaw 증가. 따라서 +오차엔 -X(우회전) 보정.
CORR_SIGN_DEFAULT = -1.0
CORR_KP = 2.5             # yaw 오차 1도당 서보 X 보정 각도
# 실측 드리프트가 최대 ~8°/s인데 우회전 풀 권한이 3.6°/s뿐 —
# 보정 상한을 풀 회전 수준까지 열어야 싸움이 됨 (6이었을 땐 역부족)
CORR_MAX_DEG = 13.0
CORR_DEADBAND_DEG = 0.5   # 이 안쪽 오차는 무시 (미세 떨림 방지)

CSV_FIELDS = ["t", "state", "corr_on", "corr_sign", "yaw0", "yaw", "yaw_err", "corr_x",
              "front", "left_front", "left_rear", "right_front", "right_rear",
              "gyro_z", "cmd_x", "cmd_y", "echo_x", "echo_y"]


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


def get_yaw(tel):
    if not tel.get("yaw_active"):
        return None
    try:
        return float(tel.get("yaw"))
    except (TypeError, ValueError):
        return None


def run(stdscr, link, writer, fp, log_path):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    ramp = ServoRamp(link, step_deg=2.5, keepalive=0.5)  # 83°/s — 브라운아웃/삐소리 나면 2.0으로
    state = "STOP"
    note = ""
    corr_on = False
    corr_sign = CORR_SIGN_DEFAULT
    yaw0 = None               # 이동 시작 순간의 yaw (상대 0점)
    n_samples = 0
    t0 = time.monotonic()
    last_log = 0.0
    last_draw = 0.0
    last_yaw_toggle = 0.0

    while True:
        now = time.monotonic()
        tel = link.poll()
        yaw = get_yaw(tel)

        # 1. 키 입력
        key = stdscr.getch()
        if key != -1:
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("w"), ord("W")):
                if state != "MOVE FORWARD":     # 연타로 기준 방위각이 풀리지 않게
                    yaw0 = yaw
                state, note = "MOVE FORWARD", ""
            elif key in (ord("s"), ord("S")):
                if state != "MOVE BACKWARD":
                    yaw0 = yaw
                state, note = "MOVE BACKWARD", ""
            elif key in (ord("a"), ord("A")):
                state, note, yaw0 = "TURN LEFT", "", None   # 의도된 회전 — 보정 없음
            elif key in (ord("d"), ord("D")):
                state, note, yaw0 = "TURN RIGHT", "", None
            elif key == ord(" "):
                state, note, yaw0 = "STOP", "", None
            elif key in (ord("c"), ord("C")):
                corr_on = not corr_on
            elif key in (ord("i"), ord("I")):
                corr_sign = -corr_sign
            elif key in (ord("e"), ord("E")):
                if now - last_yaw_toggle > YAW_TOGGLE_COOLDOWN:
                    link.send_yaw_toggle()
                    last_yaw_toggle = now

        # 2. 전방 자동 정지
        try:
            front_v = float(tel.get("front"))
        except (TypeError, ValueError):
            front_v = -1.0
        if state == "MOVE FORWARD" and 0.0 < front_v < FRONT_STOP_CM:
            state, yaw0 = "STOP", None
            note = f"!! 전방 {front_v:.0f}cm — 자동 정지"

        # 3. 서보 목표 = 프리셋 + (이동 중이면) yaw 보정
        base_x, base_y = servo_preset(state)
        yaw_err = None
        corr_x = 0.0
        moving = state in ("MOVE FORWARD", "MOVE BACKWARD")
        if moving and corr_on and yaw is not None and yaw0 is not None:
            yaw_err = yaw - yaw0
            if abs(yaw_err) > CORR_DEADBAND_DEG:
                corr_x = clamp(CORR_KP * yaw_err, -CORR_MAX_DEG, CORR_MAX_DEG)
                corr_x *= corr_sign
                if state == "MOVE BACKWARD":
                    # 후진에선 조향 반응이 반대 (자동차 후진 핸들과 동일 원리).
                    # 실측: 전진은 이 부호로 0.5° 유지 성공, 후진은 +30° 폭주 → 반전 필요
                    corr_x = -corr_x
        ramp.set_target(clamp(base_x + corr_x), base_y)
        ramp.tick()

        # 4. CSV 기록 (10Hz)
        if now - last_log >= LOG_INTERVAL:
            writer.writerow({
                "t": round(now - t0, 2), "state": state,
                "corr_on": int(corr_on), "corr_sign": int(corr_sign),
                "yaw0": yaw0, "yaw": tel.get("yaw"),
                "yaw_err": (round(yaw_err, 2) if yaw_err is not None else None),
                "corr_x": round(corr_x, 1),
                "front": tel.get("front"),
                "left_front": tel.get("left_front"),
                "left_rear": tel.get("left_rear"),
                "right_front": tel.get("right_front"),
                "right_rear": tel.get("right_rear"),
                "gyro_z": tel.get("gyro_z"),
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
            put(stdscr, 0, "직진 유지 — W전진 S후진 A좌 D우 SPACE정지 | C보정 I부호 | E yaw Q종료")
            put(stdscr, 2, f"상태: {state}   {note}")
            corr_txt = f"보정: {'★ ON' if corr_on else 'OFF'} (부호 {'+' if corr_sign > 0 else '-'})"
            if yaw_err is not None:
                corr_txt += f"   yaw오차 {yaw_err:+.1f}° → 보정 {corr_x:+.1f}°"
            put(stdscr, 3, corr_txt)
            put(stdscr, 5, f"서보 명령 x={ramp.command_x:5.1f} y={ramp.command_y:5.1f}"
                            f"   에코 x={fmt(tel.get('servo_x'), 5)} y={fmt(tel.get('servo_y'), 5)}")
            yaw_state = ("측정중" if tel.get("yaw_active") else
                         "캘리브레이션중" if tel.get("yaw_calibrating") else "꺼짐(E로 시작!)")
            put(stdscr, 7, f"IMU yaw: {fmt(tel.get('yaw'))}°  [{yaw_state}]"
                            f"   이동시작 0점: {fmt(yaw0)}")
            put(stdscr, 9, f"전방(cm): {fmt(tel.get('front'))}  (자동정지 {FRONT_STOP_CM:.0f}cm"
                            f" — 센서 전원 꺼져있으면 무력!)")
            put(stdscr, 10, f"우측(cm): 앞 {fmt(tel.get('right_front'))}"
                             f"  뒤 {fmt(tel.get('right_rear'))}")
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
    print("보정 OFF 주행과 ON 주행을 비교 분석하려면 파일명을 알려주세요")


if __name__ == "__main__":
    main()
