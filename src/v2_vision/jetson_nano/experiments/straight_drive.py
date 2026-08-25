"""전/후진 직진 유지 실험 도구 — IMU(yaw) 피드백 보정 내장. 카메라/YOLO 불필요.

배경: 캐스터(앞바퀴)가 어디로 틀어져 있는지에 따라 전/후진이 매번 다르게 휨
(실측: 전진 -0.9~+1.7°/s, 후진 +2.1°/s — 방향도 상황마다 바뀜). 회전 권한도
상황마다 달라서 상수로 예측 불가 → "정확한 상수"가 필요 없는 피드백 방식 사용.

제어 구조 (3중):
  1) 방위 오차 → 목표 회전속도 (CORR_TAU초에 걸쳐 되돌리는 완만한 값)
  2) 목표 회전속도 vs 자이로 실측 → 서보 편향. 자이로가 브레이크 역할을 해서
     목표에 다가갈수록 스스로 감속한다 (각도만 보던 예전 방식이 반대편으로
     넘어가 좌우 진동하던 원인이 이것).
  3) 적분항이 '지금 이 캐스터를 이기는 데 필요한 편향'을 주행 중 학습해서 물고
     있는다. 덕분에 틀어짐 패턴을 몰라도, 매번 달라져도 수렴한다.

키:
  W = 전진   S = 후진   A/D = 좌/우회전(보정 없음)   SPACE = 정지   Q = 종료
  I = 기준 방위각 고정/해제
      - 안 누르면: W/S를 누른 그 순간의 방위각을 그 구간 동안 유지
      - 누르면:   그때의 방위각을 못박아, A/D로 돌린 뒤에 W를 눌러도
                  무조건 그 방위각으로 되돌아온다 (해제 전까지)
  보정 ON/부호/yaw 켜기는 전부 자동이라 따로 누를 키가 없다.

안전:
  - 전진 중 전방 초음파 30cm 미만 → 자동 정지 (센서 전원 꺼져 있으면 무력!)
  - 어떤 경로로 끝나든 조이스틱 중립 복귀

기록: experiments/logs/drive_<시각>.csv (10Hz, 1초마다 flush)
"""
import csv
import curses
import math
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

# ---- 직진 보정 파라미터 (2026-08-25 실주행 로그 재튜닝) ----
# 부호 규약(실측): 서보 X + (좌) → yaw 증가 → gyro_z 양수. 후진은 반응이 반대.
#
# 왜 각도 P제어를 버렸나 (drive_20260825_104159.csv):
#   t=19.1 방위오차 +1.2°(거의 복구)인데 자이로 -17.7°/s — 도착 순간에도 브레이크가
#   없어서 반대쪽으로 넘어가고, 그러면 또 반대로 꺾어서 좌우 진동이 났다.
#   → 각도로 "목표 회전속도"를 정하고 자이로로 그 속도를 맞추는 2단(캐스케이드) 구조.
#     오차가 줄면 목표 속도도 같이 줄어드니 스스로 감속한다.
CORR_TAU = 1.5            # 방위 오차를 이 시간(초)에 걸쳐 되돌린다 (작을수록 공격적)
# 목표 회전속도 상한 °/s. 평소 주행에선 오차가 작아 3~5°/s밖에 안 쓰고, I로 고정해둔
# 방위각에서 크게 벗어난 뒤(A/D 회전 후) 되돌아올 때만 여기까지 쓴다.
# 시뮬: 50° 복귀에 5→11초, 10→7초, 16이면 오버슈트 발생 → 10이 안전한 최대치.
CORR_RATE_MAX = 10.0
CORR_KR = 0.4             # 회전속도 오차 1°/s당 서보 X (실측: 데드존 위 1° ≈ 3°/s)
# 적분항: 캐스터가 만드는 '일정한 틀어짐'을 주행 중 스스로 학습해서 물고 있는다.
# 이게 없으면 회전만 멈추고 6° 틀어진 채로 평행 주행함(시뮬 확인).
CORR_KI = 0.8
# 실측 데드존: 편향 4.5°는 0.9초간 완전 무반응(yaw 변화 0), 13°에서 -18°/s.
# 작은 보정이 아예 안 먹으니 최소 편향을 얹어줘야 선형처럼 동작한다.
# 단, 계단식으로 얹으면 0을 지날 때 좌↔우로 튀므로 KNEE 구간에서 서서히 얹는다.
# (7까지 얹으면 권한이 센 순간에 좌우로 튐 — 시뮬 스윕 결과 4가 최적)
CORR_DEADZONE = 4.0
CORR_COMP_KNEE = 1.5
# 텔레메트리가 최대 1초씩 끊긴다. 그 사이 눈 감고 최대로 밀면 8°가 그냥 넘어가므로
# 새 값이 안 오면 보정을 놓는다 (실측 폭주의 나머지 절반이 이 구간에서 생겼다).
CORR_STALE_S = 0.45
CORR_FADE_S = 0.4
CORR_MAX_DEG = 13.0
CORR_DEADBAND_DEG = 1.0   # 이 안쪽이면 되돌리지 않고 '회전속도 0' 유지만 (떨림 방지)
CORR_ABORT_DEG = 30.0     # 이만큼 벌어지면 보정이 오히려 악화 중 → 중단하고 경고
REF_SETTLE_RATE = 4.0     # 기준 방위각은 회전이 이만큼 잦아든 뒤 잡는다 (°/s)
REF_MAX_WAIT = 1.2        # 안 잦아들면 이 시간 후 그냥 잡음 (초)

CSV_FIELDS = ["t", "state", "lock", "yaw0", "yaw", "yaw_err",
              "rate_tgt", "bias", "corr_x",
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


def get_rate(tel):
    """현재 회전 각속도 °/s (yaw 증가 방향이 +). 보정의 브레이크 역할."""
    try:
        return float(tel.get("gyro_z"))
    except (TypeError, ValueError):
        return None


def run(stdscr, link, writer, fp, log_path):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    ramp = ServoRamp(link, step_deg=2.5, keepalive=0.5)  # 83°/s — 브라운아웃/삐소리 나면 2.0으로
    state = "STOP"
    note = ""
    yaw0 = None               # 지금 맞추려는 기준 방위각
    ref_pending = False       # 기준 방위각 대기중 (묵은 값 대신 새 값으로 잡으려고)
    ref_since = 0.0
    lock_mode = False         # I로 고정한 상태인가 (해제 전까지 이 방위각만 고수)
    yaw_lock = None
    lock_pending = False
    err_start = None          # 구간 시작 시 오차 (여기서 더 벌어지면 이상 신호)
    bias = 0.0                # 주행 중 학습한 '틀어짐 상쇄' 편향
    prev_now = time.monotonic()
    last_yaw_val = None
    last_fresh_t = time.monotonic()
    n_samples = 0
    t0 = time.monotonic()
    last_log = 0.0
    last_draw = 0.0

    while True:
        now = time.monotonic()
        dt = min(now - prev_now, 0.2)   # 튀는 루프 주기가 적분을 흔들지 않게 상한
        prev_now = now
        tel = link.poll()
        yaw = get_yaw(tel)
        gz = get_rate(tel)
        # 텔레메트리가 3~4Hz(최대 1초 지연)라 같은 값이 계속 온다.
        # '방금 들어온 새 값'인지 구분해야 기준 방위각을 묵은 값으로 잡지 않는다.
        yaw_fresh = yaw is not None and yaw != last_yaw_val
        if yaw_fresh:
            last_yaw_val, last_fresh_t = yaw, now

        # 1. 키 입력
        key = stdscr.getch()
        if key != -1:
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("w"), ord("W"), ord("s"), ord("S")):
                want_state = "MOVE FORWARD" if key in (ord("w"), ord("W")) else "MOVE BACKWARD"
                if state != want_state:     # 연타로 기준 방위각이 풀리지 않게
                    bias, err_start = 0.0, None
                    if lock_mode and yaw_lock is not None:
                        yaw0, ref_pending = yaw_lock, False   # 고정된 방위각으로 복귀
                    else:
                        yaw0, ref_pending, ref_since = None, True, now
                state, note = want_state, ""
            elif key in (ord("a"), ord("A"), ord("d"), ord("D")):
                # 의도된 회전 — 보정 없음. 고정해둔 방위각은 그대로 살려둔다.
                state = "TURN LEFT" if key in (ord("a"), ord("A")) else "TURN RIGHT"
                note, yaw0, ref_pending, bias, err_start = "", None, False, 0.0, None
            elif key == ord(" "):
                state, note = "STOP", ""
                yaw0, ref_pending, bias, err_start = None, False, 0.0, None
            elif key in (ord("i"), ord("I")):
                # 기준 방위각 고정/해제. 고정하면 A/D로 돌린 뒤에도 W/S가
                # 매번 새로 잡지 않고 무조건 이 방위각으로 되돌아온다.
                if lock_mode:
                    lock_mode, yaw_lock, lock_pending = False, None, False
                    note = "기준 고정 해제 — 이제 W/S 누른 방향을 유지"
                else:
                    lock_mode, lock_pending = True, True
                    ref_pending, ref_since = True, now
                    note = "기준 방위각 고정 중..."

        # 2. 전방 자동 정지
        try:
            front_v = float(tel.get("front"))
        except (TypeError, ValueError):
            front_v = -1.0
        if state == "MOVE FORWARD" and 0.0 < front_v < FRONT_STOP_CM:
            state, yaw0, ref_pending = "STOP", None, False
            bias, err_start = 0.0, None
            note = f"!! 전방 {front_v:.0f}cm — 자동 정지"

        # 3. 서보 목표 = 프리셋 + (이동 중이면) yaw 보정
        base_x, base_y = servo_preset(state)
        moving = state in ("MOVE FORWARD", "MOVE BACKWARD")

        # 3a. 기준 방위각 잡기 — 키 누른 순간의 값은 최대 1초 묵은 값이라 그대로 쓰면
        #     시작하자마자 유령 오차가 생긴다(실측 +1.8°). 새 샘플이 오고 직전 회전이
        #     잦아든 뒤에 잡는다.
        if ref_pending and yaw_fresh:
            if (gz is None or abs(gz) < REF_SETTLE_RATE) or (now - ref_since) > REF_MAX_WAIT:
                yaw0, ref_pending = yaw, False
                if lock_pending:            # I로 고정 요청한 건 여기서 확정
                    yaw_lock, lock_pending = yaw, False
                    note = f"기준 방위각 {yaw:.1f}° 고정 — 앞으로 여기만 맞춥니다"

        # 3b. 캐스케이드 보정: 방위 오차 → 목표 회전속도 → 자이로로 그 속도를 맞춤.
        #     오차가 줄면 목표 속도도 같이 줄어서 스스로 감속한다(진동의 원인이던 부분).
        yaw_err = None
        rate_tgt = None
        corr_x = 0.0
        if moving and yaw is not None and yaw0 is not None:
            yaw_err = yaw - yaw0
            if err_start is None:
                err_start = abs(yaw_err)
            # 고정(I) 상태에선 A/D로 돌린 뒤라 시작부터 크게 벌어져 있는 게 정상이다.
            # 그래서 절대값이 아니라 '시작보다 더 벌어졌는가'로 이상을 판단한다.
            abort_at = max(CORR_ABORT_DEG, err_start + CORR_ABORT_DEG)
            if abs(yaw_err) > abort_at:
                bias = 0.0
                note = f"!! 방위 {yaw_err:+.0f}° 이탈 — 보정 중단, SPACE 후 다시"
            else:
                # 데드밴드 안이면 '되돌리기'는 쉬고 회전속도만 0으로 눌러둔다
                want = 0.0 if abs(yaw_err) <= CORR_DEADBAND_DEG else -yaw_err / CORR_TAU
                rate_tgt = clamp(want, -CORR_RATE_MAX, CORR_RATE_MAX)
                rate_err = rate_tgt - (gz or 0.0)
                # 새 측정값이 있을 때만 적분 (끊긴 동안 쌓이면 폭주)
                if now - last_fresh_t <= CORR_STALE_S:
                    bias = clamp(bias + CORR_KI * rate_err * dt,
                                 -CORR_MAX_DEG, CORR_MAX_DEG)
                u = CORR_KR * rate_err + bias
                # 데드존 보상: 조이스틱이 4.5°까진 안 먹으니 최소 편향을 얹되,
                # 0 근처에선 서서히 얹어야 좌우로 튀지 않는다
                comp = CORR_DEADZONE * min(abs(u) / CORR_COMP_KNEE, 1.0)
                corr_x = clamp(u + math.copysign(comp, u), -CORR_MAX_DEG, CORR_MAX_DEG)
                # 새 측정값이 안 오는 동안엔 힘을 뺀다 (눈 감고 미는 구간 제거)
                age = now - last_fresh_t
                if age > CORR_STALE_S:
                    corr_x *= clamp((CORR_STALE_S + CORR_FADE_S - age) / CORR_FADE_S, 0.0, 1.0)
                    if corr_x == 0.0:
                        note = "텔레메트리 끊김 — 보정 대기"
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
                "lock": int(lock_mode),
                "yaw0": yaw0, "yaw": tel.get("yaw"),
                "yaw_err": (round(yaw_err, 2) if yaw_err is not None else None),
                "rate_tgt": (round(rate_tgt, 2) if rate_tgt is not None else None),
                "bias": round(bias, 2),
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
            put(stdscr, 0, "W 전진   S 후진   A/D 좌우회전   SPACE 정지"
                           "   I 기준방위각 고정/해제   Q 종료")
            put(stdscr, 2, f"상태: {state}   {note}")
            corr_txt = (f"기준: {'★ 고정 ' + fmt(yaw_lock, 1).strip() + '°' if lock_mode else 'W/S 누른 방향'}")
            if ref_pending:
                corr_txt += "   기준 방위각 잡는 중..."
            elif yaw_err is not None:
                corr_txt += (f"   오차 {yaw_err:+5.1f}°  회전속도 목표"
                             f" {fmt(rate_tgt, 5)} / 실제 {fmt(gz, 5)} °/s"
                             f"  → 보정 {corr_x:+5.1f}° (학습 {bias:+.1f}°)")
            put(stdscr, 3, corr_txt)
            put(stdscr, 5, f"서보 명령 x={ramp.command_x:5.1f} y={ramp.command_y:5.1f}"
                            f"   에코 x={fmt(tel.get('servo_x'), 5)} y={fmt(tel.get('servo_y'), 5)}")
            yaw_state = ("측정중" if tel.get("yaw_active") else
                         "캘리브레이션중" if tel.get("yaw_calibrating") else "!! 꺼짐")
            put(stdscr, 7, f"IMU yaw: {fmt(tel.get('yaw'))}°  [{yaw_state}]"
                            f"   맞추는 방위각: {fmt(yaw0)}")
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

    # yaw 측정 자동 시작 (이미 켜져 있으면 그대로 둠 — 토글이라 또 보내면 꺼짐)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        link.poll(); time.sleep(0.05)
    if not link.telemetry.get("yaw_active"):
        print("IMU yaw 캘리브레이션 시작 — 휠체어를 몇 초간 정지 상태로 두세요...")
        link.send_yaw_toggle()
        t0 = time.time()
        while time.time() - t0 < 12:
            link.poll()
            if link.telemetry.get("yaw_active"):
                break
            time.sleep(0.1)
    print("yaw:", "측정 시작됨 ✓" if link.telemetry.get("yaw_active")
          else "시작 실패 — 화면에서 E로 수동 시도")

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
