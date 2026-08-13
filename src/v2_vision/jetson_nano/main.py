"""통합 실행: ultrasound/test_sample.py 화면/조작 그대로 + 카메라(YOLO 검출)를 얹음.

- 터미널: test_sample.py와 동일한 curses 화면 (A/D/W/S로 서보 조작, 초음파 텔레메트리 표시)
  거기에 < Vision > 섹션(검출 개수 / YOLO FPS)이 추가됨
- HDMI 모니터: 검출 박스가 그려진 카메라 창 (--no-show로 끄면 터미널 화면만)

스레드 구조 (YOLO가 4Hz라서 키 반응/화면까지 4Hz로 묶이는 문제를 분리로 해결):
- UI 스레드(~33Hz): 키 입력 → 서보 명령 송신, 시리얼 텔레메트리 수신, 화면 그리기
  → test_sample.py 단독 실행 때와 같은 반응속도. 시리얼/ curses는 이 스레드만 만짐.
- 메인 스레드: 카메라 프레임 읽기 + YOLO 추론 + (옵션) cv2 카메라 창
  → 검출 개수/FPS만 shared로 UI 스레드에 넘김.

camera/live_camera.py의 카메라 파이프라인 + ultrasound/sample.ino의 시리얼 프로토콜을
그대로 가져와서 한 파일에 인라인으로 합쳤음 (로직이 크지 않아 이 편이 더 명확함).
카메라/초음파 값을 엮어서 서보를 자동으로 움직이는 판단 로직은 아직 없음.
"""
import argparse
import curses
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import serial

# TensorRT 8.0 파이썬 바인딩이 구식 np.bool을 참조 (numpy 1.24+에서 제거됨).
# .engine 모델을 로드하려면 ultralytics가 tensorrt를 import하기 전에 별칭 복원 필요.
if not hasattr(np, "bool"):
    np.bool = bool

from ultralytics import YOLO


# ======================================================
# Config (test_sample.py와 동일)
# ======================================================

SERIAL_BAUD = 115200

SERVO_STEP_DEG = 0.5

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

INITIAL_SERVO_X_DEG = 90.0
INITIAL_SERVO_Y_DEG = 90.0

# UI 스레드 주기 (키 입력 체크)
LOOP_DT = 0.03  # 약 33 Hz

# 화면 갱신 주기 — 33Hz로 전체 화면을 다시 그리면 CPU를 꽤 먹어서(YOLO 처리율까지
# 깎아먹는 것으로 실측됨) 키 폴링과 분리해 10Hz로만 갱신
DRAW_INTERVAL = 0.1

# 같은 키가 눌려있을 때 너무 빠르게 변하지 않도록 제한
COMMAND_INTERVAL = 0.05


# ======================================================
# 카메라 (camera/live_camera.py에서 그대로 가져옴)
# ======================================================

def usb_pipeline(index=0, width=1280, height=720, fps=30):
    return (
        f"v4l2src device=/dev/video{index} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def csi_pipeline(width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


# ======================================================
# ESP32 (ultrasound/sample.ino, test_sample.py와 같은 프로토콜)
# 송신 {"cmd":"servo","x":..,"y":..}\n / 수신 {"type":"telemetry", ...}\n
# ======================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def send_servo_command(ser, x_deg, y_deg):
    msg = json.dumps({"cmd": "servo", "x": round(x_deg, 1), "y": round(y_deg, 1)},
                      separators=(",", ":")) + "\n"
    ser.write(msg.encode("utf-8"))


def read_telemetry_nonblocking(ser, tel):
    """ser.in_waiting에 쌓인 줄을 전부 읽어서 tel dict를 최신 상태로 갱신."""
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
            tel["_last_raw"] = f"JSON parse error: {raw}"
            continue

        if not isinstance(data, dict):
            tel["_last_raw"] = f"ignored non-object JSON: {raw}"
            continue
        if data.get("type") != "telemetry":
            tel["_last_raw"] = raw
            continue

        tel.update(data)
        tel["_last_raw"] = raw
        tel["_last_rx"] = time.time()


# ======================================================
# 스레드 간 공유 상태
# ======================================================

class Shared:
    """메인(YOLO) ↔ UI 스레드가 주고받는 최소한의 상태.

    n_detections/fps는 단순 스칼라 대입이라 GIL 하에서 별도 락 없이 안전.
    종료는 stop 이벤트 하나로 양방향 전파 (먼저 사유를 적은 쪽이 이김).
    """

    def __init__(self):
        self.stop = threading.Event()
        self.stop_reason = None
        self.n_detections = 0
        self.fps = 0.0

    def request_stop(self, reason):
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()


# ======================================================
# 터미널 화면 (test_sample.py의 draw_screen과 동일 + Vision 섹션)
# ======================================================

def fmt(value, unit="", width=8) -> str:
    if value is None:
        return f"{'---':>{width}}{unit}"
    try:
        return f"{float(value):>{width}.1f}{unit}"
    except (TypeError, ValueError):
        return f"{'ERR':>{width}}{unit}"


def safe_addstr(stdscr, y, x, text) -> None:
    max_y, max_x = stdscr.getmaxyx()
    if y < 0 or y >= max_y or x < 0 or x >= max_x:
        return
    try:
        stdscr.addstr(y, x, text[: max_x - x])
    except curses.error:
        # 커서가 화면 맨 끝(우하단)에 닿으면 ncurses가 ERR을 반환하는 경우가 있어 무시
        pass


def draw_screen(stdscr, tel, target_x, target_y, shared, args):
    stdscr.erase()

    port = "미사용" if args.no_servo else args.serial_port

    safe_addstr(stdscr, 0, 0, "Jetson Nano 통합 실행 (Camera + Ultrasonic + Servo)")
    safe_addstr(stdscr, 1, 0, "Keys: A/D = Servo X, W/S = Servo Y, Q = quit")
    safe_addstr(stdscr, 2, 0,
                f"Port: {port} | Baud: {SERIAL_BAUD} | Step: {SERVO_STEP_DEG:.1f} deg")

    safe_addstr(stdscr, 4, 0, "< Vision >")
    safe_addstr(stdscr, 5, 0,
                f"Camera: /dev/video{args.cam} ({args.cam_width}x{args.cam_height})")
    safe_addstr(stdscr, 6, 0,
                f"검출: {shared.n_detections}개 | YOLO {shared.fps:.1f} FPS")

    safe_addstr(stdscr, 8, 0, "< Command Target >")
    safe_addstr(stdscr, 9, 0, f"Servo X target: {target_x:8.1f} deg")
    safe_addstr(stdscr, 10, 0, f"Servo Y target: {target_y:8.1f} deg")

    safe_addstr(stdscr, 12, 0, "< ESP32 Servo State >")
    safe_addstr(stdscr, 13, 0, f"Servo X current: {fmt(tel.get('servo_x'), ' deg')}")
    safe_addstr(stdscr, 14, 0, f"Servo Y current: {fmt(tel.get('servo_y'), ' deg')}")

    safe_addstr(stdscr, 16, 0, "< Ultrasonic >")
    safe_addstr(stdscr, 17, 0, f"Front distance: {fmt(tel.get('front'), ' cm')}")

    safe_addstr(stdscr, 19, 0, "< Left Side >")
    safe_addstr(stdscr, 20, 0, f"L_FRONT: {fmt(tel.get('left_front'), ' cm')}")
    safe_addstr(stdscr, 21, 0, f"L_REAR : {fmt(tel.get('left_rear'), ' cm')}")
    safe_addstr(stdscr, 22, 0, f"L_DIST : {fmt(tel.get('left_dist'), ' cm')}")
    safe_addstr(stdscr, 23, 0, f"L_ANGLE: {fmt(tel.get('left_angle'), ' deg')}")

    safe_addstr(stdscr, 25, 0, "< Right Side >")
    safe_addstr(stdscr, 26, 0, f"R_FRONT: {fmt(tel.get('right_front'), ' cm')}")
    safe_addstr(stdscr, 27, 0, f"R_REAR : {fmt(tel.get('right_rear'), ' cm')}")
    safe_addstr(stdscr, 28, 0, f"R_DIST : {fmt(tel.get('right_dist'), ' cm')}")
    safe_addstr(stdscr, 29, 0, f"R_ANGLE: {fmt(tel.get('right_angle'), ' deg')}")

    last_rx = tel.get("_last_rx")
    age = time.time() - last_rx if last_rx else None
    safe_addstr(stdscr, 31, 0, f"Last ESP32 RX: {fmt(age, ' s')} ago")

    safe_addstr(stdscr, 33, 0, "< Last raw line >")
    safe_addstr(stdscr, 34, 0, str(tel.get("_last_raw", ""))[:110])

    stdscr.refresh()


# ======================================================
# UI 스레드: 키 입력 + 시리얼 + 화면 (~33Hz, test_sample.py와 동일한 반응속도)
# ======================================================

def ui_loop(stdscr, ser, shared, args):
    tel = {}
    target_x = INITIAL_SERVO_X_DEG
    target_y = INITIAL_SERVO_Y_DEG
    last_cmd_time = 0.0
    last_draw = 0.0

    while not shared.stop.is_set():
        now = time.time()

        # 1. 키보드 입력 → 서보 목표각 (test_sample.py와 동일)
        key = stdscr.getch()
        changed = False

        if key != -1 and now - last_cmd_time >= COMMAND_INTERVAL:
            if key in (ord("q"), ord("Q")):
                shared.request_stop("사용자 종료 (q)")
                break
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

            if changed and ser is not None:
                target_x = clamp(target_x, SERVO_MIN_DEG, SERVO_MAX_DEG)
                target_y = clamp(target_y, SERVO_MIN_DEG, SERVO_MAX_DEG)
                try:
                    send_servo_command(ser, target_x, target_y)
                except (serial.SerialException, OSError) as e:
                    shared.request_stop(f"ESP32 시리얼 연결 끊김(송신): {e}")
                    break
                last_cmd_time = now

        # 2. 시리얼 수신
        if ser is not None:
            try:
                read_telemetry_nonblocking(ser, tel)
            except (serial.SerialException, OSError) as e:
                shared.request_stop(f"ESP32 시리얼 연결 끊김(수신): {e}")
                break

        # 3. 화면 그리기 (키 폴링보다 낮은 10Hz — 33Hz 전체 갱신은 CPU 낭비)
        if now - last_draw >= DRAW_INTERVAL:
            draw_screen(stdscr, tel, target_x, target_y, shared, args)
            last_draw = now

        time.sleep(LOOP_DT)


# ======================================================
# 메인 스레드: 카메라 + YOLO
# ======================================================

def run(stdscr, cap, model, ser, args):
    """종료 사유를 반환 (curses가 화면을 되돌린 뒤에 출력하려고)."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    shared = Shared()

    ui = threading.Thread(target=ui_loop, args=(stdscr, ser, shared, args), daemon=True)
    ui.start()

    loop_times = []
    t_prev = time.time()

    while not shared.stop.is_set():
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok = False
        if not ok:
            shared.request_stop("프레임 읽기 실패 — 카메라 연결 확인 (dmesg에 USB 관련 로그 있는지)")
            break

        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                             iou=0.5, verbose=False)[0]

        now = time.time()
        loop_times.append(now - t_prev)
        t_prev = now
        recent = loop_times[-30:]          # 최근 30프레임 이동 평균
        shared.fps = len(recent) / sum(recent) if sum(recent) > 0 else 0.0
        shared.n_detections = len(res.boxes)

        if not args.no_show:
            cv2.imshow("main", res.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                shared.request_stop("사용자 종료 (q, 카메라 창)")
                break

    ui.join(timeout=2)
    return shared.stop_reason


def main():
    ap = argparse.ArgumentParser(description="카메라+초음파/서보 통합 실행")
    ap.add_argument("model")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    # 30fps로 받으면 GStreamer 디코딩 스레드가 코어 하나를 통째로 먹으면서(실측 91%)
    # YOLO에 갈 CPU를 뺏음 — 추론이 어차피 초당 십수 장이라 15fps로 충분 (C920 지원값)
    ap.add_argument("--cam-fps", type=int, default=15,
                     help="카메라 프레임레이트 (C920 지원: 5/10/15/20/25/30)")
    ap.add_argument("--no-show", action="store_true", help="카메라 창 없이 터미널 화면만")
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--serial-baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--no-servo", action="store_true", help="ESP32 없이 카메라만 테스트")
    args = ap.parse_args()

    if not args.no_show:
        os.environ.setdefault("DISPLAY", ":0")
        os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

    model = YOLO(args.model)

    if args.csi:
        cap = cv2.VideoCapture(
            csi_pipeline(args.cam_width, args.cam_height, args.cam_fps), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(
            usb_pipeline(args.cam, args.cam_width, args.cam_height, args.cam_fps),
            cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("카메라를 열 수 없음 (--cam 인덱스 또는 --csi 여부 확인)")

    ser = None
    if not args.no_servo:
        try:
            ser = serial.Serial(args.serial_port, args.serial_baud, timeout=0)
            time.sleep(2.0)  # 포트 열 때 ESP32가 리셋되므로 부팅 대기
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            send_servo_command(ser, INITIAL_SERVO_X_DEG, INITIAL_SERVO_Y_DEG)
            print(f"ESP32 연결됨 ({args.serial_port})")
        except serial.SerialException as e:
            print(f"ESP32 연결 실패, 초음파/서보 없이 진행: {e}")
            ser = None

    # 젯슨 첫 추론은 CUDA 커널 준비로 수십 초 걸림. curses 화면으로 들어가기 전에
    # 미리 소진해야 사용자가 빈 화면만 보며 기다리는 상황을 피할 수 있음.
    print("모델 워밍업 중... (젯슨 첫 추론 특성상 최대 1분 정도 걸릴 수 있음)")
    t0 = time.time()
    ok, frame = cap.read()
    if ok:
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, iou=0.5, verbose=False)
    print(f"워밍업 완료 ({time.time() - t0:.1f}s) — 화면 전환")

    reason = None
    try:
        reason = curses.wrapper(run, cap, model, ser, args)
    except KeyboardInterrupt:
        reason = "Ctrl-C 종료"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()

    # curses가 화면을 원래대로 돌려놓은 뒤에 출력해야 사용자가 볼 수 있음
    if reason:
        print(reason)


if __name__ == "__main__":
    main()
