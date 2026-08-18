"""통합 조종석: 카메라(YOLO) + 초음파 + 서보(조이스틱)를 한 프로세스·한 화면에서.

- 수동 모드(기본): W/A/S/D = 전진/좌회전/후진/우회전 프리셋, SPACE = 정지(중립).
  키를 누르면 목표 프리셋이 래치되고, 서보는 램프(ServoRamp)로 천천히 접근.
- 자동 모드: G 키 — guidance의 판단기(AlignmentPlanner)가 teach로 저장한 goal을
  향해 서보를 스스로 조작. **아무 이동키나 누르면 즉시 수동 복귀 + 정지** (비상 개입)
- E 키: MPU6050 yaw 측정 토글 (첫 토글: 정지 상태 캘리브레이션 → 0점 → 측정)
- 터미널(curses): 검출/FPS, 자동모드 지시·근거, IMU, 초음파 5개, 서보 에코 표시
- HDMI: 검출 박스 카메라 창 (--no-show로 끄기)

수동/자동 모두 hardware.esp32의 실측 조이스틱 프리셋을 쓴다 (중립=87/85,
90/90 아님). 지시 문구 규약은 guidance.common.servo_targets와 동일 —
이동 지시 4종만 편향, 나머지는 전부 중립=정지.

계층 구조: hardware/(카메라·ESP32 접근) 위에 이 파일(사람조종+조종석)과
guidance/(판단·goal 관리)가 올라가고, 자동모드는 guidance 판단기를 import해서 씀.

스레드: UI 스레드(~33Hz: 키/시리얼/화면) + 메인 스레드(카메라+YOLO+판단).
시리얼 송신은 UI 스레드만 수행 (자동모드에서도 메인은 목표각만 계산해서 넘김).
"""
import argparse
import curses
import os
import sys
import threading
import time

import cv2
import numpy as np
import serial

# TensorRT 8.0 파이썬 바인딩이 구식 np.bool을 참조 (numpy 1.24+에서 제거됨).
# .engine 모델을 로드하려면 ultralytics가 tensorrt를 import하기 전에 별칭 복원 필요.
# (hasattr 검사만으로도 numpy가 FutureWarning을 찍어서 잠시 꺼둠)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

from ultralytics import YOLO

from hardware.camera import Webcam, ensure_display, DEFAULT_FPS
from hardware.esp32 import (ESP32Link, ServoRamp, DEFAULT_PORT, DEFAULT_BAUD,
                            NEUTRAL_X_DEG, NEUTRAL_Y_DEG, servo_preset)
from guidance.common import (MAP_WIDTH, STABLE_FRAMES, AlignmentPlanner,
                             FeatureSmoother, anchor_from_result,
                             draw_top_view, load_goal, relative_estimate,
                             scaled_goal_feature)


# ======================================================
# Config
# ======================================================

LOOP_DT = 0.03            # UI 스레드 주기 (키 입력 체크, 약 33Hz)
DRAW_INTERVAL = 0.1       # 화면 갱신 10Hz (33Hz 전체 갱신은 CPU 낭비 — 실측)
SERVO_KEEPALIVE = 0.5     # 목표 도달 후에도 이 주기마다 마지막 명령 재전송
YAW_TOGGLE_INTERVAL = 0.3 # E 키 연타 방지

# 수동 모드 키 → 지시 문구 (guidance 판단기와 같은 규약 → 같은 프리셋 사용)
MANUAL_KEYS = {
    ord("w"): "MOVE FORWARD", ord("W"): "MOVE FORWARD",
    ord("a"): "TURN LEFT", ord("A"): "TURN LEFT",
    ord("s"): "MOVE BACKWARD", ord("S"): "MOVE BACKWARD",
    ord("d"): "TURN RIGHT", ord("D"): "TURN RIGHT",
    ord(" "): "STOP",
}


# ======================================================
# 스레드 간 공유 상태
# ======================================================

class Shared:
    """메인(YOLO+판단) ↔ UI 스레드가 주고받는 상태.

    스칼라/튜플/dict 참조 대입이라 GIL 하에서 별도 락 없이 안전.
    종료는 stop 이벤트 하나로 양방향 전파.
    """

    def __init__(self):
        self.stop = threading.Event()
        self.stop_reason = None
        self.n_detections = 0
        self.fps = 0.0
        # 자동모드
        self.mode = "MANUAL"          # 'MANUAL' | 'AUTO' (UI 스레드가 전환)
        self.auto_available = False   # goal 로드 성공 여부
        self.goal_msg = ""            # goal 상태 안내문
        self.plan = None              # 판단기 최신 출력 dict (표시용)
        self.auto_target = None       # 자동모드 서보 목표 (x, y)

    def request_stop(self, reason):
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()


# ======================================================
# 터미널 화면
# ======================================================

def fmt(value, unit="", width=6) -> str:
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
    # 시리얼 노이즈로 널 문자(\x00) 등 제어문자가 섞여 오면 addstr이
    # ValueError를 던지므로 표시 전에 걸러냄
    text = "".join(ch for ch in text if ch.isprintable() or ch == " ")
    try:
        stdscr.addstr(y, x, text[: max_x - x])
    except curses.error:
        # 커서가 화면 맨 끝(우하단)에 닿으면 ncurses가 ERR을 반환하는 경우가 있어 무시
        pass


def draw_screen(stdscr, tel, rx_age, manual_text, ramp, shared, args):
    stdscr.erase()

    port = "미사용" if args.no_servo else args.serial_port

    safe_addstr(stdscr, 0, 0, "Jetson Nano 통합 조종석 (Camera + Ultrasonic + Servo)")
    safe_addstr(stdscr, 1, 0,
                "Keys: W/A/S/D=전진/좌/후진/우  SPACE=정지  E=Yaw  G=자동모드  Q=종료")
    safe_addstr(stdscr, 2, 0,
                f"Camera: /dev/video{args.cam} ({args.cam_width}x{args.cam_height})"
                f" | ESP32: {port}")

    safe_addstr(stdscr, 4, 0, "< Vision >")
    safe_addstr(stdscr, 5, 0,
                f"검출: {shared.n_detections}개 | YOLO {shared.fps:.1f} FPS")

    # 자동모드 섹션
    safe_addstr(stdscr, 7, 0, "< Auto >")
    if shared.mode == "AUTO":
        mode_line = "모드: ★ AUTO — 아무 이동키나 누르면 즉시 수동 복귀+정지"
    elif shared.auto_available:
        mode_line = "모드: MANUAL (G 키로 자동 시작)"
    else:
        mode_line = f"모드: MANUAL — {shared.goal_msg}"
    safe_addstr(stdscr, 8, 0, mode_line)

    plan = shared.plan
    if plan is not None:
        safe_addstr(stdscr, 9, 0,
                    f"지시: {plan['text']}  (phase={plan['phase']} "
                    f"stable={plan['stable']}/{STABLE_FRAMES})")
        detail = (f"du={fmt(plan['du'])} scale={fmt(plan['scale'])} "
                  f"sonarErr={fmt(plan['sonar_err'], 'm')}")
        if shared.auto_target is not None:
            detail += (f" | 프리셋 x={shared.auto_target[0]:.0f}"
                       f" y={shared.auto_target[1]:.0f}")
        safe_addstr(stdscr, 10, 0, detail)
    else:
        safe_addstr(stdscr, 9, 0, f"지시: (수동) {manual_text}")

    safe_addstr(stdscr, 12, 0, "< Servo (조이스틱 프리셋 + 램프) >")
    if ramp is not None:
        servo_line = (f"목표: X {ramp.target_x:6.1f}  Y {ramp.target_y:6.1f}"
                      f"   |   램프 전송: X {ramp.command_x:6.1f}"
                      f"  Y {ramp.command_y:6.1f}")
    else:
        servo_line = "목표: (서보 미사용)"
    safe_addstr(stdscr, 13, 0, servo_line)
    safe_addstr(stdscr, 14, 0,
                f"에코(펌웨어 적용값): X {fmt(tel.get('servo_x'))}"
                f"  Y {fmt(tel.get('servo_y'))}")

    safe_addstr(stdscr, 16, 0, "< IMU (MPU6050) >")
    imu = tel.get("imu_ready")
    imu_status = "---" if imu is None else ("READY" if imu else "NOT READY")
    if tel.get("yaw_calibrating"):
        yaw_state = "CALIBRATING — 휠체어 정지 유지"
    elif tel.get("yaw_active"):
        yaw_state = "MEASURING"
    else:
        yaw_state = "STOPPED"
    safe_addstr(stdscr, 17, 0, f"상태: {imu_status} | Yaw: {yaw_state}")
    safe_addstr(stdscr, 18, 0,
                f"yaw {fmt(tel.get('yaw'), ' deg')}"
                f"  gyroZ {fmt(tel.get('gyro_z'), ' dps')}"
                f"  bias {fmt(tel.get('gyro_z_bias'), ' dps')}")

    safe_addstr(stdscr, 20, 0, "< Ultrasonic (cm) >")
    safe_addstr(stdscr, 21, 0, f"Front: {fmt(tel.get('front'))}")
    safe_addstr(stdscr, 22, 0,
                f"L  f {fmt(tel.get('left_front'))}  r {fmt(tel.get('left_rear'))}"
                f"  d {fmt(tel.get('left_dist'))}  a {fmt(tel.get('left_angle'))}")
    safe_addstr(stdscr, 23, 0,
                f"R  f {fmt(tel.get('right_front'))}  r {fmt(tel.get('right_rear'))}"
                f"  d {fmt(tel.get('right_dist'))}  a {fmt(tel.get('right_angle'))}")

    safe_addstr(stdscr, 25, 0, f"Last ESP32 RX: {fmt(rx_age, ' s')} ago")
    safe_addstr(stdscr, 26, 0, "raw: " + str(tel.get("_last_raw", ""))[:100])

    stdscr.refresh()


# ======================================================
# UI 스레드: 키 입력 + 시리얼 + 화면 (~33Hz)
# ======================================================

def ui_loop(stdscr, link, ramp, shared, args):
    # 이 스레드에서 예외가 새면 화면만 조용히 죽고 YOLO는 계속 도는 좀비가
    # 되므로, 무슨 예외든 전체 종료로 연결한다
    try:
        _ui_loop(stdscr, link, ramp, shared, args)
    except Exception as e:
        shared.request_stop(f"UI 스레드 오류로 종료: {type(e).__name__}: {e}")


def _ui_loop(stdscr, link, ramp, shared, args):
    manual_text = "STOP"        # 수동 목표 지시 문구 (키로 래치)
    last_draw = 0.0
    last_yaw_toggle = 0.0

    def set_ramp_target(text):
        if ramp is not None:
            ramp.set_target(*servo_preset(text, args.invert_x, args.invert_y))

    def exit_auto():
        """자동 → 수동 복귀. 안전을 위해 무조건 중립(정지)으로 램프."""
        nonlocal manual_text
        shared.mode = "MANUAL"
        shared.auto_target = None
        manual_text = "STOP"
        set_ramp_target("STOP")

    while not shared.stop.is_set():
        now = time.time()

        # 1. 키보드 입력
        key = stdscr.getch()

        if key != -1:
            if key in (ord("q"), ord("Q")):
                shared.request_stop("사용자 종료 (q)")
                break
            elif key in (ord("g"), ord("G")):
                if shared.mode == "AUTO":
                    exit_auto()
                elif shared.auto_available:
                    # 자동 진입은 정지 상태에서 시작 (판단기 리셋은 메인 스레드가 함)
                    manual_text = "STOP"
                    set_ramp_target("STOP")
                    shared.mode = "AUTO"
            elif key in (ord("e"), ord("E")):
                if link is not None and \
                        now - last_yaw_toggle >= YAW_TOGGLE_INTERVAL:
                    try:
                        link.send_yaw_toggle()
                    except (serial.SerialException, OSError) as e:
                        shared.request_stop(f"ESP32 시리얼 연결 끊김(송신): {e}")
                        break
                    last_yaw_toggle = now
            elif key in MANUAL_KEYS:
                if shared.mode == "AUTO":
                    # 자동 주행 중 사람 개입 = 비상정지 (키 자체는 적용 안 함)
                    exit_auto()
                else:
                    manual_text = MANUAL_KEYS[key]
                    set_ramp_target(manual_text)

        # 2. 자동모드 목표 반영 (메인 스레드가 계산한 프리셋)
        if shared.mode == "AUTO":
            tgt = shared.auto_target
            if tgt is not None and ramp is not None:
                ramp.set_target(*tgt)

        # 3. 서보 램프 송신 — 목표가 어디서 왔든(수동/자동) 여기 한 곳에서만 전송
        if ramp is not None:
            try:
                ramp.tick()
            except (serial.SerialException, OSError) as e:
                shared.request_stop(f"ESP32 시리얼 연결 끊김(송신): {e}")
                break

        # 4. 시리얼 수신
        tel, rx_age = {}, None
        if link is not None:
            try:
                tel = link.poll()
                rx_age = link.rx_age()
            except (serial.SerialException, OSError) as e:
                shared.request_stop(f"ESP32 시리얼 연결 끊김(수신): {e}")
                break

        # 5. 화면 그리기 (10Hz)
        if now - last_draw >= DRAW_INTERVAL:
            draw_screen(stdscr, tel, rx_age, manual_text, ramp, shared, args)
            last_draw = now

        time.sleep(LOOP_DT)


# ======================================================
# 메인 스레드: 카메라 + YOLO + (자동모드) 판단
# ======================================================

def run(stdscr, cam, model, link, ramp, planner, shared, args):
    """종료 사유를 반환 (curses가 화면을 되돌린 뒤에 출력하려고)."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1)

    ui = threading.Thread(target=ui_loop,
                          args=(stdscr, link, ramp, shared, args),
                          daemon=True)
    ui.start()

    smoother = FeatureSmoother(alpha=0.35)
    anchor = planner.goal.get("anchor", "side_mirror") if planner else None
    side = (planner.goal["sonar"].get("side", "right") if planner else "right")
    prev_mode = "MANUAL"

    loop_times = []
    t_prev = time.time()

    while not shared.stop.is_set():
        frame = cam.read()
        if frame is None:
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

        # 앵커/초음파는 goal이 있으면 항상 계산 (지도·마커 표시용, 추론 재사용이라 저렴)
        mode = shared.mode
        feature = None
        sonar = None
        if planner is not None:
            if mode == "AUTO" and prev_mode != "AUTO":   # 자동모드 진입 → 초기화
                planner.reset()
                smoother.value = None
            feature = smoother.update(anchor_from_result(res, anchor))
            sonar = link.side_pair_m(side) if link is not None else None

        # 자동모드 판단 (서보 전송은 UI 스레드가 함)
        if mode == "AUTO" and planner is not None:
            plan = planner.update(feature, sonar, frame.shape)
            shared.plan = plan
            shared.auto_target = servo_preset(
                plan["text"], args.invert_x, args.invert_y)
        elif mode == "MANUAL":
            shared.plan = None
            shared.auto_target = None
        prev_mode = mode

        if not args.no_show:
            out = res.plot()
            if planner is not None:
                # guide.py와 같은 시각 요소: 목표 마커 / 현재 앵커 / 지시 배너 / 미니맵
                goal_f = scaled_goal_feature(planner.goal, frame.shape)
                cv2.drawMarker(out, (int(goal_f["u"]), int(goal_f["v"])),
                               (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                if feature:
                    cv2.circle(out, (int(feature["u"]), int(feature["v"])),
                               7, (0, 255, 0), 2)
                plan = shared.plan
                if plan is not None:
                    cv2.rectangle(out, (0, 0), (out.shape[1], 44),
                                  (20, 20, 20), -1)
                    cv2.putText(out, plan["text"], (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, plan["color"], 2)
                estimate = (relative_estimate(planner.goal, feature, sonar,
                                              frame.shape[1], args.hfov)
                            if feature is not None and sonar is not None
                            else None)
                panel = draw_top_view(
                    out.shape[0], pose=estimate,
                    target=planner.goal["taught_relative_pose"],
                    width=MAP_WIDTH)
                if estimate:
                    cv2.putText(panel, f"x={estimate['x']:+.2f}m", (6, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                    cv2.putText(panel, f"y~={estimate['y_proxy']:+.2f}m", (6, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                    yaw_deg = estimate["yaw"] * 180.0 / 3.141592653589793
                    cv2.putText(panel, f"yaw={yaw_deg:+.1f}deg", (6, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                out = np.hstack((out, panel))
            if args.display_scale != 1.0:
                # 작은 모니터용: 표시만 축소 (캡처/추론 해상도는 그대로)
                out = cv2.resize(out, None, fx=args.display_scale,
                                 fy=args.display_scale)
            cv2.imshow("main", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                shared.request_stop("사용자 종료 (q, 카메라 창)")
                break

    ui.join(timeout=2)
    return shared.stop_reason


def main():
    ap = argparse.ArgumentParser(description="카메라+초음파/서보 통합 조종석")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("model", nargs="?",
                    default=os.path.join(here, "camera", "best_v5_poster.pt"),
                    help="모델 경로 (기본: camera/best_v5_poster.pt)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=DEFAULT_FPS,
                     help="카메라 프레임레이트 (C920 지원: 5/10/15/20/25/30)")
    ap.add_argument("--no-show", action="store_true", help="카메라 창 없이 터미널 화면만")
    ap.add_argument("--serial-port", default=DEFAULT_PORT)
    ap.add_argument("--serial-baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--no-servo", action="store_true", help="ESP32 없이 카메라만 테스트")
    # 자동모드 (G 키) — guidance 판단기 사용
    ap.add_argument("--goal", default=os.path.join(here, "guidance",
                                                    "real_front_goal.json"),
                    help="teach.py로 저장한 goal 파일 (자동모드용)")
    # 편향각은 hardware/esp32.py의 실측 프리셋을 그대로 사용 (수동/자동 공통)
    ap.add_argument("--invert-x", action="store_true",
                    help="좌/우가 반대로 움직이면 지정 (프리셋 좌우 맞바꿈)")
    ap.add_argument("--invert-y", action="store_true",
                    help="전/후가 반대로 움직이면 지정 (프리셋 전후 맞바꿈)")
    # C920 수평 화각: 16:9에서 약 70.4도(=1.229rad) — 지도 위치 추정에 사용
    ap.add_argument("--hfov", type=float, default=1.229)
    ap.add_argument("--display-scale", type=float, default=0.5,
                    help="카메라+지도 창 배율 (작은 모니터면 0.5 등, 추론엔 영향 없음)")
    args = ap.parse_args()

    if not args.no_show:
        ensure_display()

    model = YOLO(args.model)

    try:
        cam = Webcam(index=args.cam, width=args.cam_width, height=args.cam_height,
                     fps=args.cam_fps, csi=args.csi)
    except RuntimeError as e:
        sys.exit(str(e))

    link = None
    ramp = None
    if not args.no_servo:
        try:
            link = ESP32Link(args.serial_port, args.serial_baud)
            link.send_servo(NEUTRAL_X_DEG, NEUTRAL_Y_DEG)  # 초기 중립(정지)
            ramp = ServoRamp(link, keepalive=SERVO_KEEPALIVE)
            print(f"ESP32 연결됨 ({args.serial_port})")
        except serial.SerialException as e:
            print(f"ESP32 연결 실패, 초음파/서보 없이 진행: {e}")
            link = None

    shared = Shared()
    planner = None
    try:
        planner = AlignmentPlanner(load_goal(args.goal))
        shared.auto_available = True
        shared.goal_msg = f"goal: {os.path.basename(args.goal)}"
        print(f"goal 로드됨 ({args.goal}) — G 키로 자동모드 사용 가능")
    except (FileNotFoundError, ValueError, KeyError) as e:
        shared.goal_msg = "goal 없음 (guidance/teach.py로 먼저 저장) — 자동모드 비활성"
        print(f"goal 로드 실패 → 자동모드 비활성: {e}")

    # 젯슨 첫 추론은 CUDA 커널 준비로 수십 초 걸림. curses 화면으로 들어가기 전에
    # 미리 소진해야 사용자가 빈 화면만 보며 기다리는 상황을 피할 수 있음.
    print("모델 워밍업 중... (젯슨 첫 추론 특성상 최대 1분 정도 걸릴 수 있음)")
    t0 = time.time()
    frame = cam.read()
    if frame is not None:
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, iou=0.5, verbose=False)
    print(f"워밍업 완료 ({time.time() - t0:.1f}s) — 화면 전환")

    reason = None
    try:
        reason = curses.wrapper(run, cam, model, link, ramp, planner, shared,
                                args)
    except KeyboardInterrupt:
        reason = "Ctrl-C 종료"
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if link is not None:
            # 어떤 경로로 끝나든 조이스틱은 중립(정지)으로 되돌리고 종료.
            # 마지막 송신 위치에서 램프로 복귀 (급점프 → 브라운아웃 방지),
            # 최대 2초 (최악 이동폭 50도 ≈ 1초).
            try:
                ramp.set_target(NEUTRAL_X_DEG, NEUTRAL_Y_DEG)
                deadline = time.time() + 2.0
                while not ramp.settled and time.time() < deadline:
                    ramp.tick()
                    time.sleep(0.01)
            except Exception:
                pass
            link.close()

    # curses가 화면을 원래대로 돌려놓은 뒤에 출력해야 사용자가 볼 수 있음
    if reason:
        print(reason)


if __name__ == "__main__":
    main()
