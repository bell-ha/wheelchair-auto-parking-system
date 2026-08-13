#!/usr/bin/env python3
"""ROS-free human movement guide for Jetson Nano; no motor output."""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (DEFAULT_GOAL, DEFAULT_MODEL, MAP_WIDTH, FeatureSmoother,
                    SerialSonar, Webcam, draw_top_view, extract_anchor,
                    load_goal, load_model, relative_estimate,
                    scaled_goal_feature)


EPS_U = 0.035
EPS_SCALE = 0.060
EPS_SONAR_DIFF = 0.035
STABLE_FRAMES = 8
CAMERA_FAIL_SECONDS = 5.0

# --drive 모드: 지시를 서보(조이스틱 액추에이터)로 실제 전송할 때의 값들
SERVO_CENTER = 90.0        # 중앙 = 조이스틱 중립 = 정지
SERVO_KEEPALIVE = 0.5      # 같은 목표각이라도 이 주기(초)마다 재전송


class GuideApp:

    WINDOW = 'Jetson Human Alignment Guide - NO MOTOR OUTPUT'

    def __init__(self, args):
        self.args = args
        self.goal = load_goal(args.goal)
        self.anchor = self.goal.get('anchor', 'side_mirror')
        self.webcam = Webcam(
            args.camera, args.width, args.height, args.fps, args.csi)
        side = self.goal['sonar'].get('side', 'right')
        self.sonar = SerialSonar(
            args.serial_port, args.baud, side,
            args.sonar_front_index, args.sonar_rear_index)
        self.model = load_model(args.model)
        self.smoother = FeatureSmoother(alpha=0.35)
        self.phase = 'VISION'
        self.stable = 0
        self.last_sent = None
        self.last_sent_time = 0.0
        if args.drive:
            print('*** DRIVE MODE: 지시가 서보(조이스틱)로 실제 전송됩니다 ***')
            print('*** 첫 테스트는 반드시 휠체어 전원을 끄거나 바퀴를 띄운 상태로! ***')

    @staticmethod
    def _direction(value, positive, negative):
        return positive if value > 0.0 else negative

    def _servo_targets(self, text):
        """지시 문구 → 서보 목표각. 이동 지시 4개만 편향, 나머지(STOP류/미지)는 중앙=정지.

        어느 방향이 물리적으로 좌/전진인지는 서보-조이스틱 장착 방향에 달렸으므로
        --invert-x/--invert-y로 현장에서 맞출 것.
        """
        x = y = SERVO_CENTER
        sx = -1.0 if self.args.invert_x else 1.0
        sy = -1.0 if self.args.invert_y else 1.0
        if text == 'TURN LEFT':
            x = SERVO_CENTER - sx * self.args.turn_deg
        elif text == 'TURN RIGHT':
            x = SERVO_CENTER + sx * self.args.turn_deg
        elif text == 'MOVE FORWARD':
            y = SERVO_CENTER + sy * self.args.drive_deg
        elif text == 'MOVE BACKWARD':
            y = SERVO_CENTER - sy * self.args.drive_deg
        return x, y

    def _drive(self, text):
        """--drive일 때만 호출. 목표각이 바뀌었거나 keepalive 주기가 지나면 전송."""
        x, y = self._servo_targets(text)
        now = time.monotonic()
        if (x, y) != self.last_sent or now - self.last_sent_time > SERVO_KEEPALIVE:
            self.sonar.send_servo(x, y)
            self.last_sent = (x, y)
            self.last_sent_time = now
        return x, y

    def _instruction(self, feature, sonar, image_shape):
        if feature is None:
            self.stable = 0
            return 'STOP - FIND TARGET', (0, 0, 255), None
        goal_f = scaled_goal_feature(self.goal, image_shape)
        du = (goal_f['u'] - feature['u']) / (image_shape[1] / 2.0)
        scale = goal_f['size'] / max(feature['size'], 1e-3) - 1.0

        # Same ordering as the simulator: finish visual IBVS first, then latch
        # it and use only the side-sonar pair for parallel alignment.
        if self.phase == 'VISION':
            vision_ok = abs(du) < EPS_U and abs(scale) < EPS_SCALE
            self.stable = self.stable + 1 if vision_ok else 0
            if self.stable >= STABLE_FRAMES:
                self.phase, self.stable = 'SONAR', 0
                return 'STOP - IBVS LOCKED', (0, 255, 255), (du, scale, None)
            if abs(du) >= EPS_U:
                text = self._direction(du, 'TURN LEFT', 'TURN RIGHT')
            elif abs(scale) >= EPS_SCALE:
                text = self._direction(scale, 'MOVE FORWARD', 'MOVE BACKWARD')
            else:
                text = 'HOLD STILL'
            return text, (0, 200, 255), (du, scale, None)

        if sonar is None:
            self.stable = 0
            return 'STOP - SONAR INVALID', (0, 0, 255), (du, scale, None)
        taught = self.goal['sonar']
        raw_error = ((sonar['front'] - sonar['rear']) -
                     (taught['front'] - taught['rear']))
        side_sign = 1.0 if taught.get('side', 'right') == 'right' else -1.0
        error = side_sign * raw_error
        sonar_ok = abs(error) < EPS_SONAR_DIFF
        self.stable = self.stable + 1 if sonar_ok else 0
        if self.stable >= STABLE_FRAMES:
            return 'ALIGNED - STOP', (0, 255, 0), (du, scale, error)
        return self._direction(error, 'TURN LEFT', 'TURN RIGHT'), \
            (0, 200, 255), (du, scale, error)

    def _map(self, estimate, height):
        target = self.goal['taught_relative_pose']
        panel = draw_top_view(
            height, pose=estimate, target=target, width=MAP_WIDTH)
        if estimate:
            cv2.putText(panel, f"x={estimate['x']:+.2f}m", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.putText(panel, f"y~={estimate['y_proxy']:+.2f}m", (6, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.putText(panel,
                        f"yaw={math.degrees(estimate['yaw']):+.1f}deg",
                        (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (255, 255, 255), 1)
        cv2.putText(panel, 'y~: monocular proxy, not measured GT',
                    (6, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (130, 180, 255), 1)
        return panel

    def run(self):
        fail_since = None
        try:
            while True:
                frame = self.webcam.read()
                sonar = self.sonar.poll()
                if frame is None:
                    # 카메라가 죽으면(USB 등) 빈 루프만 돌지 말고 사유를 남기고 종료
                    if fail_since is None:
                        fail_since = time.monotonic()
                    elif time.monotonic() - fail_since > CAMERA_FAIL_SECONDS:
                        raise RuntimeError(
                            'camera produced no frames for '
                            f'{CAMERA_FAIL_SECONDS:.0f}s; check USB/dmesg')
                    continue
                fail_since = None
                raw = extract_anchor(
                    self.model, frame, self.anchor, self.args.confidence,
                    self.args.imgsz)
                feature = self.smoother.update(raw)
                text, color, errors = self._instruction(
                    feature, sonar, frame.shape)
                servo_xy = self._drive(text) if self.args.drive else None
                estimate = (relative_estimate(
                    self.goal, feature, sonar, frame.shape[1], self.args.hfov)
                    if feature is not None and sonar is not None else None)

                out = frame.copy()
                if feature:
                    cv2.circle(out, (int(feature['u']), int(feature['v'])),
                               7, (0, 255, 0), 2)
                goal_f = scaled_goal_feature(self.goal, frame.shape)
                cv2.drawMarker(out, (int(goal_f['u']), int(goal_f['v'])),
                               (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.rectangle(out, (0, 0), (out.shape[1], 88),
                              (20, 20, 20), -1)
                cv2.putText(out, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)
                detail = f"phase={self.phase} stable={self.stable}/{STABLE_FRAMES}"
                if errors:
                    detail += f" du={errors[0]:+.3f} scale={errors[1]:+.3f}"
                    if errors[2] is not None:
                        detail += f" sonarYaw={errors[2]:+.3f}m"
                if servo_xy is not None:
                    detail += f" | DRIVE x={servo_xy[0]:.0f} y={servo_xy[1]:.0f}"
                else:
                    detail += " | display only"
                cv2.putText(out, detail, (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (220, 220, 220), 1)
                shown = np.hstack((out, self._map(estimate, out.shape[0])))
                if self.args.display_scale != 1.0:
                    # 작은 모니터용: 표시만 축소 (캡처/추론 해상도는 그대로)
                    shown = cv2.resize(shown, None, fx=self.args.display_scale,
                                       fy=self.args.display_scale)
                cv2.imshow(self.WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.phase, self.stable = 'VISION', 0
                elif key == ord('q'):
                    break
        finally:
            # 어떤 경로로 끝나든(q, 예외, 카메라 사망) 조이스틱은 반드시 중립으로
            if self.args.drive:
                try:
                    for _ in range(3):
                        self.sonar.send_servo(SERVO_CENTER, SERVO_CENTER)
                        time.sleep(0.05)
                except Exception:
                    pass
            self.webcam.release()
            self.sonar.close()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    # C920은 4:3 해상도(640x480 등)를 요청하면 좌우가 크롭되므로 16:9 기본값 사용
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--csi', action='store_true')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--sonar-front-index', type=int, default=0)
    parser.add_argument('--sonar-rear-index', type=int, default=2)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    # C920 수평 화각: 16:9에서 약 70.4도(=1.229rad). 4:3 해상도에선 크롭돼 더 좁아짐
    parser.add_argument('--hfov', type=float, default=1.229)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--goal', default=str(DEFAULT_GOAL))
    # 서보(조이스틱 액추에이터) 실구동 — 기본은 화면 표시만
    parser.add_argument('--drive', action='store_true',
                        help='지시를 서보로 실제 전송 (첫 테스트는 휠체어 전원 OFF로!)')
    parser.add_argument('--turn-deg', type=float, default=10.0,
                        help='좌/우회전 시 서보 X 편향각 (중앙 90 기준, 기본 10)')
    parser.add_argument('--drive-deg', type=float, default=10.0,
                        help='전/후진 시 서보 Y 편향각 (중앙 90 기준, 기본 10)')
    parser.add_argument('--invert-x', action='store_true',
                        help='좌/우 방향이 반대로 움직이면 지정')
    parser.add_argument('--invert-y', action='store_true',
                        help='전/후 방향이 반대로 움직이면 지정')
    parser.add_argument('--display-scale', type=float, default=1.0,
                        help='표시 창 배율 (작은 모니터면 0.5 등, 추론엔 영향 없음)')
    return parser.parse_args()


if __name__ == '__main__':
    GuideApp(parse_args()).run()
