#!/usr/bin/env python3
"""단독 실행용 정렬 가이드 (cv2 창 기반).

판단 로직은 common.AlignmentPlanner에 있고, main.py의 자동모드도 같은
판단기를 쓴다 — 이 파일은 "guidance만 따로 돌리고 싶을 때"의 진입점.
기본은 지시 화면 표시만, --drive를 주면 서보(조이스틱)로 실제 전송.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (DEFAULT_GOAL, DEFAULT_MODEL, MAP_WIDTH, STABLE_FRAMES,
                    SERVO_CENTER, AlignmentPlanner, FeatureSmoother,
                    SerialSonar, Webcam, draw_telemetry_overlay, draw_top_view,
                    extract_anchor, load_goal, load_model, relative_estimate,
                    scaled_goal_feature, servo_targets)


CAMERA_FAIL_SECONDS = 5.0
SERVO_KEEPALIVE = 0.5      # --drive: 같은 목표각이라도 이 주기(초)마다 재전송


class GuideApp:

    WINDOW = 'Jetson Human Alignment Guide'

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
        self.planner = AlignmentPlanner(self.goal)
        self.last_sent = None
        self.last_sent_time = 0.0
        if args.drive:
            print('*** DRIVE MODE: 지시가 서보(조이스틱)로 실제 전송됩니다 ***')
            print('*** 첫 테스트는 반드시 휠체어 전원을 끄거나 바퀴를 띄운 상태로! ***')

    def _drive(self, text):
        """--drive일 때만 호출. 목표각이 바뀌었거나 keepalive 주기가 지나면 전송."""
        x, y = servo_targets(text, self.args.turn_deg, self.args.drive_deg,
                             self.args.invert_x, self.args.invert_y)
        now = time.monotonic()
        if (x, y) != self.last_sent or now - self.last_sent_time > SERVO_KEEPALIVE:
            self.sonar.send_servo(x, y)
            self.last_sent = (x, y)
            self.last_sent_time = now
        return x, y

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
                plan = self.planner.update(feature, sonar, frame.shape)
                text, color = plan['text'], plan['color']
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
                detail = (f"phase={self.planner.phase} "
                          f"stable={self.planner.stable}/{STABLE_FRAMES}")
                if plan['du'] is not None:
                    detail += f" du={plan['du']:+.3f} scale={plan['scale']:+.3f}"
                if plan['sonar_err'] is not None:
                    detail += f" sonarYaw={plan['sonar_err']:+.3f}m"
                if servo_xy is not None:
                    detail += f" | DRIVE x={servo_xy[0]:.0f} y={servo_xy[1]:.0f}"
                else:
                    detail += " | display only"
                cv2.putText(out, detail, (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (220, 220, 220), 1)
                draw_telemetry_overlay(out, self.sonar)
                shown = np.hstack((out, self._map(estimate, out.shape[0])))
                if self.args.display_scale != 1.0:
                    # 작은 모니터용: 표시만 축소 (캡처/추론 해상도는 그대로)
                    shown = cv2.resize(shown, None, fx=self.args.display_scale,
                                       fy=self.args.display_scale)
                cv2.imshow(self.WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.planner.reset()
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
    parser.add_argument('--fps', type=int, default=15)
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
    parser.add_argument('--display-scale', type=float, default=0.5,
                        help='표시 창 배율 (작은 모니터면 0.5 등, 추론엔 영향 없음)')
    return parser.parse_args()


if __name__ == '__main__':
    GuideApp(parse_args()).run()
