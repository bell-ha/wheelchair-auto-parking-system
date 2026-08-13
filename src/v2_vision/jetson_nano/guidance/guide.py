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

    @staticmethod
    def _direction(value, positive, negative):
        return positive if value > 0.0 else negative

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
                cv2.putText(out, detail, (12, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (220, 220, 220), 1)
                cv2.imshow(self.WINDOW,
                           np.hstack((out, self._map(estimate, out.shape[0]))))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.phase, self.stable = 'VISION', 0
                elif key == ord('q'):
                    break
        finally:
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
    return parser.parse_args()


if __name__ == '__main__':
    GuideApp(parse_args()).run()
