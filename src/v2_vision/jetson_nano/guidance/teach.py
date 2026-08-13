#!/usr/bin/env python3
"""ROS-free webcam/serial teach tool for Jetson Nano."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (DEFAULT_GOAL, DEFAULT_MODEL, MAP_WIDTH, SerialSonar,
                    Webcam, draw_top_view, extract_anchor, load_model,
                    map_to_pose, save_goal)


CAMERA_FAIL_SECONDS = 5.0


class TeachApp:

    WINDOW = 'Jetson Front Teach'

    def __init__(self, args):
        self.args = args
        self.webcam = Webcam(
            args.camera, args.width, args.height, args.fps, args.csi)
        self.sonar = SerialSonar(
            args.serial_port, args.baud, args.side,
            args.sonar_front_index, args.sonar_rear_index)
        self.model = load_model(args.model)
        self.frame = None
        self.target = None
        cv2.namedWindow(self.WINDOW)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

    def _mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return
        # 표시 배율 적용 중이면 클릭 좌표를 원본 좌표계로 환산
        scale = self.args.display_scale
        x = int(x / scale)
        y = int(y / scale)
        camera_width = self.frame.shape[1]
        if camera_width <= x < camera_width + MAP_WIDTH:
            px, py = map_to_pose(
                x - camera_width, y, self.frame.shape[0], MAP_WIDTH)
            self.target = {'x': px, 'y': py, 'yaw': self.args.target_yaw}
            print(f'GT target selected: x={px:.3f}, y={py:.3f}, '
                  f'yaw={self.args.target_yaw:.3f}')

    def run(self):
        fail_since = None
        try:
            while True:
                self.frame = self.webcam.read()
                sonar = self.sonar.poll()
                if self.frame is None:
                    # 카메라가 죽으면(USB 등) 빈 루프만 돌지 말고 사유를 남기고 종료
                    if fail_since is None:
                        fail_since = time.monotonic()
                    elif time.monotonic() - fail_since > CAMERA_FAIL_SECONDS:
                        raise RuntimeError(
                            'camera produced no frames for '
                            f'{CAMERA_FAIL_SECONDS:.0f}s; check USB/dmesg')
                    continue
                fail_since = None
                feature = extract_anchor(
                    self.model, self.frame, self.args.anchor,
                    self.args.confidence, self.args.imgsz)
                out = self.frame.copy()
                if feature:
                    cv2.rectangle(out, (int(feature['x1']), int(feature['y1'])),
                                  (int(feature['x2']), int(feature['y2'])),
                                  (0, 255, 0), 2)
                    cv2.circle(out, (int(feature['u']), int(feature['v'])),
                               5, (0, 255, 0), 2)
                sonar_text = (f"sonar f={sonar['front']:.3f} r={sonar['rear']:.3f}"
                              if sonar else 'sonar INVALID')
                cv2.putText(out, f"{self.args.anchor}: {'OK' if feature else 'MISS'}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)
                cv2.putText(out, sonar_text, (8, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(out, '[s] SAVE  [q] QUIT', (8, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255, 255, 255), 2)
                panel = draw_top_view(
                    self.frame.shape[0], target=self.target, width=MAP_WIDTH)
                cv2.putText(panel, 'CLICK GT TARGET POSITION', (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 255, 255), 1)
                shown = np.hstack((out, panel))
                if self.args.display_scale != 1.0:
                    # 작은 모니터용: 표시만 축소 (클릭 좌표는 _mouse에서 역환산)
                    shown = cv2.resize(shown, None, fx=self.args.display_scale,
                                       fy=self.args.display_scale)
                cv2.imshow(self.WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if feature is None or sonar is None or self.target is None:
                        print('SAVE BLOCKED: click target + feature + sonar required')
                    else:
                        save_goal(self.args.goal, feature, sonar, self.frame.shape,
                                  self.target, self.args.anchor, self.args.side)
                        print(f'goal saved: {self.args.goal}')
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
    parser.add_argument('--side', choices=('left', 'right'), default='right')
    parser.add_argument('--sonar-front-index', type=int, default=0)
    parser.add_argument('--sonar-rear-index', type=int, default=2)
    parser.add_argument('--target-yaw', type=float, default=-1.5708)
    parser.add_argument('--anchor', default='side_mirror')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--goal', default=str(DEFAULT_GOAL))
    parser.add_argument('--display-scale', type=float, default=1.0,
                        help='표시 창 배율 (작은 모니터면 0.5 등, 추론엔 영향 없음)')
    return parser.parse_args()


if __name__ == '__main__':
    TeachApp(parse_args()).run()
