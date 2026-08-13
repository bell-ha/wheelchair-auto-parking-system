#!/usr/bin/env python3
"""ROS-free helpers for Jetson Nano human-guided alignment.

카메라/ESP32 하드웨어 접근은 상위의 hardware/ 공용 계층을 사용한다
(main.py와 같은 모듈 — 설정/프로토콜 변경 시 hardware/ 한 곳만 고치면 됨).
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # hardware/ 공용 계층 import용
from hardware.camera import Webcam as _HwWebcam, ensure_display, DEFAULT_FPS
from hardware.esp32 import ESP32Link

DEFAULT_GOAL = HERE / 'real_front_goal.json'
# 이 리포 구조 기준: jetson_nano/guidance/ 에서 한 단계 위의 camera/ 에 모델이 있음
DEFAULT_MODEL = HERE.parent / 'camera' / 'best_v5_poster.pt'
# 좌/우 측면의 앞-뒤 초음파 센서 간 거리(m).
# ultrasound/sample.ino 의 SIDE_SENSOR_SPACING_CM = 31.5 와 반드시 일치해야 함
# (yaw 추정이 이 값에 정비례 — 틀리면 기울기 각도가 그 배율만큼 왜곡됨).
SONAR_BASELINE = 0.315
SONAR_MAX = 3.0
MAP_WIDTH = 360


class Webcam(_HwWebcam):
    """hardware.camera.Webcam + 창을 젯슨 HDMI(:0)로 보내는 환경 설정."""

    def __init__(self, index=0, width=1280, height=720, fps=DEFAULT_FPS, csi=False):
        ensure_display()
        super().__init__(index=index, width=width, height=height, fps=fps, csi=csi)


class SerialSonar:
    """hardware.esp32.ESP32Link의 guidance용 얇은 래퍼.

    poll()은 측면 앞/뒤 초음파 쌍을 m 단위 {'front','rear'}로 반환 (무효면 None).
    front_index/rear_index 인자는 구버전 CLI 호환용으로 받기만 하고 사용 안 함
    (현 펌웨어는 {side}_front/{side}_rear 이름으로 보냄).
    """

    def __init__(self, port='/dev/ttyUSB0', baud=115200, side='right',
                 front_index=0, rear_index=2, stale_seconds=0.7):
        self.link = ESP32Link(port, baud, stale_seconds=stale_seconds)
        self.side = side

    def poll(self):
        self.link.poll()
        return self.link.side_pair_m(self.side)

    def send_servo(self, x_deg, y_deg):
        """서보(조이스틱 액추에이터) 명령 송신 — 같은 ESP32 연결 재사용."""
        self.link.send_servo(x_deg, y_deg)

    def close(self):
        self.link.close()


def load_model(path=DEFAULT_MODEL):
    return YOLO(str(path))


def extract_anchor(model, frame, anchor='side_mirror', confidence=0.25,
                   imgsz=640):
    result = model.predict(
        frame, verbose=False, conf=confidence, imgsz=imgsz, iou=0.5)[0]
    candidates = []
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        if name != anchor:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        candidates.append({
            'u': (x1 + x2) / 2.0,
            'v': (y1 + y2) / 2.0,
            'size': math.hypot(x2 - x1, y2 - y1),
            'conf': float(box.conf[0]),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })
    return max(candidates, key=lambda item: item['conf']) if candidates else None


class FeatureSmoother:

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def update(self, feature):
        if feature is None:
            return None
        if self.value is None:
            self.value = dict(feature)
        else:
            for key in ('u', 'v', 'size'):
                self.value[key] = (self.alpha * feature[key] +
                                   (1.0 - self.alpha) * self.value[key])
            for key in ('conf', 'x1', 'y1', 'x2', 'y2'):
                self.value[key] = feature[key]
        return dict(self.value)


def compact_feature(feature):
    return {key: float(feature[key]) for key in ('u', 'v', 'size')}


def save_goal(path, feature, sonar, image_shape, target_pose, anchor, side):
    payload = {
        'schema': 'jetson_human_guidance_v1',
        'anchor': anchor,
        'vision': {anchor: compact_feature(feature)},
        'sonar': {'side': side, **sonar},
        'camera': {'width': image_shape[1], 'height': image_shape[0]},
        'taught_relative_pose': target_pose,
    }
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def load_goal(path):
    with open(path, encoding='utf-8') as fp:
        goal = json.load(fp)
    anchor = goal.get('anchor', 'side_mirror')
    if anchor not in goal.get('vision', {}) or not goal.get('sonar'):
        raise ValueError('goal requires one vision anchor and two sonar values')
    return goal


def scaled_goal_feature(goal, image_shape):
    anchor = goal.get('anchor', 'side_mirror')
    feature = goal['vision'][anchor]
    camera = goal.get('camera', {})
    sx = image_shape[1] / max(float(camera.get('width', image_shape[1])), 1.0)
    sy = image_shape[0] / max(float(camera.get('height', image_shape[0])), 1.0)
    size_scale = math.sqrt(0.5 * (sx * sx + sy * sy))
    return {'u': feature['u'] * sx, 'v': feature['v'] * sy,
            'size': feature['size'] * size_scale}


def relative_estimate(goal, feature, sonar, image_width, hfov_rad):
    target = goal['taught_relative_pose']
    taught = goal['sonar']
    side_sign = 1.0 if taught.get('side', 'right') == 'right' else -1.0
    current_mean = 0.5 * (sonar['front'] + sonar['rear'])
    taught_mean = 0.5 * (taught['front'] + taught['rear'])
    x = float(target['x']) + side_sign * (current_mean - taught_mean)
    diff_delta = ((sonar['front'] - sonar['rear']) -
                  (taught['front'] - taught['rear']))
    yaw_delta = side_sign * math.atan2(diff_delta, SONAR_BASELINE)
    yaw = float(target['yaw']) + yaw_delta
    goal_f = scaled_goal_feature(goal, (1, image_width, 3))
    fx = image_width / (2.0 * math.tan(hfov_rad / 2.0))
    current_bearing = math.atan2(feature['u'] - image_width / 2.0, fx)
    goal_bearing = math.atan2(goal_f['u'] - image_width / 2.0, fx)
    y_proxy = float(target['y']) + current_mean * (
        math.tan(current_bearing) - math.tan(goal_bearing))
    return {'x': x, 'y_proxy': y_proxy, 'yaw': yaw,
            'sonar_mean': current_mean, 'yaw_delta': yaw_delta}


def map_geometry(height, width=MAP_WIDTH):
    scale = min(65.0, max(35.0, (height - 40.0) / 5.2))
    return width / 2.0, height / 2.0, scale


def pose_to_map(x, y, height, width=MAP_WIDTH):
    cx, cy, scale = map_geometry(height, width)
    return int(cx + scale * x), int(cy - scale * y)


def map_to_pose(px, py, height, width=MAP_WIDTH):
    cx, cy, scale = map_geometry(height, width)
    return (px - cx) / scale, (cy - py) / scale


def draw_top_view(height, pose=None, target=None, width=MAP_WIDTH):
    panel = np.full((height, width, 3), 35, dtype=np.uint8)
    car_center = pose_to_map(0.0, 0.0, height, width)
    _, _, scale = map_geometry(height, width)
    half_w, half_l = int(0.9 * scale), int(2.25 * scale)
    cv2.rectangle(panel,
                  (car_center[0] - half_w, car_center[1] - half_l),
                  (car_center[0] + half_w, car_center[1] + half_l),
                  (105, 105, 105), -1)
    cv2.putText(panel, 'CAR', (car_center[0] - 20, car_center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if target is not None:
        point = pose_to_map(target['x'], target['y'], height, width)
        cv2.drawMarker(panel, point, (255, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(panel, 'GT TARGET', (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    if pose is not None:
        point = pose_to_map(pose['x'], pose['y_proxy'], height, width)
        tip = (int(point[0] + 25 * math.cos(pose['yaw'])),
               int(point[1] - 25 * math.sin(pose['yaw'])))
        cv2.circle(panel, point, 12, (0, 200, 255), 2)
        cv2.arrowedLine(panel, point, tip, (0, 200, 255), 2)
    return panel
