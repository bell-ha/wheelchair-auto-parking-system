#!/usr/bin/env python3
"""ROS-free helpers for Jetson Nano human-guided alignment."""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DEFAULT_GOAL = HERE / 'real_front_goal.json'
# 이 리포 구조 기준: jetson_nano/guidance/ 에서 한 단계 위의 camera/ 에 모델이 있음
DEFAULT_MODEL = HERE.parent / 'camera' / 'best_v5_poster.pt'
# 좌/우 측면의 앞-뒤 초음파 센서 간 거리(m).
# ultrasound/sample.ino 의 SIDE_SENSOR_SPACING_CM = 31.5 와 반드시 일치해야 함
# (yaw 추정이 이 값에 정비례 — 틀리면 기울기 각도가 그 배율만큼 왜곡됨).
SONAR_BASELINE = 0.315
SONAR_MAX = 3.0
MAP_WIDTH = 360


class Webcam:

    def __init__(self, index=0, width=1280, height=720, fps=30, csi=False):
        os.environ.setdefault('DISPLAY', ':0')
        os.environ.setdefault('XAUTHORITY', '/run/user/1000/gdm/Xauthority')
        if csi:
            pipeline = (
                'nvarguscamerasrc ! '
                f'video/x-raw(memory:NVMM), width={width}, height={height}, '
                f'framerate={fps}/1 ! nvvidconv ! '
                'video/x-raw, format=BGRx ! videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink drop=1 max-buffers=1 sync=false')
        else:
            pipeline = (
                f'v4l2src device=/dev/video{index} ! '
                f'image/jpeg,width={width},height={height},framerate={fps}/1 ! '
                'jpegdec ! videoconvert ! video/x-raw,format=BGR ! '
                'appsink drop=1 max-buffers=1 sync=false')
        self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            raise RuntimeError(
                'cannot open Jetson camera; check camera index/--csi and '
                'GStreamer: YES in cv2.getBuildInformation()')

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self):
        self.capture.release()


class SerialSonar:
    """Read two ultrasonic distances from a non-blocking serial stream.

    Accepted newline-delimited formats:
      0.455,0.466
      front=0.455,rear=0.466
      {"front": 0.455, "rear": 0.466}
    Distances must be metres.
    """

    def __init__(self, port='/dev/ttyUSB0', baud=115200, side='right',
                 front_index=0, rear_index=2, stale_seconds=0.7):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('pyserial is required: pip3 install pyserial') from exc
        self.serial = serial.Serial(port, baudrate=baud, timeout=0)
        # 현재 펌웨어(ultrasound/sample.ino)는 {side}_front/{side}_rear 이름으로
        # 보냄. 'us' 배열 스키마는 삭제된 구버전 펌웨어 호환용 레거시 경로.
        self.side = side
        self.front_index = front_index
        self.rear_index = rear_index
        self.stale_seconds = stale_seconds
        self.values = None
        self.last_update = 0.0
        self.buffer = bytearray()

    def _parse(self, line):
        text = line.strip()
        if not text:
            return None
        try:
            if text.startswith('{'):
                data = json.loads(text)
                named_front = f'{self.side}_front'
                named_rear = f'{self.side}_rear'
                if named_front in data and named_rear in data:
                    # ultrasound/sample.ino telemetry schema, centimetres.
                    front = float(data[named_front]) / 100.0
                    rear = float(data[named_rear]) / 100.0
                elif 'us' in data:
                    # Legacy schema (removed ultrasound.ino), centimetres.
                    values = data['us']
                    front = float(values[self.front_index]) / 100.0
                    rear = float(values[self.rear_index]) / 100.0
                else:
                    # Compatibility with the simple JSON protocol. Values larger
                    # than the controller's metre range are interpreted as cm.
                    front, rear = float(data['front']), float(data['rear'])
                    if data.get('unit') == 'cm' or max(front, rear) >= SONAR_MAX:
                        front /= 100.0
                        rear /= 100.0
            elif '=' in text:
                data = {}
                for item in text.split(','):
                    key, value = item.split('=', 1)
                    data[key.strip()] = float(value)
                front, rear = data['front'], data['rear']
            else:
                front, rear = (float(value) for value in text.split(',', 1))
        except (ValueError, KeyError, IndexError, TypeError,
                json.JSONDecodeError):
            return None
        if not all(math.isfinite(v) and 0.0 < v < SONAR_MAX
                   for v in (front, rear)):
            return None
        return {'front': front, 'rear': rear}

    def poll(self):
        waiting = self.serial.in_waiting
        if waiting:
            self.buffer.extend(self.serial.read(waiting))
        while b'\n' in self.buffer:
            raw, _, self.buffer = self.buffer.partition(b'\n')
            parsed = self._parse(raw.decode('utf-8', errors='ignore'))
            if parsed is not None:
                self.values = parsed
                self.last_update = time.monotonic()
        if (self.values is None or
                time.monotonic() - self.last_update > self.stale_seconds):
            return None
        return dict(self.values)

    def close(self):
        self.serial.close()


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
