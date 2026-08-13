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

    @property
    def telemetry(self):
        """최신 원시 텔레메트리 dict (cm 단위 그대로) — 화면 표시용."""
        return self.link.telemetry

    def rx_age(self):
        return self.link.rx_age()

    def close(self):
        self.link.close()


def draw_telemetry_overlay(image, sonar):
    """카메라 프레임 좌하단에 ESP32 원시 텔레메트리를 표시 (main.py 화면의 축약판).

    'sonar INVALID'만 보고 원인을 알 수 없는 문제 해결용 — 실제로 어떤 값이
    오는지(-1인지, 수신이 끊겼는지, 범위 밖인지)를 눈으로 확인할 수 있게 함.
    거리 단위는 펌웨어 원본 그대로 cm.
    """
    tel = sonar.telemetry
    age = sonar.rx_age()

    def f(key):
        value = tel.get(key)
        try:
            return f'{float(value):.0f}'
        except (TypeError, ValueError):
            return '--'

    rx = 'RX ---' if age is None else f'RX {age:.1f}s'
    lines = [
        f'{rx} | front {f("front")}cm',
        f'L f{f("left_front")} r{f("left_rear")} d{f("left_dist")} a{f("left_angle")}',
        f'R f{f("right_front")} r{f("right_rear")} d{f("right_dist")} a{f("right_angle")}',
        f'servo echo {f("servo_x")}/{f("servo_y")}',
    ]

    height = image.shape[0]
    y0 = height - 8 - 18 * (len(lines) - 1)
    cv2.rectangle(image, (0, y0 - 16), (255, height), (20, 20, 20), -1)
    # 수신이 끊겼거나 1초 이상 오래되면 첫 줄을 빨간색으로
    head_color = (0, 0, 255) if (age is None or age > 1.0) else (180, 255, 180)
    for i, line in enumerate(lines):
        color = head_color if i == 0 else (200, 200, 200)
        cv2.putText(image, line, (6, y0 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def load_model(path=DEFAULT_MODEL):
    return YOLO(str(path))


def anchor_from_result(result, anchor='side_mirror'):
    """이미 수행된 YOLO Results에서 앵커 클래스만 뽑아 최고 신뢰도 하나를 반환.

    main.py처럼 predict를 한 번만 돌리고 (검출 개수 표시 + 앵커 추출을)
    같이 하고 싶을 때 사용.
    """
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


def extract_anchor(model, frame, anchor='side_mirror', confidence=0.25,
                   imgsz=640):
    result = model.predict(
        frame, verbose=False, conf=confidence, imgsz=imgsz, iou=0.5)[0]
    return anchor_from_result(result, anchor)


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


# ======================================================
# 정렬 판단기 — guide.py(단독 실행)와 main.py(자동모드) 공용
# ======================================================

EPS_U = 0.035
EPS_SCALE = 0.060
EPS_SONAR_DIFF = 0.035
STABLE_FRAMES = 8

SERVO_CENTER = 90.0


class AlignmentPlanner:
    """teach로 저장한 goal 대비 현재 상태를 비교해 이동 지시를 내는 상태기계.

    ①VISION 단계: 앵커(사이드미러) 위치·크기를 goal에 맞춤 (IBVS)
    ②SONAR 단계: 측면 초음파 앞/뒤 차이를 goal과 맞춰 평행 정렬

    update()가 매 프레임 지시 dict를 반환:
      {'text': 'TURN LEFT', 'color': (B,G,R), 'phase': 'VISION',
       'stable': 3, 'du': .., 'scale': .., 'sonar_err': ..(SONAR 단계만)}
    이동 지시는 TURN LEFT/RIGHT, MOVE FORWARD/BACKWARD 4종뿐이고
    나머지(STOP류/HOLD/ALIGNED)는 전부 "정지"로 해석하면 됨.
    """

    def __init__(self, goal):
        self.goal = goal
        self.phase = 'VISION'
        self.stable = 0

    def reset(self):
        self.phase, self.stable = 'VISION', 0

    @staticmethod
    def _direction(value, positive, negative):
        return positive if value > 0.0 else negative

    def update(self, feature, sonar, image_shape):
        out = {'phase': self.phase, 'stable': self.stable,
               'du': None, 'scale': None, 'sonar_err': None}

        if feature is None:
            self.stable = 0
            out.update(text='STOP - FIND TARGET', color=(0, 0, 255),
                       stable=0)
            return out

        goal_f = scaled_goal_feature(self.goal, image_shape)
        du = (goal_f['u'] - feature['u']) / (image_shape[1] / 2.0)
        scale = goal_f['size'] / max(feature['size'], 1e-3) - 1.0
        out.update(du=du, scale=scale)

        if self.phase == 'VISION':
            vision_ok = abs(du) < EPS_U and abs(scale) < EPS_SCALE
            self.stable = self.stable + 1 if vision_ok else 0
            out['stable'] = self.stable
            if self.stable >= STABLE_FRAMES:
                self.phase, self.stable = 'SONAR', 0
                out.update(text='STOP - IBVS LOCKED', color=(0, 255, 255))
                return out
            if abs(du) >= EPS_U:
                text = self._direction(du, 'TURN LEFT', 'TURN RIGHT')
            elif abs(scale) >= EPS_SCALE:
                text = self._direction(scale, 'MOVE FORWARD', 'MOVE BACKWARD')
            else:
                text = 'HOLD STILL'
            out.update(text=text, color=(0, 200, 255))
            return out

        # SONAR 단계
        if sonar is None:
            self.stable = 0
            out.update(text='STOP - SONAR INVALID', color=(0, 0, 255),
                       stable=0, phase=self.phase)
            return out
        taught = self.goal['sonar']
        raw_error = ((sonar['front'] - sonar['rear']) -
                     (taught['front'] - taught['rear']))
        side_sign = 1.0 if taught.get('side', 'right') == 'right' else -1.0
        error = side_sign * raw_error
        out['sonar_err'] = error
        sonar_ok = abs(error) < EPS_SONAR_DIFF
        self.stable = self.stable + 1 if sonar_ok else 0
        out['stable'] = self.stable
        out['phase'] = self.phase
        if self.stable >= STABLE_FRAMES:
            out.update(text='ALIGNED - STOP', color=(0, 255, 0))
            return out
        if sonar_ok:
            # 허용범위 안에서 확정 카운트 중일 땐 정지 유지 — 원본은 여기서도
            # TURN을 반환해서 --drive 시 정렬 막판에 서보가 좌우로 떨렸음
            out.update(text='HOLD STILL', color=(0, 200, 255))
            return out
        out.update(text=self._direction(error, 'TURN LEFT', 'TURN RIGHT'),
                   color=(0, 200, 255))
        return out


def servo_targets(text, turn_deg=10.0, drive_deg=10.0,
                  invert_x=False, invert_y=False):
    """지시 문구 → 서보 목표각 (x, y). 이동 지시 4개만 편향, 나머지는 중앙=정지.

    어느 방향이 물리적으로 좌/전진인지는 서보-조이스틱 장착 방향에 달렸으므로
    invert_x/invert_y로 현장에서 맞출 것.
    """
    x = y = SERVO_CENTER
    sx = -1.0 if invert_x else 1.0
    sy = -1.0 if invert_y else 1.0
    if text == 'TURN LEFT':
        x = SERVO_CENTER - sx * turn_deg
    elif text == 'TURN RIGHT':
        x = SERVO_CENTER + sx * turn_deg
    elif text == 'MOVE FORWARD':
        y = SERVO_CENTER + sy * drive_deg
    elif text == 'MOVE BACKWARD':
        y = SERVO_CENTER - sy * drive_deg
    return x, y


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
