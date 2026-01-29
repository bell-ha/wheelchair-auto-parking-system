# 프레임 수 조절
# 스무딩: 100이면 즉각
# DistGain: 거리 스케일 보정
# Rear_mapYaw: 해당 카메라가 맵에서 어느 방향을 보고 있는지
# Left_mapYaw
# Rear, Left_Sens: 마커에서 얻은 값을 실제 각도로 조정
# Rear,Left_InstAngle: 카메라 설치 각도 보정
# Rear,Left_InstOffset: 카메라 설치 오프셋 보정(기준점 잡기)
# 경로 모드: 주차/출차(1: 출차, 0: 주차)
# 출차 방향: 왼쪽/오른쪽(0: 왼쪽, 1: 오른쪽)


import cv2
import numpy as np
import math
import heapq

# ==========================================
# 1) 공통 Intrinsic (사용자 제공)
# ==========================================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE_M = 0.25  # 25cm = 0.25m

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

obj_points = np.array([
    [-MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
    [ MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
    [ MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0],
    [-MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0]
], dtype=np.float32)


def normalize_deg_0_360(d):
    d = d % 360.0
    return d if d >= 0 else d + 360.0


class IntegratedWheelchairMapTracker:
    def __init__(self):
        # ====== 맵/물리 ======
        self.marker_h_cm = 72.0

        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5
        self.off_x, self.off_y = 200, 150

        self.wc_w_cm, self.wc_l_cm = 57.0, 100.0
        self.half_len_px = (self.wc_l_cm / 2.0) * self.map_scale

        # 상태(중심 기준으로 유지)
        self.center_pos = None
        self.heading_angle = 0.0  # map rad
        self.is_initialized = False

        # 영상
        self.cap_rear = cv2.VideoCapture("rear.mp4")
        self.cap_left = cv2.VideoCapture("left.mp4")
        self.total_frames = int(min(
            self.cap_rear.get(cv2.CAP_PROP_FRAME_COUNT),
            self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT)
        ))
        self.curr_rear = None
        self.curr_left = None

        # 카메라 파라미터
        self.cams = {
            "rear": {
                "name": "Rear",
                "pos_px": np.array([301.4 + self.off_x, 540.0 + self.off_y], dtype=np.float32),
                "h_cm": 105.0,
                "map_angle_deg": 90.0,
                "yaw_trim_deg": 0.0,

                "sens": 1.6,
                "install_angle": 0.0,
                "install_offset": 0.0,

                "color": (100, 120, 255)
            },
            "left": {
                "name": "Left",
                "pos_px": np.array([200.0 + self.off_x, 270.0 + self.off_y], dtype=np.float32),
                "h_cm": 110.0,
                "map_angle_deg": 157.0,
                "yaw_trim_deg": 0.0,

                "sens": 1.6,
                "install_angle": 113.0,
                "install_offset": 50.84,

                "color": (255, 120, 100)
            }
        }

        # 기본값(아래 UI 초기값으로 덮어씀)
        self.alpha = 0.75
        self.dist_gain = 1.00

        # ==========================================
        # UI (초기값: 스샷 기준)
        # ==========================================
        self.win_name = "Integrated Wheelchair Tracker (ID0 front / ID1 rear) + Path"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        INIT_FRAME = 1478
        INIT_SMOOTH = 100          # alpha=1.00
        INIT_DISTGAIN = 188        # dist_gain=1.88
        INIT_REAR_MAPYAW = 93      # yaw_trim_deg = +3
        INIT_LEFT_MAPYAW = 98      # yaw_trim_deg = +8
        INIT_REAR_SENS_X10 = 16    # 1.6
        INIT_LEFT_SENS_X10 = 16    # 1.6
        INIT_REAR_INSTANGLE = 0
        INIT_LEFT_INSTANGLE = 113
        INIT_REAR_INSTOFFSET_X10 = 0
        INIT_LEFT_INSTOFFSET_X10 = 508   # 50.8 (50.84 근사)

        INIT_FRAME = int(max(0, min(INIT_FRAME, self.total_frames - 1)))

        # 내부 변수 동기화
        self.alpha = max(0.01, INIT_SMOOTH / 100.0)
        self.dist_gain = max(0.01, INIT_DISTGAIN / 100.0)

        self.cams["rear"]["yaw_trim_deg"] = float(INIT_REAR_MAPYAW - 90)
        self.cams["left"]["yaw_trim_deg"] = float(INIT_LEFT_MAPYAW - 90)

        self.cams["rear"]["sens"] = INIT_REAR_SENS_X10 / 10.0
        self.cams["left"]["sens"] = INIT_LEFT_SENS_X10 / 10.0

        self.cams["rear"]["install_angle"] = float(INIT_REAR_INSTANGLE)
        self.cams["left"]["install_angle"] = float(INIT_LEFT_INSTANGLE)

        self.cams["rear"]["install_offset"] = INIT_REAR_INSTOFFSET_X10 / 10.0
        self.cams["left"]["install_offset"] = INIT_LEFT_INSTOFFSET_X10 / 10.0

        # 트랙바 생성(초기값 반영)
        cv2.createTrackbar("Frame", self.win_name, INIT_FRAME, max(0, self.total_frames - 1), self.on_frame_change)
        cv2.createTrackbar("Smooth(%)", self.win_name, INIT_SMOOTH, 100, self.on_alpha)
        cv2.createTrackbar("DistGain(%)", self.win_name, INIT_DISTGAIN, 200, self.on_dist_gain)

        cv2.createTrackbar("Rear_MapYaw", self.win_name, INIT_REAR_MAPYAW, 180,
                           lambda v: self.set_cam("rear", "yaw_trim_deg", v - 90))
        cv2.createTrackbar("Left_MapYaw", self.win_name, INIT_LEFT_MAPYAW, 180,
                           lambda v: self.set_cam("left", "yaw_trim_deg", v - 90))

        cv2.createTrackbar("Rear_Sens(x10)", self.win_name, INIT_REAR_SENS_X10, 30,
                           lambda v: self.set_cam("rear", "sens", v / 10.0))
        cv2.createTrackbar("Left_Sens(x10)", self.win_name, INIT_LEFT_SENS_X10, 30,
                           lambda v: self.set_cam("left", "sens", v / 10.0))

        cv2.createTrackbar("Rear_InstAngle", self.win_name, INIT_REAR_INSTANGLE, 180,
                           lambda v: self.set_cam("rear", "install_angle", float(v)))
        cv2.createTrackbar("Left_InstAngle", self.win_name, INIT_LEFT_INSTANGLE, 180,
                           lambda v: self.set_cam("left", "install_angle", float(v)))

        cv2.createTrackbar("Rear_InstOffset(x10)", self.win_name, INIT_REAR_INSTOFFSET_X10, 1800,
                           lambda v: self.set_cam("rear", "install_offset", v / 10.0))
        cv2.createTrackbar("Left_InstOffset(x10)", self.win_name, INIT_LEFT_INSTOFFSET_X10, 1800,
                           lambda v: self.set_cam("left", "install_offset", v / 10.0))

        # ===== 경로/장애물 UI =====
        cv2.createTrackbar("Mode", self.win_name, 1, 1, self.on_mode)     # 1=Parking, 0=Exit
        cv2.createTrackbar("ExitDir", self.win_name, 0, 1, self.on_exit)  # 0/1

        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        # ===== 차량 장애물(첫 번째 draw_static_map의 사각형과 동일) =====
        self.car_x1 = 200 + self.off_x
        self.car_y1 = 180 + self.off_y
        self.car_x2 = 400 + self.off_x
        self.car_y2 = 540 + self.off_y

        car_cx = (self.car_x1 + self.car_x2) / 2.0
        car_rear_y = self.car_y2

        # ===== 경로 상태 =====
        self.dynamic_obstacles = []   # [(x,y,r), ...]

        self.parking_mode = True
        self.stage = 0
        self.goal_idx = 0
        self.exit_choice = 0
        self.goal_selected = False
        self.path = []

        # ===== 목표(예시) =====
        # 각도는 "map 기준 deg"(0=+x, 90=+y(아래), -90=위)
        self.goals = [
            [(car_cx - 120, car_rear_y + 160, -90), (car_cx + 120, car_rear_y + 160, -90)],  # S0 후보 2개
            [(car_cx,       car_rear_y + 90,  -90)],                                          # S1 정렬
            [(car_cx,       car_rear_y + 30,  -90)],                                          # S2 최종
        ]

        self.exit_goals = [
            [(car_cx,       car_rear_y + 140, -90)],                                          # S0 후진
            [(car_cx - 220, self.off_y + 400, None), (car_cx + 220, self.off_y + 400, None)], # S1 경유
            [(car_cx - 220, self.off_y + 300, None), (car_cx + 220, self.off_y + 300, None)], # S2 최종(ExitDir)
        ]

        # 시작 프레임 로드
        self.on_frame_change(INIT_FRAME)

    # -------------------------
    # Trackbar callbacks
    # -------------------------
    def on_frame_change(self, v):
        self.cap_rear.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_rear = self.cap_rear.read()
        _, self.curr_left = self.cap_left.read()

    def on_alpha(self, v):
        self.alpha = max(0.01, v / 100.0)

    def on_dist_gain(self, v):
        self.dist_gain = max(0.01, v / 100.0)

    def set_cam(self, cam_key, key, val):
        self.cams[cam_key][key] = float(val)

    # -------------------------
    # Path UI callbacks
    # -------------------------
    def on_mode(self, v):
        self.parking_mode = (v == 1)
        self.stage, self.goal_idx = 0, 0
        self.goal_selected = False
        self.path = []

    def on_exit(self, v):
        self.exit_choice = int(v)
        # 출차 방향 바꾸면 경로 재계획
        self.path = []

    def mouse_callback(self, event, x, y, flags, param):
        """맵 창 클릭으로 장애물 추가/제거"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dynamic_obstacles.append((x, y, 30))
            if self.is_initialized:
                self.update_path()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (ox, oy, r) in enumerate(self.dynamic_obstacles):
                if math.hypot(ox - x, oy - y) < r:
                    self.dynamic_obstacles.pop(i)
                    if self.is_initialized:
                        self.update_path()
                    break

    # -------------------------
    # Draw map (static + obstacles + goals)
    # -------------------------
    def draw_static_map(self, img):
        step = int(20 * self.map_scale * 2)
        for x in range(0, self.grid_w + 1, step):
            c = (45, 45, 45) if x % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x + x, self.off_y),
                     (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (45, 45, 45) if y % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x, self.off_y + y),
                     (self.off_x + self.grid_w, self.off_y + y), c, 1)

        # 차량(고정 장애물)
        cv2.rectangle(img, (self.car_x1, self.car_y1), (self.car_x2, self.car_y2), (35, 35, 45), -1)
        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h), (180, 180, 180), 2)

        # 카메라
        for cfg in self.cams.values():
            cp = tuple(cfg["pos_px"].astype(int))
            cv2.circle(img, cp, 7, cfg["color"], -1)
            cv2.putText(img, cfg["name"], (cp[0]-25, cp[1]+25),
                        0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 동적 장애물
        for ox, oy, r in self.dynamic_obstacles:
            cv2.circle(img, (int(ox), int(oy)), int(r), (0, 0, 150), -1)
            cv2.circle(img, (int(ox), int(oy)), int(r), (0, 0, 255), 2)

        # 목표 표시
        self.draw_goals(img)

    def draw_goals(self, img):
        goals = self.goals if self.parking_mode else self.exit_goals

        for si, stage_goals in enumerate(goals):
            for gi, g in enumerate(stage_goals):
                gp = (int(g[0]), int(g[1]))
                is_curr = (si == self.stage and gi == self.goal_idx)

                # 출차 마지막 단계는 ExitDir 선택 표시
                if (not self.parking_mode) and (si == len(goals) - 1):
                    is_curr = (si == self.stage and gi == self.exit_choice)

                col = (0, 255, 0) if is_curr else (100, 100, 100)
                cv2.circle(img, gp, 10, col, -1 if is_curr else 2)
                cv2.putText(img, f"S{si}", (gp[0]-8, gp[1]-15), 0, 0.4, col, 1)

                if g[2] is not None:
                    ax = int(gp[0] + 25 * math.cos(math.radians(g[2])))
                    ay = int(gp[1] + 25 * math.sin(math.radians(g[2])))
                    cv2.arrowedLine(img, gp, (ax, ay), (150, 150, 255), 2, tipLength=0.4)

        # 상태 텍스트
        mode_txt = "PARK" if self.parking_mode else "EXIT"
        cv2.putText(img, f"MODE:{mode_txt}  Stage:{self.stage}  GoalIdx:{self.goal_idx}  ExitDir:{self.exit_choice}",
                    (10, 20), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # -------------------------
    # Angle helpers
    # -------------------------
    @staticmethod
    def compass_deg_to_map_rad(compass_deg):
        # compass: 0=북, 90=동 -> map: 0=+x, 90=+y(아래)
        mdeg = (compass_deg + 270.0) % 360.0
        return math.radians(mdeg)

    def marker_to_center(self, marker_pos_px, heading_map_rad, marker_id):
        """
        규칙:
          - ID 0 = 앞(front) 마커: 중심은 '뒤'로 half_len
          - ID 1 = 뒤(rear) 마커: 중심은 '앞'으로 half_len
        """
        dx = self.half_len_px * math.cos(heading_map_rad)
        dy = self.half_len_px * math.sin(heading_map_rad)

        if marker_id == 0:   # front marker
            return marker_pos_px - np.array([dx, dy], dtype=np.float32)
        else:                # rear marker (id==1)
            return marker_pos_px + np.array([dx, dy], dtype=np.float32)

    # -------------------------
    # Per-camera solvePnP (ID0/ID1 only)
    # -------------------------
    def process_camera_all_markers(self, frame, cam_key, monitor_frame):
        cfg = self.cams[cam_key]
        corners, ids, _ = detector.detectMarkers(frame)
        results = []

        if ids is None:
            return results

        cv2.aruco.drawDetectedMarkers(monitor_frame, corners, ids)

        for i in range(len(ids)):
            mid = int(ids[i][0])
            if mid not in (0, 1):
                continue  # 반드시 앞=0, 뒤=1만 사용

            c = corners[i].reshape(4, 2)

            undist = cv2.fisheye.undistortPoints(
                corners[i].reshape(-1, 1, 2),
                K, D, P=K
            )

            ok, rvec, tvec = cv2.solvePnP(obj_points, undist, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3)
            dist_m = float(np.linalg.norm(tvec))

            dh_m = abs(cfg["h_cm"] - self.marker_h_cm) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * self.dist_gain

            bearing_rad = math.atan2(tvec[0], tvec[2])
            bearing_deg = math.degrees(bearing_rad)

            ray_deg = cfg["map_angle_deg"] + cfg["yaw_trim_deg"] + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cfg["pos_px"] + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            # yaw 계산(사용자 수식) + ID1이면 180° flip
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
            raw_yaw = math.atan2(-rmat[2, 0], sy) * 180.0 / math.pi

            current_total = (raw_yaw * cfg["sens"]) + cfg["install_angle"]
            final_yaw_compass = current_total - cfg["install_offset"]

            if mid == 1:
                final_yaw_compass = normalize_deg_0_360(final_yaw_compass + 180.0)

            heading_map_rad = self.compass_deg_to_map_rad(final_yaw_compass)

            # marker -> center 변환
            center_pos = self.marker_to_center(marker_pos, heading_map_rad, mid)

            # 가중치
            cx = float(np.mean(c[:, 0]))
            rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            weight = float(max(0.05, w_center * w_dist))

            # 모니터 표시 (축 경고가 싫으면 이 라인만 주석 처리)
            cv2.drawFrameAxes(monitor_frame, K, None, rvec, tvec, 0.07)
            bx, by = int(c[0][0]), int(c[0][1])
            cv2.putText(monitor_frame,
                        f"{cfg['name']} ID:{mid} yaw:{final_yaw_compass:6.1f} (w:{weight:.2f})",
                        (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            results.append({
                "marker_id": mid,
                "marker_pos": marker_pos,
                "center_pos": center_pos,
                "heading": heading_map_rad,
                "weight": weight,
                "cam_key": cam_key,
                "dbg": {
                    "ground_cm": ground_cm,
                    "bearing_deg": bearing_deg,
                    "ray_deg": ray_deg,
                    "final_yaw_compass": final_yaw_compass
                }
            })

        return results

    # -------------------------
    # Draw wheelchair (colored)
    # -------------------------
    def render_wheelchair(self, m_map, center_pos, heading,
                          body_color=(0, 255, 0), front_color=(0, 0, 255),
                          thickness=2, label=None):
        w_px = (self.wc_w_cm * self.map_scale) / 2.0
        l_px = (self.wc_l_cm * self.map_scale) / 2.0

        base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]], dtype=np.float32)
        rot_m = np.array([[math.cos(heading), -math.sin(heading)],
                          [math.sin(heading),  math.cos(heading)]], dtype=np.float32)
        pts = (base_pts @ rot_m.T) + center_pos

        cv2.polylines(m_map, [pts.astype(np.int32)], True, body_color, thickness, cv2.LINE_AA)
        cv2.line(m_map, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), front_color, thickness + 1)

        cv2.arrowedLine(
            m_map,
            tuple(center_pos.astype(int)),
            (int(center_pos[0] + 45 * math.cos(heading)),
             int(center_pos[1] + 45 * math.sin(heading))),
            body_color, thickness
        )

        if label is not None:
            wx, wy = int(center_pos[0]), int(center_pos[1])
            cv2.putText(m_map, label, (wx + 10, wy + 15), 0, 0.5, body_color, 2, cv2.LINE_AA)

    # -------------------------
    # Fuse per-cam estimates
    # -------------------------
    def fuse_estimates(self, det_list):
        if len(det_list) == 0:
            return None

        total_w = sum(d["weight"] for d in det_list)
        if total_w <= 1e-6:
            return None

        avg_center = sum(d["center_pos"] * d["weight"] for d in det_list) / total_w
        avg_sin = sum(math.sin(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_cos = sum(math.cos(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_heading = math.atan2(avg_sin, avg_cos)

        return avg_center, avg_heading, total_w

    # =========================================================
    #                 PATH PLANNING (A*)
    # =========================================================
    def is_obstacle(self, px, py):
        # 휠체어 안전 반경(대략)
        safe_margin = (self.wc_w_cm * self.map_scale / 2.0) + 30.0

        # 차량 장애물(마진 포함)
        if (self.car_x1 - safe_margin) <= px <= (self.car_x2 + safe_margin) and \
           (self.car_y1 - safe_margin) <= py <= (self.car_y2 + safe_margin):
            return True

        # 동적 장애물
        for ox, oy, r in self.dynamic_obstacles:
            if math.hypot(px - ox, py - oy) < (r + safe_margin):
                return True

        return False

    def simplify_path(self, path, epsilon=20.0):
        if len(path) < 3:
            return path

        pts = np.array(path, dtype=np.float32)

        def point_line_dist(p, a, b):
            if np.allclose(a, b):
                return np.linalg.norm(p - a)
            return abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = point_line_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax, idx = d, i

        if dmax > epsilon:
            left = self.simplify_path(path[:idx+1], epsilon)
            right = self.simplify_path(path[idx:], epsilon)
            return left[:-1] + right
        return [path[0], path[-1]]

    def interpolate_path(self, path, interval=30.0):
        if len(path) < 2:
            return path
        new_path = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i], dtype=np.float32)
            p2 = np.array(path[i+1], dtype=np.float32)
            dist = float(np.linalg.norm(p2 - p1))
            new_path.append(path[i])
            if dist > interval:
                n = int(dist // interval)
                for j in range(1, n + 1):
                    t = j / (n + 1)
                    ip = p1 * (1 - t) + p2 * t
                    new_path.append([float(ip[0]), float(ip[1])])
        new_path.append(path[-1])
        return new_path

    def astar(self, start, goal):
        sn = (int(start[0]), int(start[1]))
        gn = (int(goal[0]), int(goal[1]))

        # 시작이 장애물이라도 그냥 직선 반환(안전장치)
        if self.is_obstacle(*sn):
            return [[start[0], start[1]], [goal[0], goal[1]]]

        # 경사각 제한(수직 위주)
        ALLOWED_SLOPE = math.radians(20)
        SLOPE_PENALTY_WEIGHT = 200.0
        ROTATION_PENALTY = 100.0

        open_l = []
        heapq.heappush(open_l, (0.0, sn, (0, 0)))
        came = {}
        g_s = {sn: 0.0}

        moves = [(0, 12), (0, -12), (12, 0), (-12, 0),
                 (9, 9), (9, -9), (-9, 9), (-9, -9)]

        while open_l:
            _, curr, prev_dir = heapq.heappop(open_l)

            if math.dist(curr, gn) < 25:
                res = [list(curr)]
                while curr in came:
                    curr = came[curr]
                    res.append(list(curr))
                res = res[::-1]
                res = self.simplify_path(res, epsilon=20.0)
                res = self.interpolate_path(res, interval=30.0)
                return res

            for dx, dy in moves:
                nb = (curr[0] + dx, curr[1] + dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h):
                    continue
                if self.is_obstacle(nb[0], nb[1]):
                    continue

                move_cost = math.dist(curr, nb)

                slope_penalty = 0.0
                if dx != 0:
                    current_slope = math.atan2(abs(dx), abs(dy) if abs(dy) > 1e-6 else 1e-6)
                    if current_slope > ALLOWED_SLOPE:
                        slope_penalty = SLOPE_PENALTY_WEIGHT * (current_slope / (math.pi/2))

                rot_penalty = ROTATION_PENALTY if (prev_dir != (0, 0) and prev_dir != (dx, dy)) else 0.0

                tg = g_s[curr] + move_cost + slope_penalty + rot_penalty

                if nb not in g_s or tg < g_s[nb]:
                    came[nb] = curr
                    g_s[nb] = tg
                    f_score = tg + math.dist(nb, gn) * 1.5
                    heapq.heappush(open_l, (f_score, nb, (dx, dy)))

        # 실패 시 직선
        return [[start[0], start[1]], [goal[0], goal[1]]]

    # -------------------------
    # Goal / Stage logic
    # -------------------------
    def get_goal(self):
        goals = self.goals if self.parking_mode else self.exit_goals

        # 출차 마지막 stage는 ExitDir로 선택
        if (not self.parking_mode) and (self.stage == len(goals) - 1):
            g = goals[self.stage][self.exit_choice]
            return (g[0], g[1]), g[2]

        g = goals[self.stage][self.goal_idx]
        return (g[0], g[1]), g[2]

    def select_nearest(self, pos):
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_selected or self.stage != 0:
            return
        dists = [math.dist(pos, g[0:2]) for g in goals[0]]
        self.goal_idx = int(dists.index(min(dists)))
        self.goal_selected = True

    def check_reached(self, pos):
        gpos, gang_deg = self.get_goal()
        dist = math.dist(pos, gpos)
        if dist < 20:
            if gang_deg is None:
                return True
            # heading_angle은 rad, gang_deg는 deg(map기준)
            target = math.radians(gang_deg)
            diff = abs(math.atan2(math.sin(target - self.heading_angle),
                                  math.cos(target - self.heading_angle)))
            return diff < math.radians(20)
        return False

    def advance(self):
        goals = self.goals if self.parking_mode else self.exit_goals

        # 출차(stage==1)에서는 exit_choice에 따라 경유를 자동 선택하고 싶으면 여기서 로직 추가 가능
        if self.goal_idx < len(goals[self.stage]) - 1:
            self.goal_idx += 1
        elif self.stage < len(goals) - 1:
            self.stage += 1
            self.goal_idx = 0

        # stage 전환 시 경로 초기화
        self.path = []
        self.goal_selected = False

    def update_path(self):
        if not self.is_initialized:
            return

        center = np.array([float(self.center_pos[0]), float(self.center_pos[1])], dtype=np.float32)
        gpos, _ = self.get_goal()

        # stage0에서 후보 goal 중 가까운 것 선택
        if self.stage == 0:
            self.select_nearest(center)

        # 도착 체크
        if self.check_reached(center):
            self.advance()

        gpos, _ = self.get_goal()

        # 재계획 조건
        need_replan = False

        if not self.path or len(self.path) < 2:
            need_replan = True
        else:
            # 1) 경로 위 장애물 생겼나(샘플링 체크)
            for i in range(len(self.path) - 1):
                p1 = np.array(self.path[i], dtype=np.float32)
                p2 = np.array(self.path[i + 1], dtype=np.float32)
                for t in (0.3, 0.6, 0.9):
                    pt = p1 * (1 - t) + p2 * t
                    if self.is_obstacle(pt[0], pt[1]):
                        need_replan = True
                        break
                if need_replan:
                    break

            # 2) 경로 이탈(가장 가까운 선분까지 거리)
            if not need_replan:
                min_d = float("inf")
                for i in range(len(self.path) - 1):
                    p1 = np.array(self.path[i], dtype=np.float32)
                    p2 = np.array(self.path[i + 1], dtype=np.float32)
                    v = p2 - p1
                    w = center - p1
                    l2 = float(np.dot(v, v))
                    if l2 < 1e-6:
                        d = float(np.linalg.norm(center - p1))
                    else:
                        t = max(0.0, min(1.0, float(np.dot(w, v) / l2)))
                        proj = p1 + t * v
                        d = float(np.linalg.norm(center - proj))
                    min_d = min(min_d, d)
                if min_d > 70:
                    need_replan = True

        if need_replan:
            self.path = self.astar(center, gposust = (gpos[0], gpos[1])) if False else self.astar(center, gpos)
        else:
            # 웨이포인트 진행(첫 점 지나쳤으면 pop)
            if len(self.path) > 1:
                p1 = np.array(self.path[0], dtype=np.float32)
                p2 = np.array(self.path[1], dtype=np.float32)
                v_path = p2 - p1
                v_wc = center - p1
                dist_to_p1 = float(np.linalg.norm(center - p1))
                dot = float(np.dot(v_path, v_wc))
                if dist_to_p1 < 25 or dot > 0:
                    if len(self.path) > 2:
                        self.path.pop(0)

    def draw_path(self, img):
        if len(self.path) < 2 or not self.is_initialized:
            return

        pts = np.array(self.path, np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        # 방향 안내(현재 중심 -> 다음 웨이포인트)
        center = np.array([float(self.center_pos[0]), float(self.center_pos[1])], dtype=np.float32)
        target = np.array(self.path[1], dtype=np.float32) if len(self.path) > 1 else np.array(self.path[-1], dtype=np.float32)

        dx, dy = float(target[0] - center[0]), float(target[1] - center[1])
        target_yaw = math.atan2(dy, dx)

        yaw_err = math.degrees(math.atan2(math.sin(target_yaw - self.heading_angle),
                                          math.cos(target_yaw - self.heading_angle)))

        pivot = (int(center[0]), int(center[1]))
        cv2.ellipse(img, pivot, (45, 45), 0,
                    -math.degrees(self.heading_angle), -math.degrees(target_yaw),
                    (0, 200, 255) if yaw_err > 0 else (255, 150, 0), 2)

        cv2.putText(img, f"Rot: {yaw_err:+.1f}deg", (pivot[0] + 50, pivot[1] - 70),
                    0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        play = False

        while True:
            if play:
                r0, self.curr_rear = self.cap_rear.read()
                r1, self.curr_left = self.cap_left.read()
                if not r0 or not r1:
                    self.on_frame_change(0)
                    continue
                curr_pos = int(self.cap_rear.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Frame", self.win_name, min(curr_pos, self.total_frames - 1))
            else:
                target = cv2.getTrackbarPos("Frame", self.win_name)
                self.on_frame_change(target)

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)

            mon_rear = self.curr_rear.copy() if self.curr_rear is not None else None
            mon_left = self.curr_left.copy() if self.curr_left is not None else None

            detected = []
            if self.curr_rear is not None and mon_rear is not None:
                detected += self.process_camera_all_markers(self.curr_rear, "rear", mon_rear)
            if self.curr_left is not None and mon_left is not None:
                detected += self.process_camera_all_markers(self.curr_left, "left", mon_left)

            # 레이/마커 표시
            for d in detected:
                cfg = self.cams[d["cam_key"]]
                cp = tuple(cfg["pos_px"].astype(int))
                mp = tuple(d["marker_pos"].astype(int))

                dist_px = int(d["dbg"]["ground_cm"] * self.map_scale)
                ray_deg = d["dbg"]["ray_deg"]

                cv2.ellipse(m_map, cp, (dist_px, dist_px), 0, ray_deg - 5, ray_deg + 5,
                            cfg["color"], 2, cv2.LINE_AA)
                cv2.line(m_map, cp, mp, cfg["color"], 1, cv2.LINE_AA)
                cv2.circle(m_map, mp, 4, (255, 255, 0), -1)
                cv2.putText(m_map, f"ID{d['marker_id']}", (mp[0] + 6, mp[1] - 6),
                            0, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

            # 카메라별 휠체어(파랑/빨강)
            rear_list = [d for d in detected if d["cam_key"] == "rear"]
            left_list = [d for d in detected if d["cam_key"] == "left"]

            rear_est = self.fuse_estimates(rear_list)
            left_est = self.fuse_estimates(left_list)

            if rear_est is not None:
                r_center, r_head, r_w = rear_est
                self.render_wheelchair(m_map, r_center, r_head,
                                       body_color=(255, 0, 0), front_color=(255, 255, 255),
                                       thickness=2, label=f"REAR (w:{r_w:.2f})")

            if left_est is not None:
                l_center, l_head, l_w = left_est
                self.render_wheelchair(m_map, l_center, l_head,
                                       body_color=(0, 0, 255), front_color=(255, 255, 255),
                                       thickness=2, label=f"LEFT (w:{l_w:.2f})")

            # 최종 통합 + 스무딩
            if len(detected) > 0:
                total_w = sum(x["weight"] for x in detected)
                avg_center = sum(x["center_pos"] * x["weight"] for x in detected) / total_w
                avg_sin = sum(math.sin(x["heading"]) * x["weight"] for x in detected) / total_w
                avg_cos = sum(math.cos(x["heading"]) * x["weight"] for x in detected) / total_w
                avg_heading = math.atan2(avg_sin, avg_cos)

                if play and self.is_initialized:
                    self.center_pos = self.center_pos * (1 - self.alpha) + avg_center * self.alpha
                    diff = (avg_heading - self.heading_angle + math.pi) % (2 * math.pi) - math.pi
                    self.heading_angle += diff * self.alpha
                else:
                    self.center_pos = avg_center
                    self.heading_angle = avg_heading
                    self.is_initialized = True

            # ===== 경로 업데이트(통합 중심/헤딩 기준) =====
            if self.is_initialized:
                self.update_path()
                self.draw_path(m_map)

            # 통합 휠체어(초록)
            if self.is_initialized:
                self.render_wheelchair(m_map, self.center_pos, self.heading_angle,
                                       body_color=(0, 255, 0), front_color=(0, 255, 255),
                                       thickness=3, label="FUSED")

            # 도움말
            cv2.putText(m_map, "L-Click:Add Obstacle | R-Click:Remove | C:Clear | SPACE:Play/Pause | Q/ESC:Quit",
                        (10, self.map_h - 10), 0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(self.win_name, m_map)

            m0 = cv2.resize(mon_rear, (640, 360)) if mon_rear is not None else np.zeros((360, 640, 3), np.uint8)
            m1 = cv2.resize(mon_left, (640, 360)) if mon_left is not None else np.zeros((360, 640, 3), np.uint8)
            cv2.imshow("Monitor", np.hstack([m0, m1]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.path = []
            elif key == ord('q') or key == 27:
                break

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    IntegratedWheelchairMapTracker().run()
