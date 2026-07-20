# angle_focus.py
import cv2
import numpy as np
import math

# ==========================================
# 1) 공통 Intrinsic (사용자 제공)
# ==========================================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE_M = 0.25  # 25cm = 0.25m

# ==========================================
# ArUco detector (튜닝 포함)
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_params = cv2.aruco.DetectorParameters()

# ✅ 작은/기울어진/조명변화 마커에 좀 더 강하게
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 30
aruco_params.cornerRefinementMinAccuracy = 0.1

aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 53
aruco_params.adaptiveThreshWinSizeStep = 4

aruco_params.minMarkerPerimeterRate = 0.01
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.polygonalApproxAccuracyRate = 0.03

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


def ang_diff_rad(a, b):
    """a-b 를 [-pi, pi]"""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class IntegratedWheelchairMapTracker:
    def __init__(self):
        # ====== 맵/물리 ======
        self.marker_h_cm_default = 72.0

        # ✅ ID별 마커 높이
        self.marker_h_cm_by_id = {
            0: 70.0,  # id0
            1: 56.0   # id1
        }

        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5
        self.off_x, self.off_y = 200, 150

        self.wc_w_cm, self.wc_l_cm = 57.0, 100.0
        self.half_len_px = (self.wc_l_cm / 2.0) * self.map_scale

        # 상태(중심 기준)
        self.center_pos = None
        self.heading_angle = 0.0  # map rad
        self.is_initialized = False

        # lost-hold
        self.lost = 0
        self.max_lost = 8

        # 품질 파라미터(트랙바로 조절)
        self.min_area_px2 = 120
        self.edge_gate_px = 30
        self.max_reproj_err_px = 6.0
        self.max_outlier_deg = 35
        self.flip_guard = True

        # 스무딩
        self.alpha = 0.25
        self.smooth_pause = False

        # 거리 보정
        self.dist_gain = 1.88

        # 영상
        self.cap_rear = cv2.VideoCapture("test_rear.mp4")
        self.cap_left = cv2.VideoCapture("test_left.mp4")
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

        # ==========================================
        # UI (초기값: 사용자 스샷 기준)
        # ==========================================
        self.win_name = "Angle Focus (ID0/ID1 same mount) - FUSED"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        INIT_FRAME = 1478

        INIT_SMOOTH = 25           # 0~100 (%)
        INIT_SMOOTH_PAUSE = 0      # 0/1
        INIT_DISTGAIN = 188        # 1.88

        INIT_REAR_MAPYAW = 93
        INIT_LEFT_MAPYAW = 98
        INIT_REAR_SENS_X10 = 16
        INIT_LEFT_SENS_X10 = 16
        INIT_REAR_INSTANGLE = 0
        INIT_LEFT_INSTANGLE = 113
        INIT_REAR_INSTOFFSET_X10 = 0
        INIT_LEFT_INSTOFFSET_X10 = 508

        INIT_MINAREA = self.min_area_px2
        INIT_EDGEGATE = self.edge_gate_px
        INIT_MAXERR_X10 = int(self.max_reproj_err_px * 10)
        INIT_MAXOUTLIER = self.max_outlier_deg
        INIT_HOLD = self.max_lost
        INIT_FLIPGUARD = 1

        INIT_FRAME = int(max(0, min(INIT_FRAME, self.total_frames - 1)))

        # 내부 변수 동기화
        self.alpha = max(0.01, INIT_SMOOTH / 100.0)
        self.smooth_pause = (INIT_SMOOTH_PAUSE == 1)
        self.dist_gain = max(0.01, INIT_DISTGAIN / 100.0)

        self.cams["rear"]["yaw_trim_deg"] = float(INIT_REAR_MAPYAW - 90)
        self.cams["left"]["yaw_trim_deg"] = float(INIT_LEFT_MAPYAW - 90)
        self.cams["rear"]["sens"] = INIT_REAR_SENS_X10 / 10.0
        self.cams["left"]["sens"] = INIT_LEFT_SENS_X10 / 10.0
        self.cams["rear"]["install_angle"] = float(INIT_REAR_INSTANGLE)
        self.cams["left"]["install_angle"] = float(INIT_LEFT_INSTANGLE)
        self.cams["rear"]["install_offset"] = INIT_REAR_INSTOFFSET_X10 / 10.0
        self.cams["left"]["install_offset"] = INIT_LEFT_INSTOFFSET_X10 / 10.0

        self.min_area_px2 = int(INIT_MINAREA)
        self.edge_gate_px = int(INIT_EDGEGATE)
        self.max_reproj_err_px = float(INIT_MAXERR_X10) / 10.0
        self.max_outlier_deg = int(INIT_MAXOUTLIER)
        self.max_lost = int(INIT_HOLD)
        self.flip_guard = (INIT_FLIPGUARD == 1)

        # 트랙바
        cv2.createTrackbar("Frame", self.win_name, INIT_FRAME, max(0, self.total_frames - 1), self.on_frame_change)
        cv2.createTrackbar("Smooth(%)", self.win_name, INIT_SMOOTH, 100, self.on_alpha)
        cv2.createTrackbar("SmoothPause", self.win_name, INIT_SMOOTH_PAUSE, 1, self.on_smooth_pause)
        cv2.createTrackbar("DistGain(%)", self.win_name, INIT_DISTGAIN, 250, self.on_dist_gain)

        cv2.createTrackbar("MinArea", self.win_name, INIT_MINAREA, 3000, self.on_min_area)
        cv2.createTrackbar("EdgeGate(px)", self.win_name, INIT_EDGEGATE, 200, self.on_edge_gate)
        cv2.createTrackbar("MaxErr(x0.1)", self.win_name, INIT_MAXERR_X10, 200, self.on_max_err)
        cv2.createTrackbar("MaxOutlier(deg)", self.win_name, INIT_MAXOUTLIER, 120, self.on_max_outlier)
        cv2.createTrackbar("HoldFrames", self.win_name, INIT_HOLD, 30, self.on_hold)
        cv2.createTrackbar("FlipGuard", self.win_name, INIT_FLIPGUARD, 1, self.on_flip_guard)

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

    def on_smooth_pause(self, v):
        self.smooth_pause = (v == 1)

    def on_dist_gain(self, v):
        self.dist_gain = max(0.01, v / 100.0)

    def on_min_area(self, v):
        self.min_area_px2 = max(0, int(v))

    def on_edge_gate(self, v):
        self.edge_gate_px = max(0, int(v))

    def on_max_err(self, v):
        self.max_reproj_err_px = max(0.1, float(v) / 10.0)

    def on_max_outlier(self, v):
        self.max_outlier_deg = max(0, int(v))

    def on_hold(self, v):
        self.max_lost = max(0, int(v))

    def on_flip_guard(self, v):
        self.flip_guard = (v == 1)

    def set_cam(self, cam_key, key, val):
        self.cams[cam_key][key] = float(val)

    # -------------------------
    # Draw map
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

        # 차량 영역
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y),
                      (400 + self.off_x, 540 + self.off_y), (35, 35, 45), -1)

        # 경계
        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h), (180, 180, 180), 2)

        # 카메라 표시
        for cfg in self.cams.values():
            cp = tuple(cfg["pos_px"].astype(int))
            cv2.circle(img, cp, 7, cfg["color"], -1)
            cv2.putText(img, cfg["name"], (cp[0] - 25, cp[1] + 25),
                        0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # -------------------------
    # Angle helpers
    # -------------------------
    @staticmethod
    def compass_deg_to_map_rad(compass_deg):
        # compass: 0=북, 90=동 -> map: 0=+x, 90=+y(아래)
        mdeg = (compass_deg + 270.0) % 360.0
        return math.radians(mdeg)

    def marker_to_center(self, marker_pos_px, heading_map_rad):
        """
        ✅ 핵심 수정:
        - id0/id1이 '같은 위치(뒤판)'에 붙어있다면, 중심은 항상 '앞으로 +L/2'
        - id1은 yaw만 +180으로 휠체어 헤딩으로 맞추고,
          center는 항상 marker + half_len로 통일하는게 안정적
        """
        dx = self.half_len_px * math.cos(heading_map_rad)
        dy = self.half_len_px * math.sin(heading_map_rad)
        return marker_pos_px + np.array([dx, dy], dtype=np.float32)

    def choose_flip_near_prev(self, h_rad):
        """h 와 h+pi 중 이전 heading에 더 가까운 것 선택"""
        if not self.is_initialized or self.center_pos is None:
            return h_rad
        h2 = (h_rad + math.pi) % (2 * math.pi)
        d1 = abs(ang_diff_rad(h_rad, self.heading_angle))
        d2 = abs(ang_diff_rad(h2, self.heading_angle))
        return h2 if d2 < d1 else h_rad

    def reproj_err(self, rvec, tvec, undist_pts):
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec.reshape(3, 1), K, None)
        proj = proj.reshape(-1, 2)
        und = undist_pts.reshape(-1, 2)
        e = np.mean(np.linalg.norm(proj - und, axis=1))
        return float(e)

    # -------------------------
    # Per-camera solvePnP (ID0/ID1 only)
    # -------------------------
    def process_camera_all_markers(self, frame_bgr, cam_key, monitor_frame):
        cfg = self.cams[cam_key]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        results = []

        if ids is None:
            return results

        cv2.aruco.drawDetectedMarkers(monitor_frame, corners, ids)

        H, W = gray.shape[:2]

        for i in range(len(ids)):
            mid = int(ids[i][0])
            if mid not in (0, 1):
                continue

            c = corners[i].reshape(4, 2).astype(np.float32)

            # (1) Edge gate
            if self.edge_gate_px > 0:
                minx, miny = np.min(c[:, 0]), np.min(c[:, 1])
                maxx, maxy = np.max(c[:, 0]), np.max(c[:, 1])
                eg = self.edge_gate_px
                if (minx < eg) or (miny < eg) or (maxx > (W - eg)) or (maxy > (H - eg)):
                    cv2.putText(monitor_frame, f"REJECT edge (ID{mid})", (20, 30 + 18 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    continue

            # (2) Area gate
            area = abs(cv2.contourArea(c))
            if area < self.min_area_px2:
                cv2.putText(monitor_frame, f"REJECT area {area:.0f} (ID{mid})", (20, 30 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # undistort points
            undist = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)

            # solvePnP: IPPE_SQUARE -> ITERATIVE
            ok, rvec, tvec = cv2.solvePnP(obj_points, undist, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(obj_points, undist, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3)

            # (3) reprojection err gate
            err = self.reproj_err(rvec, tvec, undist)
            if err > self.max_reproj_err_px:
                cv2.putText(monitor_frame, f"REJECT err {err:.1f}px (ID{mid})", (20, 30 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # 거리 계산 (tvec는 m)
            dist_m = float(np.linalg.norm(tvec))

            marker_h_cm = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cfg["h_cm"] - marker_h_cm) / 100.0

            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * self.dist_gain

            # bearing -> map ray
            bearing_rad = math.atan2(tvec[0], tvec[2])
            bearing_deg = math.degrees(bearing_rad)

            ray_deg = cfg["map_angle_deg"] + cfg["yaw_trim_deg"] + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cfg["pos_px"] + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            # yaw (사용자 수식) + ID1이면 180° flip (휠체어 헤딩으로 통일)
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
            raw_yaw = math.atan2(-rmat[2, 0], sy) * 180.0 / math.pi

            current_total = (raw_yaw * cfg["sens"]) + cfg["install_angle"]
            final_yaw_compass = current_total - cfg["install_offset"]
            if mid == 1:
                final_yaw_compass = normalize_deg_0_360(final_yaw_compass + 180.0)

            heading_map_rad = self.compass_deg_to_map_rad(final_yaw_compass)

            # flip guard (180 플립 억제)
            if self.flip_guard:
                heading_map_rad = self.choose_flip_near_prev(heading_map_rad)

            # outlier gate
            if self.is_initialized:
                if abs(ang_diff_rad(heading_map_rad, self.heading_angle)) > math.radians(self.max_outlier_deg):
                    cv2.putText(monitor_frame, f"REJECT outlier (ID{mid})", (20, 30 + 18 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    continue

            # ✅ marker -> center (항상 +)
            center_pos = self.marker_to_center(marker_pos, heading_map_rad)

            # weight: center proximity + err + area + dist
            cx = float(np.mean(c[:, 0]))
            rel_x = (cx - W / 2) / (W / 2)
            w_center = max(0.05, 1.0 - abs(rel_x))

            err_w = 1.0 / (1.0 + err * err)
            area_w = min(1.0, area / 1500.0)
            dist_w = 1.0 / (1.0 + ground_m / 2.0)

            weight = float(max(0.02, w_center * err_w * area_w * dist_w))

            # 모니터 표시
            cv2.drawFrameAxes(monitor_frame, K, None, rvec, tvec, 0.07)
            bx, by = int(c[0][0]), int(c[0][1])
            cv2.putText(
                monitor_frame,
                f"{cfg['name']} ID:{mid} yaw:{final_yaw_compass:6.1f} err:{err:.1f}px area:{area:.0f} w:{weight:.2f}",
                (bx, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

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
                    "final_yaw_compass": final_yaw_compass,
                    "marker_h_cm": marker_h_cm,
                    "reproj_err": err,
                    "area": area
                }
            })

        return results

    # -------------------------
    # Draw wheelchair
    # -------------------------
    def render_wheelchair(self, m_map, center_pos, heading,
                          body_color=(0, 255, 0), front_color=(0, 0, 255),
                          thickness=2, label=None):
        if center_pos is None:
            return
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
    # Fuse
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

        return avg_center.astype(np.float32), float(avg_heading), float(total_w)

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

            # 맵
            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)

            # 모니터
            mon_rear = self.curr_rear.copy() if self.curr_rear is not None else None
            mon_left = self.curr_left.copy() if self.curr_left is not None else None

            # detect
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

            # 카메라별 추정치
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

            # 최종 융합
            apply_smoothing = play or self.smooth_pause

            if len(detected) > 0:
                fused = self.fuse_estimates(detected)
                if fused is not None:
                    avg_center, avg_heading, total_w = fused

                    if apply_smoothing and self.is_initialized and self.center_pos is not None:
                        self.center_pos = self.center_pos * (1.0 - self.alpha) + avg_center * self.alpha
                        diff = ang_diff_rad(avg_heading, self.heading_angle)
                        self.heading_angle = self.heading_angle + diff * self.alpha
                    else:
                        self.center_pos = avg_center
                        self.heading_angle = avg_heading
                        self.is_initialized = True

                    self.lost = 0
            else:
                # ✅ hold
                if self.is_initialized:
                    self.lost += 1
                    if self.lost > self.max_lost:
                        self.is_initialized = False
                        self.center_pos = None

            # 통합 휠체어
            if self.is_initialized and self.center_pos is not None:
                self.render_wheelchair(m_map, self.center_pos, self.heading_angle,
                                       body_color=(0, 255, 0), front_color=(0, 255, 255),
                                       thickness=3, label="FUSED")

                # 각도 표시(북=0 기준)
                compass_deg = (math.degrees(self.heading_angle) + 90.0) % 360.0
                cv2.putText(m_map, f"FUSED Angle(N=0): {compass_deg:.1f}deg",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

            # 상태 텍스트
            cv2.putText(m_map, f"Detections: {len(detected)} (rear:{len(rear_list)} left:{len(left_list)})  lost:{self.lost}/{self.max_lost}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            cv2.putText(m_map, f"alpha:{self.alpha:.2f}  smooth_pause:{int(self.smooth_pause)}  dist_gain:{self.dist_gain:.2f}  edge:{self.edge_gate_px}  minA:{self.min_area_px2}  maxErr:{self.max_reproj_err_px:.1f}  outlier:{self.max_outlier_deg}",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.putText(m_map, "SPACE: Play/Pause | q/ESC: Quit",
                        (20, self.map_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            # 출력
            cv2.imshow(self.win_name, m_map)

            m0 = cv2.resize(mon_rear, (640, 360)) if mon_rear is not None else np.zeros((360, 640, 3), np.uint8)
            m1 = cv2.resize(mon_left, (640, 360)) if mon_left is not None else np.zeros((360, 640, 3), np.uint8)
            cv2.imshow("Monitor", np.hstack([m0, m1]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q') or key == 27:
                break

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    IntegratedWheelchairMapTracker().run()
