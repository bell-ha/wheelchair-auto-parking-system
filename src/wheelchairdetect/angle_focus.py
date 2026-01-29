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

        # 기본값(어차피 아래 UI 초기값으로 덮어씀)
        self.alpha = 0.75
        self.dist_gain = 1.00

        # ==========================================
        # UI (초기값: 스샷 기준)
        # ==========================================
        self.win_name = "Integrated Wheelchair Tracker (ID0 front / ID1 rear)"
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

        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y),
                      (400 + self.off_x, 540 + self.off_y), (35, 35, 45), -1)
        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h), (180, 180, 180), 2)

        for cfg in self.cams.values():
            cp = tuple(cfg["pos_px"].astype(int))
            cv2.circle(img, cp, 7, cfg["color"], -1)
            cv2.putText(img, cfg["name"], (cp[0]-25, cp[1]+25),
                        0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

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

            # 통합 휠체어(초록)
            if self.is_initialized:
                self.render_wheelchair(m_map, self.center_pos, self.heading_angle,
                                       body_color=(0, 255, 0), front_color=(0, 255, 255),
                                       thickness=3, label="FUSED")

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
