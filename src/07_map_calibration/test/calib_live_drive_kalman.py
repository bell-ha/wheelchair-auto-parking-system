#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teleop_v31_kf_per_camera_4points_with_hide.py
#
# 요구사항:
# - 4점만 표시:
#     rear_before (PnP only)
#     rear_after  (v3.1 measurement -> rear Kalman filter output)
#     left_before (PnP only)
#     left_after  (v3.1 measurement -> left Kalman filter output)
#
# - 카메라가 오래 못 보면(Timeout) / 끊긴 뒤 너무 많이 이동하면(Distance gate)
#   해당 카메라의 "after(KF)"를 화면에서 숨김
#   (다시 마커 잡히면 바로 나타남)
#
# - 조종:
#   OpenCV MAP 창 포커스에서 WASD/Space/X/Q
#   UDP send 부호는 control.py 방식 유지:
#     A=+w, D=-w
#   KF 예측에서는 화면 회전 방향 정렬을 위해 w만 부호 반전:
#     w_kf = -(w_send)/1000 [rad/s]
#
# Requires:
#   analyze/calib_params_ridge.csv

import json, socket, time, math, csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# =======================
# USER SETTINGS
# =======================
SERVER_IP = "172.25.244.144"
SERVER_PORT = 25001

SEND_HZ = 30
FWD_MM_S = 200
REV_MM_S = -200
YAW_MRAD_S = 300

CALIB_CSV = "analyze/calib_params_ridge.csv"
DIST_GAIN_FIXED = 0.90

# =======================
# HIDE POLICY (핵심)
# =======================
# 마커가 이 시간 이상 안 보이면 after(KF) 숨김
HIDE_TIMEOUT_S = 2.0
# 마지막으로 마커가 보였던 위치에서 이 거리(cm) 이상 벗어나면 after(KF) 숨김
HIDE_MAX_DRIFT_CM = 300.0


# =========================
# UDP helper (never crash)
# =========================
def udp_send(sock, addr, payload: dict) -> bool:
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), addr)
        return True
    except OSError:
        return False


# =========================
# Camera intrinsics
# =========================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

# =========================
# ArUco / PnP
# =========================
MARKER_SIZE_M = 0.25
OBJ_POINTS = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0]
], dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================
# Angle helpers
# =========================
def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi

def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(down)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)

def map_deg_to_south0_deg(map_deg: float) -> float:
    return wrap360(map_deg - 90.0)

def south0_deg_to_map_deg(south0_deg: float) -> float:
    return wrap360(south0_deg + 90.0)

def view_sym_from_rel(rel_deg: float) -> float:
    a = abs(rel_deg)
    return float(min(a, abs(180.0 - a)))  # 0~90


# =========================
# Config
# =========================
@dataclass
class CamCfg:
    key: str
    index: int
    pos_world_px: np.ndarray
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = DIST_GAIN_FIXED


# =========================
# v3.1 runtime
# =========================
FEAT_KEYS = ["bias", "g", "g2", "b", "b2", "v", "g*b", "reproj"]

def rayvec_from_south0(ray_ang_south0_deg: float):
    map_deg = south0_deg_to_map_deg(ray_ang_south0_deg)
    r = math.radians(map_deg)
    return math.cos(r), math.sin(r)

def from_ray_frame(e_par, e_perp, ray_ang_south0_deg):
    ux, uy = rayvec_from_south0(ray_ang_south0_deg)
    vx, vy = -uy, ux
    dx = e_par * ux + e_perp * vx
    dy = e_par * uy + e_perp * vy
    return dx, dy

def X_v3(ground_m, bearing_deg, view_sym_deg, reproj):
    g = float(ground_m)              # meters
    b = abs(float(bearing_deg))      # degrees
    v = float(view_sym_deg)          # degrees
    r = float(reproj)                # pixels
    return np.array([1.0, g, g*g, b, b*b, v, g*b, r], dtype=float)

def load_calib_params_v31(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing v3.1 params CSV: {csv_path}")
    params = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = row["cam"].strip().lower()
            def get(prefix):
                return np.array([float(row[f"{prefix}{k}"]) for k in FEAT_KEYS], dtype=float)
            wpar = get("wpar_")
            wper = get("wper_")
            wtheta = None
            if f"wtheta_{FEAT_KEYS[0]}" in row:
                wtheta = get("wtheta_")
            params[cam] = (wpar, wper, wtheta)
    return params

def apply_v31(cam_key: str, params, *,
              pred_x, pred_y, pred_ang_south0_deg,
              ground_m, bearing_deg, ray_ang_south0_deg,
              view_sym_deg, reproj):
    cam_key = cam_key.lower()
    if cam_key not in params:
        return float(pred_x), float(pred_y), float(pred_ang_south0_deg)

    wpar, wper, wtheta = params[cam_key]
    X = X_v3(ground_m, bearing_deg, view_sym_deg, reproj)

    epar = float(X @ wpar)
    eper = float(X @ wper)
    dx, dy = from_ray_frame(epar, eper, ray_ang_south0_deg)

    x2 = float(pred_x) + dx
    y2 = float(pred_y) + dy

    ang2 = float(pred_ang_south0_deg)
    if wtheta is not None:
        dth = float(X @ wtheta)
        ang2 = wrap360(ang2 + dth)
    return x2, y2, ang2


# =========================
# Camera-wise KF + visibility gate
# =========================
class CamKF:
    def __init__(self):
        self.x = np.zeros((3, 1), dtype=float)
        self.P = np.diag([60.0**2, 60.0**2, (20.0 * math.pi/180)**2]).astype(float)
        self.initialized = False

        self.q_xy = 10.0
        self.q_th = 12.0 * math.pi/180
        self.r_xy = 25.0
        self.r_th = 15.0 * math.pi/180

        # visibility gate state
        self.last_seen_t = None
        self.last_seen_pos = None  # np.array([x,y])
        self.visible = False

    def init_from_meas(self, mx, my, mth, now):
        self.x[:] = np.array([[mx], [my], [wrap_pi(mth)]], dtype=float)
        self.P[:] = np.diag([25.0**2, 25.0**2, (10.0 * math.pi/180)**2]).astype(float)
        self.initialized = True
        self.last_seen_t = now
        self.last_seen_pos = np.array([mx, my], dtype=float)
        self.visible = True

    def predict(self, v_cm_s, w_rad_s, dt):
        if dt <= 0.0 or (not self.initialized):
            return

        x = float(self.x[0, 0]); y = float(self.x[1, 0]); th = float(self.x[2, 0])
        x2 = x + v_cm_s * dt * math.cos(th)
        y2 = y + v_cm_s * dt * math.sin(th)
        th2 = wrap_pi(th + w_rad_s * dt)
        self.x[:] = np.array([[x2], [y2], [th2]], dtype=float)

        F = np.eye(3, dtype=float)
        F[0, 2] = -v_cm_s * dt * math.sin(th)
        F[1, 2] =  v_cm_s * dt * math.cos(th)

        Q = np.diag([(self.q_xy**2)*dt, (self.q_xy**2)*dt, (self.q_th**2)*dt]).astype(float)
        self.P = F @ self.P @ F.T + Q

    def update(self, mx, my, mth, now, quality_scale=1.0):
        if not self.initialized:
            self.init_from_meas(mx, my, mth, now)
            return

        z = np.array([[mx], [my], [wrap_pi(mth)]], dtype=float)
        H = np.eye(3, dtype=float)
        R = np.diag([(self.r_xy*quality_scale)**2, (self.r_xy*quality_scale)**2, (self.r_th*quality_scale)**2]).astype(float)

        y = z - self.x
        y[2, 0] = wrap_pi(float(y[2, 0]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2, 0] = wrap_pi(float(self.x[2, 0]))
        self.P = (np.eye(3, dtype=float) - K @ H) @ self.P

        # update last seen info
        self.last_seen_t = now
        self.last_seen_pos = np.array([mx, my], dtype=float)
        self.visible = True

    def refresh_visibility(self, now, timeout_s, max_d_cm):
        if self.last_seen_t is None or (not self.initialized):
            self.visible = False
            return

        if (now - self.last_seen_t) > timeout_s:
            self.visible = False
            return

        if self.last_seen_pos is not None:
            x, y, _ = self.get()
            dx = x - float(self.last_seen_pos[0])
            dy = y - float(self.last_seen_pos[1])
            if math.hypot(dx, dy) > max_d_cm:
                self.visible = False
                return

        self.visible = True

    def get(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])


# =========================
# App
# =========================
class App:
    def __init__(self):
        # canvas
        self.canvas_w, self.canvas_h = 2000, 2000
        self.draw_off_x, self.draw_off_y = 700, 500

        # coordinate alignment
        self.grid_world_origin = np.array([200.0, 150.0], dtype=np.float32)
        self.grid_w, self.grid_h = 600, 720
        gx, gy = float(self.grid_world_origin[0]), float(self.grid_world_origin[1])
        self.car_zone_world = ((200 + gx, 180 + gy), (400 + gx, 540 + gy))

        # marker geometry
        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # cams
        rear_base = np.array([301.4, 540.0], dtype=np.float32)
        left_base = np.array([200.0, 270.0], dtype=np.float32)
        self.cams = {
            "rear": CamCfg("rear", 0, rear_base + self.grid_world_origin, 105.5, 90.0, 1.6, 0.0, 0.0, yaw_trim_deg=3.0, dist_gain=DIST_GAIN_FIXED),
            "left": CamCfg("left", 1, left_base + self.grid_world_origin, 110.0, 157.0, 1.6, 113.0, 50.84, yaw_trim_deg=8.0, dist_gain=DIST_GAIN_FIXED),
        }

        # v3.1
        self.params_v31 = load_calib_params_v31(CALIB_CSV)
        print("[LOAD] v3.1 params:", list(self.params_v31.keys()))
        print("[INFO] dist_gain fixed =", DIST_GAIN_FIXED)
        print("[INFO] hide policy: timeout=%.1fs  drift=%.0fcm" % (HIDE_TIMEOUT_S, HIDE_MAX_DRIFT_CM))

        # cameras
        self.cap0 = cv2.VideoCapture(self.cams["rear"].index)
        self.cap1 = cv2.VideoCapture(self.cams["left"].index)
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            raise RuntimeError("Camera open failed. index(0/1) 확인")

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (SERVER_IP, SERVER_PORT)
        self.udp_ok = True
        self.last_udp_err_t = 0.0

        # teleop
        self.v_mm_s = 0
        self.w_mrad_s = 0
        self.last_key = "STOP"
        self.last_send = 0.0
        self.dt_send = 1.0 / float(SEND_HZ)

        # per-camera KF
        self.kf_rear = CamKF()
        self.kf_left = CamKF()

        self.last_t = time.time()

        # windows
        self.win_map = "MAP | 4 points + hide-after policy"
        self.win_mon = "MONITOR (rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)

    def close(self):
        try:
            udp_send(self.sock, self.addr, {"stop": True})
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.cap0.release()
            self.cap1.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def w2c(self, p_world: np.ndarray) -> np.ndarray:
        return p_world + np.array([self.draw_off_x, self.draw_off_y], dtype=np.float32)

    def marker_to_center_world(self, marker_pos_world: np.ndarray, heading_map_rad: float, marker_id: int) -> np.ndarray:
        offset_cm = float(self.center_offset_cm_by_id.get(marker_id, 23.0))
        dx = offset_cm * math.cos(heading_map_rad)
        dy = offset_cm * math.sin(heading_map_rad)
        sign = -1.0 if marker_id == 0 else +1.0
        return marker_pos_world + np.array([sign * dx, sign * dy], dtype=np.float32)

    def estimate_before_and_v31(self, frame_bgr, cam: CamCfg):
        """
        Returns:
          before: (center_world, heading_map)
          meas_v31: (center_world, heading_map)  # v3.1 output (measurement for KF)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return None

        H, W = frame_bgr.shape[:2]
        best, best_score = None, -1.0

        for i, mid_arr in enumerate(ids):
            mid = int(mid_arr[0])
            if mid not in (0, 1):
                continue

            und = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)
            ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            if float(tvec[2]) <= 0.01:
                continue

            dist_m = float(np.linalg.norm(tvec))
            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * cam.dist_gain

            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))

            ray_deg_map = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad_map = math.radians(ray_deg_map)

            marker_world = cam.pos_world_px + np.array([
                ground_cm * math.cos(ray_rad_map),
                ground_cm * math.sin(ray_rad_map)
            ], dtype=np.float32)

            # yaw -> compass -> heading map rad
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset
            yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)
            heading_before_map = compass_deg_to_map_rad(yaw_compass)

            center_before = self.marker_to_center_world(marker_world, heading_before_map, mid)

            # reproj
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            # v3.1 inputs
            heading_before_map_deg = wrap360(math.degrees(heading_before_map))
            pred_ang_before_south0 = map_deg_to_south0_deg(heading_before_map_deg)
            ray_ang_south0 = map_deg_to_south0_deg(wrap360(ray_deg_map))
            view_rel_deg = wrap180(pred_ang_before_south0 - ray_ang_south0)
            view_sym_deg = view_sym_from_rel(view_rel_deg)

            x2, y2, ang2_south0 = apply_v31(
                cam.key, self.params_v31,
                pred_x=float(center_before[0]),
                pred_y=float(center_before[1]),
                pred_ang_south0_deg=float(pred_ang_before_south0),
                ground_m=float(ground_m),
                bearing_deg=float(bearing_deg),
                ray_ang_south0_deg=float(ray_ang_south0),
                view_sym_deg=float(view_sym_deg),
                reproj=float(reproj_err),
            )
            center_v31 = np.array([x2, y2], dtype=np.float32)
            heading_v31_map = math.radians(south0_deg_to_map_deg(ang2_south0))

            # best marker selection
            cx = float(np.mean(corners[i].reshape(4, 2)[:, 0]))
            rel_x = (cx - W / 2) / (W / 2)
            score = (1.0 - abs(rel_x)) * (1.0 / (1.0 + ground_m))

            if score > best_score:
                best_score = score
                best = {
                    "before_center": center_before,
                    "before_heading": float(heading_before_map),
                    "meas_center": center_v31,
                    "meas_heading": float(heading_v31_map),
                }

        return best

    # ---- UI drawing ----
    def draw_grid(self, img, x0, y0, w, h, step, col_minor, col_major, major_step):
        for x in range(0, w + 1, step):
            col = col_major if (x % major_step) == 0 else col_minor
            cv2.line(img, (x0 + x, y0), (x0 + x, y0 + h), col, 1)
        for y in range(0, h + 1, step):
            col = col_major if (y % major_step) == 0 else col_minor
            cv2.line(img, (x0, y0 + y), (x0 + w, y0 + y), col, 1)

    def draw_static(self, canvas):
        step, major = 20, 100
        self.draw_grid(canvas, 0, 0, self.canvas_w - 1, self.canvas_h - 1, step, (25, 25, 25), (45, 45, 45), major)

        g0 = self.w2c(self.grid_world_origin).astype(int)
        gx, gy = int(g0[0]), int(g0[1])
        self.draw_grid(canvas, gx, gy, self.grid_w, self.grid_h, step, (45, 45, 45), (80, 80, 80), major)
        cv2.rectangle(canvas, (gx, gy), (gx + self.grid_w, gy + self.grid_h), (200, 200, 200), 2)

        (x0w, y0w), (x1w, y1w) = self.car_zone_world
        p0 = self.w2c(np.array([x0w, y0w], np.float32)).astype(int)
        p1 = self.w2c(np.array([x1w, y1w], np.float32)).astype(int)
        cv2.rectangle(canvas, tuple(p0), tuple(p1), (35, 35, 45), -1)

        for key, cam in self.cams.items():
            cp = self.w2c(cam.pos_world_px).astype(int)
            cv2.circle(canvas, tuple(cp), 6, (220, 220, 220), -1)
            cv2.putText(canvas, key, (int(cp[0]) + 8, int(cp[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    def draw_pose(self, canvas, center_world, heading_map_rad, color, label, radius, thickness, arrow_len_cm):
        c = self.w2c(center_world).astype(int)
        cv2.circle(canvas, tuple(c), radius, color, -1)
        end_world = center_world + np.array([arrow_len_cm * math.cos(heading_map_rad),
                                             arrow_len_cm * math.sin(heading_map_rad)], dtype=np.float32)
        e = self.w2c(end_world).astype(int)
        cv2.arrowedLine(canvas, tuple(c), tuple(e), color, thickness, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(canvas, label, (int(c[0]) + 10, int(c[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ---- control mapping (control.py 유지) ----
    def handle_key(self, k: int):
        if k == ord('w'):
            self.v_mm_s, self.w_mrad_s = FWD_MM_S, 0
            self.last_key = "W"
        elif k == ord('s'):
            self.v_mm_s, self.w_mrad_s = REV_MM_S, 0
            self.last_key = "S"
        elif k == ord('a'):
            self.v_mm_s, self.w_mrad_s = 0, +YAW_MRAD_S
            self.last_key = "A"
        elif k == ord('d'):
            self.v_mm_s, self.w_mrad_s = 0, -YAW_MRAD_S
            self.last_key = "D"
        elif k == ord('x') or k == 32:
            self.v_mm_s, self.w_mrad_s = 0, 0
            self.last_key = "STOP"
            self.udp_ok = udp_send(self.sock, self.addr, {"stop": True})

    def maybe_send(self):
        now = time.time()
        if (now - self.last_send) >= self.dt_send:
            ok = udp_send(self.sock, self.addr, {"v": int(self.v_mm_s), "w": int(self.w_mrad_s)})
            self.udp_ok = ok
            if (not ok) and (now - self.last_udp_err_t > 1.0):
                print(f"[WARN] UDP send failed (Host down?): {self.addr}")
                self.last_udp_err_t = now
            self.last_send = now

    def run(self):
        try:
            while True:
                now = time.time()
                dt = now - self.last_t
                self.last_t = now
                dt = max(0.0, min(dt, 0.2))

                # predict both KFs from command
                v_cm_s = self.v_mm_s / 10.0
                w_rad_s_send = (self.w_mrad_s / 1000.0)
                w_rad_s_kf = -w_rad_s_send  # 화면 방향 맞춤용 반전

                self.kf_rear.predict(v_cm_s, w_rad_s_kf, dt)
                self.kf_left.predict(v_cm_s, w_rad_s_kf, dt)

                ok0, fr0 = self.cap0.read()
                ok1, fr1 = self.cap1.read()

                rear = self.estimate_before_and_v31(fr0, self.cams["rear"]) if (ok0 and fr0 is not None) else None
                left = self.estimate_before_and_v31(fr1, self.cams["left"]) if (ok1 and fr1 is not None) else None

                # update each KF with its own measurement
                if rear is not None:
                    mx, my = float(rear["meas_center"][0]), float(rear["meas_center"][1])
                    mth = float(rear["meas_heading"])
                    self.kf_rear.update(mx, my, mth, now)

                if left is not None:
                    mx, my = float(left["meas_center"][0]), float(left["meas_center"][1])
                    mth = float(left["meas_heading"])
                    self.kf_left.update(mx, my, mth, now)

                # refresh visibility gates (hide policy)
                self.kf_rear.refresh_visibility(now, HIDE_TIMEOUT_S, HIDE_MAX_DRIFT_CM)
                self.kf_left.refresh_visibility(now, HIDE_TIMEOUT_S, HIDE_MAX_DRIFT_CM)

                # draw
                canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 15
                self.draw_static(canvas)

                # HUD/Legend
                cv2.putText(canvas, "Keys: W/A/S/D | Space/X=STOP | Q/ESC=QUIT",
                            (10, 25), 0, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"CMD(send): v={self.v_mm_s} mm/s, w={self.w_mrad_s} mrad/s (last={self.last_key})",
                            (10, 55), 0, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"HIDE: timeout={HIDE_TIMEOUT_S:.1f}s, drift>{HIDE_MAX_DRIFT_CM:.0f}cm => after(KF) hidden",
                            (10, 85), 0, 0.52, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"UDP: {'OK' if self.udp_ok else 'FAIL'} -> {SERVER_IP}:{SERVER_PORT}",
                            (10, 110), 0, 0.55, (200, 200, 200), 2, cv2.LINE_AA)

                y0 = 140
                cv2.putText(canvas, "Legend (4 points; after may hide):", (10, y0), 0, 0.60, (230,230,230), 2, cv2.LINE_AA); y0 += 22
                cv2.putText(canvas, "rear_before : PnP only (only when seen)", (10, y0), 0, 0.52, (0,140,255), 2, cv2.LINE_AA); y0 += 20
                cv2.putText(canvas, "rear_after  : v3.1 -> KF (may hide)", (10, y0), 0, 0.52, (0,0,255), 2, cv2.LINE_AA); y0 += 20
                cv2.putText(canvas, "left_before : PnP only (only when seen)", (10, y0), 0, 0.52, (255,220,0), 2, cv2.LINE_AA); y0 += 20
                cv2.putText(canvas, "left_after  : v3.1 -> KF (may hide)", (10, y0), 0, 0.52, (0,255,0), 2, cv2.LINE_AA)

                # draw rear_before only if seen
                if rear is not None:
                    self.draw_pose(canvas, rear["before_center"], rear["before_heading"],
                                   (0, 140, 255), "rear_before", 4, 1, 45)

                # draw rear_after if visible (or if just initialized & still visible)
                if self.kf_rear.visible and self.kf_rear.initialized:
                    rx, ry, rth = self.kf_rear.get()
                    self.draw_pose(canvas, np.array([rx, ry], np.float32), rth,
                                   (0, 0, 255), "rear_after(KF)", 6, 2, 55)

                # draw left_before
                if left is not None:
                    self.draw_pose(canvas, left["before_center"], left["before_heading"],
                                   (255, 220, 0), "left_before", 4, 1, 45)

                # draw left_after if visible
                if self.kf_left.visible and self.kf_left.initialized:
                    lx, ly, lth = self.kf_left.get()
                    self.draw_pose(canvas, np.array([lx, ly], np.float32), lth,
                                   (0, 255, 0), "left_after(KF)", 6, 2, 55)

                # monitor
                mon0 = cv2.resize(fr0, (640, 360)) if (ok0 and fr0 is not None) else np.zeros((360, 640, 3), np.uint8)
                mon1 = cv2.resize(fr1, (640, 360)) if (ok1 and fr1 is not None) else np.zeros((360, 640, 3), np.uint8)
                cv2.imshow(self.win_mon, np.hstack([mon0, mon1]))
                cv2.imshow(self.win_map, canvas)

                # key
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    break
                if k != 255:
                    self.handle_key(k)

                self.maybe_send()

        except KeyboardInterrupt:
            pass
        finally:
            self.close()


if __name__ == "__main__":
    App().run()