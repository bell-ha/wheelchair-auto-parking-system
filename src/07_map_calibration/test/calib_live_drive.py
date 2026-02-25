#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teleop_v31_map_opencv_keys.py
#
# macOS 안정 버전:
# - OpenCV GUI(imshow/waitKey)는 메인 스레드에서만 사용 (스레드X, cursesX)
# - OpenCV 창 포커스에서 WASD/Space/X/Q로 조종
# - UDP로 v,w를 SEND_HZ로 전송
# - v3.1 before/after map 표시
#
# Requires:
#   analyze/calib_params_ridge.csv
#
# Run:
#   python teleop_v31_map_opencv_keys.py

import json, socket, time, math, csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =======================
# USER SETTINGS (Teleop)
# =======================
SERVER_IP = "172.25.244.144"
SERVER_PORT = 25001

SEND_HZ = 30
FWD_MM_S = 200
REV_MM_S = -200
YAW_MRAD_S = 300

CALIB_CSV = "analyze/calib_params_ridge.csv"
DIST_GAIN_FIXED = 0.90


# =========================
# UDP helper
# =========================
def udp_send(sock, addr, payload: dict):
    sock.sendto(json.dumps(payload).encode("utf-8"), addr)


# =========================
# Camera intrinsics (shared fisheye)
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

def compass_deg_to_map_rad(compass_deg: float) -> float:
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
# v3.1 runtime (CSV)
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
# UI / Estimator
# =========================
class App:
    def __init__(self):
        # big canvas
        self.canvas_w, self.canvas_h = 2000, 2000
        self.draw_off_x, self.draw_off_y = 700, 500  # draw-only

        # coord alignment (old off_x/off_y)
        self.grid_world_origin = np.array([200.0, 150.0], dtype=np.float32)
        self.grid_w, self.grid_h = 600, 720

        gx, gy = float(self.grid_world_origin[0]), float(self.grid_world_origin[1])
        self.car_zone_world = ((200 + gx, 180 + gy), (400 + gx, 540 + gy))

        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        rear_base = np.array([301.4, 540.0], dtype=np.float32)
        left_base = np.array([200.0, 270.0], dtype=np.float32)

        self.cams = {
            "rear": CamCfg("rear", 0, rear_base + self.grid_world_origin, 105.5, 90.0, 1.6, 0.0, 0.0, yaw_trim_deg=3.0, dist_gain=DIST_GAIN_FIXED),
            "left": CamCfg("left", 1, left_base + self.grid_world_origin, 110.0, 157.0, 1.6, 113.0, 50.84, yaw_trim_deg=8.0, dist_gain=DIST_GAIN_FIXED),
        }

        self.params_v31 = load_calib_params_v31(CALIB_CSV)
        print("[LOAD] v3.1 params:", list(self.params_v31.keys()))
        print("[INFO] dist_gain fixed =", DIST_GAIN_FIXED)

        self.cap0 = cv2.VideoCapture(self.cams["rear"].index)
        self.cap1 = cv2.VideoCapture(self.cams["left"].index)
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            raise RuntimeError("Camera open failed. index(0/1) 확인")

        self.win_map = "MAP (focus here for WASD) - q/ESC quit"
        self.win_mon = "MONITOR (rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (SERVER_IP, SERVER_PORT)

        self.v = 0
        self.w = 0
        self.last_key = "STOP"
        self.last_send = 0.0
        self.dt = 1.0 / float(SEND_HZ)

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

    def estimate_best_before_after(self, frame_bgr, cam: CamCfg):
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

            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset
            yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)
            heading_before_map = compass_deg_to_map_rad(yaw_compass)

            center_before = self.marker_to_center_world(marker_world, heading_before_map, mid)

            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

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
            center_after = np.array([x2, y2], dtype=np.float32)
            heading_after_map = math.radians(south0_deg_to_map_deg(ang2_south0))

            cx = float(np.mean(corners[i].reshape(4, 2)[:, 0]))
            rel_x = (cx - W / 2) / (W / 2)
            score = (1.0 - abs(rel_x)) * (1.0 / (1.0 + ground_m))

            if score > best_score:
                best_score = score
                best = {
                    "before": {"center": center_before, "heading": float(heading_before_map)},
                    "after":  {"center": center_after,  "heading": float(heading_after_map)},
                }

        return best

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

    def handle_key(self, keycode: int):
        # keycode from cv2.waitKey (lowercase)
        if keycode == ord('w'):
            self.v, self.w = FWD_MM_S, 0
            self.last_key = "W"
        elif keycode == ord('s'):
            self.v, self.w = REV_MM_S, 0
            self.last_key = "S"
        elif keycode == ord('a'):
            self.v, self.w = 0, YAW_MRAD_S
            self.last_key = "A"
        elif keycode == ord('d'):
            self.v, self.w = 0, -YAW_MRAD_S
            self.last_key = "D"
        elif keycode == ord('x') or keycode == 32:  # space
            self.v, self.w = 0, 0
            self.last_key = "STOP"
            udp_send(self.sock, self.addr, {"stop": True})
        # q/esc handled in main loop

    def maybe_send(self):
        now = time.time()
        if (now - self.last_send) >= self.dt:
            udp_send(self.sock, self.addr, {"v": self.v, "w": self.w})
            self.last_send = now

    def run(self):
        try:
            while True:
                ok0, fr0 = self.cap0.read()
                ok1, fr1 = self.cap1.read()

                if not ok0 or fr0 is None or not ok1 or fr1 is None:
                    # 카메라가 순간 끊겨도 프로그램은 살아있게
                    self.maybe_send()
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord('q')):
                        break
                    time.sleep(0.01)
                    continue

                rear = self.estimate_best_before_after(fr0, self.cams["rear"])
                left = self.estimate_best_before_after(fr1, self.cams["left"])

                canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 15
                self.draw_static(canvas)

                # legend + command
                cv2.putText(canvas, "Focus MAP window for keys: W/A/S/D, X or Space=STOP, Q/ESC=QUIT",
                            (10, 25), 0, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(canvas, "rear BEFORE", (10, 55), 0, 0.5, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "rear AFTER(v3.1)", (10, 78), 0, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "left BEFORE", (10, 101), 0, 0.5, (255, 220, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, "left AFTER(v3.1)", (10, 124), 0, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(canvas, f"CMD: v={self.v} mm/s, w={self.w} mrad/s  (last={self.last_key})",
                            (10, 155), 0, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

                if rear is not None:
                    self.draw_pose(canvas, rear["before"]["center"], rear["before"]["heading"],
                                   (0, 140, 255), "rear_before", 4, 1, 45)
                    self.draw_pose(canvas, rear["after"]["center"], rear["after"]["heading"],
                                   (0, 0, 255), "rear_after(v3.1)", 6, 2, 55)

                if left is not None:
                    self.draw_pose(canvas, left["before"]["center"], left["before"]["heading"],
                                   (255, 220, 0), "left_before", 4, 1, 45)
                    self.draw_pose(canvas, left["after"]["center"], left["after"]["heading"],
                                   (0, 255, 0), "left_after(v3.1)", 6, 2, 55)

                # monitor windows: guard against shape mismatch
                mon0 = cv2.resize(fr0, (640, 360))
                mon1 = cv2.resize(fr1, (640, 360))
                mon = np.hstack([mon0, mon1])  # now always same shape
                cv2.imshow(self.win_mon, mon)
                cv2.imshow(self.win_map, canvas)

                # key handling
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):  # ESC or q
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