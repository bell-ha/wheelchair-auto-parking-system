#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# wc_tracker_v31_before_after_bigcanvas_fixed_coords.py
#
# Fix:
#   - grid/vehicle area uses grid_world_origin (like old off_x/off_y)
#   - camera positions ALSO include grid_world_origin (so everything shares same world frame)
#
# Shows (per cam):
#   BEFORE: PnP-based projection result (uses K,D + solvePnP)
#   AFTER : v3.1 applied on BEFORE
#
# Requires:
#   analyze/calib_params_ridge.csv
#
# Run:
#   python wc_tracker_v31_before_after_bigcanvas_fixed_coords.py
#
# Keys:
#   q / ESC : quit

import cv2
import numpy as np
import math
import csv
from dataclasses import dataclass
from pathlib import Path

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
# Helpers
# =========================
def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(down)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)

def map_deg_to_south0_deg(map_deg: float) -> float:
    # map deg: 0=+x, 90=+y(down) -> south0: 0=+y, 90=-x, 180=-y, 270=+x
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
    pos_world_px: np.ndarray     # world coords, px==cm
    h_cm: float
    map_angle_deg: float         # map deg: 0=+x, 90=+y(down)
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90      # FIXED

# =========================
# v3.1 runtime (CSV)
# feature: X=[1,g,g^2,abs(b),abs(b)^2,v,g*abs(b),reproj]
# predicts: e_par, e_perp, dtheta (ray frame)
# =========================
FEAT_KEYS = ["bias", "g", "g2", "b", "b2", "v", "g*b", "reproj"]

def rayvec_from_south0(ray_ang_south0_deg: float):
    map_deg = south0_deg_to_map_deg(ray_ang_south0_deg)
    r = math.radians(map_deg)
    return math.cos(r), math.sin(r)  # map coords unit vec

def from_ray_frame(e_par, e_perp, ray_ang_south0_deg):
    ux, uy = rayvec_from_south0(ray_ang_south0_deg)
    vx, vy = -uy, ux
    dx = e_par * ux + e_perp * vx
    dy = e_par * uy + e_perp * vy
    return dx, dy

def X_v3(ground_m, bearing_deg, view_sym_deg, reproj):
    # NOTE: v3.1 feature 정의 그대로
    g = float(ground_m)              # meters
    b = abs(float(bearing_deg))      # degrees
    v = float(view_sym_deg)          # degrees
    r = float(reproj)                # pixels
    return np.array([1.0, g, g*g, b, b*b, v, g*b, r], dtype=float)

def load_calib_params_v31(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing v3.1 params CSV: {csv_path}")

    params = {}  # cam -> (wpar, wper, wtheta_or_None)
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
        return float(pred_x), float(pred_y), float(pred_ang_south0_deg), 0.0, 0.0, 0.0

    wpar, wper, wtheta = params[cam_key]
    X = X_v3(ground_m, bearing_deg, view_sym_deg, reproj)

    epar = float(X @ wpar)
    eper = float(X @ wper)
    dx, dy = from_ray_frame(epar, eper, ray_ang_south0_deg)

    x2 = float(pred_x) + dx
    y2 = float(pred_y) + dy

    ang2 = float(pred_ang_south0_deg)
    dth = 0.0
    if wtheta is not None:
        dth = float(X @ wtheta)
        ang2 = wrap360(ang2 + dth)

    return x2, y2, ang2, epar, eper, dth

# =========================
# App
# =========================
class App:
    def __init__(self):
        # 1px == 1cm
        self.scale = 1.0

        # Big canvas for display
        self.canvas_w, self.canvas_h = 2000, 2000

        # draw offset: just to keep everything visible
        # (does NOT affect math; only drawing)
        self.draw_off_x, self.draw_off_y = 700, 500

        # -------------------------------------------------
        # ✅ This is the key alignment variable:
        #   It plays the role of your old off_x/off_y.
        #   Grid, car-zone, camera positions all live in this SAME "world".
        # -------------------------------------------------
        self.grid_world_origin = np.array([200.0, 150.0], dtype=np.float32)

        # Active area size (same as before)
        self.grid_w, self.grid_h = 600, 720

        # Car zone in world coords: ((200+offx,180+offy),(400+offx,540+offy))
        gx, gy = float(self.grid_world_origin[0]), float(self.grid_world_origin[1])
        self.car_zone_world = ((200 + gx, 180 + gy), (400 + gx, 540 + gy))

        # marker heights (cm)
        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0

        # marker -> center offset (cm)
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # -------------------------------------------------
        # ✅ Camera positions: (tocsv base values) + grid_world_origin
        #   This is exactly the coordinate fix you need.
        # -------------------------------------------------
        rear_base = np.array([301.4, 540.0], dtype=np.float32)
        left_base = np.array([200.0, 270.0], dtype=np.float32)

        self.cams = {
            "rear": CamCfg(
                key="rear", index=0,
                pos_world_px=rear_base + self.grid_world_origin,
                h_cm=105.5,
                map_angle_deg=90.0,
                sens=1.6,
                install_angle=0.0,
                install_offset=0.0,
                yaw_trim_deg=3.0,
                dist_gain=0.90,   # FIXED
            ),
            "left": CamCfg(
                key="left", index=1,
                pos_world_px=left_base + self.grid_world_origin,
                h_cm=110.0,
                map_angle_deg=157.0,
                sens=1.6,
                install_angle=113.0,
                install_offset=50.84,
                yaw_trim_deg=8.0,
                dist_gain=0.90,   # FIXED
            ),
        }

        # Load v3.1 params
        self.params_v31 = load_calib_params_v31("analyze/calib_params_ridge.csv")
        print("[LOAD] v3.1 params:", list(self.params_v31.keys()))
        print("[INFO] grid_world_origin =", self.grid_world_origin.tolist())
        print("[INFO] rear cam pos_world =", self.cams["rear"].pos_world_px.tolist())
        print("[INFO] left cam pos_world =", self.cams["left"].pos_world_px.tolist())

        # Open cameras
        self.cap_rear = cv2.VideoCapture(self.cams["rear"].index)
        self.cap_left = cv2.VideoCapture(self.cams["left"].index)
        if not self.cap_rear.isOpened() or not self.cap_left.isOpened():
            raise RuntimeError("카메라 오픈 실패: 인덱스(0/1) 확인")

        # Windows
        self.win_map = "MAP (coords fixed) BEFORE vs AFTER v3.1"
        self.win_mon = "monitor(rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)

        print("[Keys] q/ESC : quit")

    # world -> canvas (draw only)
    def w2c(self, p_world: np.ndarray) -> np.ndarray:
        return p_world + np.array([self.draw_off_x, self.draw_off_y], dtype=np.float32)

    # marker->center in world coords
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
        best = None
        best_score = -1.0

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

            # distance + ground
            dist_m = float(np.linalg.norm(tvec))
            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * cam.dist_gain  # FIXED 0.90

            # bearing
            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))

            # ray direction in map (map deg)
            ray_deg_map = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad_map = math.radians(ray_deg_map)

            # marker pos in world
            marker_world = cam.pos_world_px + np.array([
                ground_cm * math.cos(ray_rad_map),
                ground_cm * math.sin(ray_rad_map)
            ], dtype=np.float32)

            # yaw -> compass -> heading_map
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset
            yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)
            heading_before_map = compass_deg_to_map_rad(yaw_compass)

            # BEFORE center (world)
            center_before_world = self.marker_to_center_world(marker_world, heading_before_map, mid)

            # reprojection error
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            # angles for v3.1
            heading_before_map_deg = wrap360(math.degrees(heading_before_map))
            pred_ang_before_south0 = map_deg_to_south0_deg(heading_before_map_deg)
            ray_ang_south0 = map_deg_to_south0_deg(wrap360(ray_deg_map))
            view_rel_deg = wrap180(pred_ang_before_south0 - ray_ang_south0)
            view_sym_deg = view_sym_from_rel(view_rel_deg)

            # AFTER v3.1
            x_after, y_after, pred_ang_after_south0, epar, eper, dth = apply_v31(
                cam.key, self.params_v31,
                pred_x=float(center_before_world[0]),
                pred_y=float(center_before_world[1]),
                pred_ang_south0_deg=float(pred_ang_before_south0),
                ground_m=float(ground_m),
                bearing_deg=float(bearing_deg),
                ray_ang_south0_deg=float(ray_ang_south0),
                view_sym_deg=float(view_sym_deg),
                reproj=float(reproj_err),
            )
            center_after_world = np.array([x_after, y_after], dtype=np.float32)

            after_map_deg = south0_deg_to_map_deg(pred_ang_after_south0)
            heading_after_map = math.radians(after_map_deg)

            # choose best marker
            cx = float(np.mean(corners[i].reshape(4, 2)[:, 0]))
            rel_x = (cx - W / 2) / (W / 2)
            score = (1.0 - abs(rel_x)) * (1.0 / (1.0 + ground_m))

            if score > best_score:
                best_score = score
                best = {
                    "cam_key": cam.key,
                    "marker_id": mid,
                    "before": {"center_world": center_before_world, "heading_map": float(heading_before_map)},
                    "after":  {"center_world": center_after_world,  "heading_map": float(heading_after_map)},
                    "dbg": {
                        "dx_px": float(center_after_world[0] - center_before_world[0]),
                        "dy_px": float(center_after_world[1] - center_before_world[1]),
                        "epar": float(epar), "eper": float(eper), "dth": float(dth),
                        "g": float(ground_m), "b": float(bearing_deg), "v": float(view_sym_deg), "r": float(reproj_err),
                    }
                }

        return best

    # ----- drawing -----
    def draw_grid(self, img, x0, y0, w, h, step, col_minor, col_major, major_step):
        for x in range(0, w + 1, step):
            col = col_major if (x % major_step) == 0 else col_minor
            cv2.line(img, (x0 + x, y0), (x0 + x, y0 + h), col, 1)
        for y in range(0, h + 1, step):
            col = col_major if (y % major_step) == 0 else col_minor
            cv2.line(img, (x0, y0 + y), (x0 + w, y0 + y), col, 1)

    def draw_static_map(self, img):
        step = int(20 * self.scale)
        major = int(100 * self.scale)

        # full canvas grid
        self.draw_grid(img, 0, 0, self.canvas_w - 1, self.canvas_h - 1, step, (25,25,25), (45,45,45), major)

        # active area (grid_world_origin -> canvas)
        g0 = self.w2c(self.grid_world_origin).astype(int)
        gx, gy = int(g0[0]), int(g0[1])
        self.draw_grid(img, gx, gy, self.grid_w, self.grid_h, step, (45,45,45), (80,80,80), major)
        cv2.rectangle(img, (gx, gy), (gx + self.grid_w, gy + self.grid_h), (200,200,200), 2)

        # car zone
        (x0w, y0w), (x1w, y1w) = self.car_zone_world
        p0 = self.w2c(np.array([x0w, y0w], np.float32)).astype(int)
        p1 = self.w2c(np.array([x1w, y1w], np.float32)).astype(int)
        cv2.rectangle(img, tuple(p0), tuple(p1), (35,35,45), -1)

        # camera positions
        for cam in self.cams.values():
            cp = self.w2c(cam.pos_world_px).astype(int)
            cv2.circle(img, tuple(cp), 6, (220,220,220), -1)
            cv2.putText(img, cam.key, (int(cp[0])+8, int(cp[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

    def draw_point_arrow(self, img, center_world, heading_map_rad, color, label, radius=5, thickness=2, arrow_len_cm=55.0):
        c = self.w2c(center_world).astype(int)
        cv2.circle(img, tuple(c), radius, color, -1)

        end_world = center_world + np.array([arrow_len_cm * math.cos(heading_map_rad),
                                             arrow_len_cm * math.sin(heading_map_rad)], dtype=np.float32)
        e = self.w2c(end_world).astype(int)
        cv2.arrowedLine(img, tuple(c), tuple(e), color, thickness, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(img, label, (int(c[0])+10, int(c[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def draw_legend(self, img):
        y = 22
        cv2.putText(img, "Legend:", (10, y), 0, 0.62, (230,230,230), 2, cv2.LINE_AA); y += 26
        cv2.putText(img, "rear BEFORE (PnP only)", (10, y), 0, 0.52, (0,140,255), 2, cv2.LINE_AA); y += 20
        cv2.putText(img, "rear AFTER  (v3.1)",     (10, y), 0, 0.52, (0,0,255),   2, cv2.LINE_AA); y += 20
        cv2.putText(img, "left BEFORE (PnP only)", (10, y), 0, 0.52, (255,220,0), 2, cv2.LINE_AA); y += 20
        cv2.putText(img, "left AFTER  (v3.1)",     (10, y), 0, 0.52, (0,255,0),   2, cv2.LINE_AA); y += 20

    def draw_debug(self, img, rear_det, left_det):
        y = self.canvas_h - 50
        cv2.putText(img, "dist_gain fixed: rear=0.90 left=0.90", (10, y), 0, 0.55, (220,220,220), 2, cv2.LINE_AA)
        y += 18

        def line(prefix, det):
            nonlocal y
            if not det:
                cv2.putText(img, f"{prefix}: (no marker)", (10, y), 0, 0.45, (120,120,120), 2, cv2.LINE_AA); y += 16
                return
            d = det["dbg"]
            cv2.putText(
                img,
                f"{prefix}: dx={d['dx_px']:.1f}px dy={d['dy_px']:.1f}px | epar={d['epar']:.3f} eper={d['eper']:.3f} dth={d['dth']:.3f} | g={d['g']:.2f} b={d['b']:.1f} v={d['v']:.1f} r={d['r']:.2f}",
                (10, y), 0, 0.42, (200,200,200), 1, cv2.LINE_AA
            )
            y += 16

        line("rear", rear_det)
        line("left", left_det)

    def run(self):
        while True:
            ok0, fr0 = self.cap_rear.read()
            ok1, fr1 = self.cap_left.read()
            if not ok0 or fr0 is None or not ok1 or fr1 is None:
                continue

            rear_det = self.estimate_best_before_after(fr0, self.cams["rear"])
            left_det = self.estimate_best_before_after(fr1, self.cams["left"])

            canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(canvas)
            self.draw_legend(canvas)

            if rear_det is not None:
                cb = rear_det["before"]["center_world"]
                hb = rear_det["before"]["heading_map"]
                ca = rear_det["after"]["center_world"]
                ha = rear_det["after"]["heading_map"]
                self.draw_point_arrow(canvas, cb, hb, color=(0,140,255), label="rear_before", radius=4, thickness=1, arrow_len_cm=45)
                self.draw_point_arrow(canvas, ca, ha, color=(0,0,255),   label="rear_after(v3.1)", radius=6, thickness=2, arrow_len_cm=55)

            if left_det is not None:
                cb = left_det["before"]["center_world"]
                hb = left_det["before"]["heading_map"]
                ca = left_det["after"]["center_world"]
                ha = left_det["after"]["heading_map"]
                self.draw_point_arrow(canvas, cb, hb, color=(255,220,0), label="left_before", radius=4, thickness=1, arrow_len_cm=45)
                self.draw_point_arrow(canvas, ca, ha, color=(0,255,0),   label="left_after(v3.1)", radius=6, thickness=2, arrow_len_cm=55)

            self.draw_debug(canvas, rear_det, left_det)

            mon0 = cv2.resize(fr0, (640, 360))
            mon1 = cv2.resize(fr1, (640, 360))
            cv2.imshow(self.win_mon, np.hstack([mon0, mon1]))
            cv2.imshow(self.win_map, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    App().run()