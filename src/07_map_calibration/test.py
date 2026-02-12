#!/usr/bin/env python3
# wc_tracker_versioned_affine_calib_distgain90_mp4.py
#
# ✅ dist_gain=0.90 고정
# ✅ calib_vN.json / raw_vN.json 버전 관리
# ✅ c/a는 "기록만"
# ✅ s에서 raw_vN -> calib_v(N+1) (아핀+거리) 피팅 후 즉시 적용/버전업
# ✅ 카메라 대신 MP4로 테스트 가능 (../command/1_rear.mp4, ../command/1_left.mp4)
# ✅ 영상 끝나면 자동 루프

import cv2
import numpy as np
import math
import time
import json
import os
import glob
from dataclasses import dataclass
from collections import deque

# =========================
# ✅ INPUT MODE: MP4 files
# =========================
USE_VIDEO_FILES = True  # True면 mp4, False면 카메라(0/1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAR_VIDEO = os.path.normpath(os.path.join(BASE_DIR, "../command/1_rear.mp4"))
LEFT_VIDEO = os.path.normpath(os.path.join(BASE_DIR, "../command/1_left.mp4"))
LOOP_VIDEO = True  # 영상 끝나면 처음으로 되감기

# =========================
# Intrinsic (shared fisheye)
# =========================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE_M = 0.25  # 25cm

# =========================
# ArUco
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

OBJ_POINTS = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0]
], dtype=np.float32)


def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg


def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(아래)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)


@dataclass
class CamCfg:
    key: str
    index: int
    pos_px: np.ndarray
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90  # ✅ fixed


@dataclass
class CalibModel:
    version: int
    # position: p* = (A0 + d*A1) p + (b0 + d*b1)
    A0: np.ndarray  # (2,2)
    A1: np.ndarray  # (2,2)
    b0: np.ndarray  # (2,)
    b1: np.ndarray  # (2,)
    # yaw: theta* = theta + (a + k*d)
    yaw_a: float
    yaw_k: float
    created_at: float
    parent_version: int = 0
    made_from_raw_version: int = 0

    def transform_pos(self, p_raw: np.ndarray, d: float) -> np.ndarray:
        d = float(d if d is not None else 0.0)
        M = self.A0 + self.A1 * d
        t = self.b0 + self.b1 * d
        return (M @ p_raw.astype(np.float32).reshape(2,)) + t.astype(np.float32)

    def transform_yaw(self, yaw_raw: float, d: float) -> float:
        d = float(d if d is not None else 0.0)
        return wrap_pi(float(yaw_raw) + float(self.yaw_a + self.yaw_k * d))


class WheelchairTracker:
    def __init__(self):
        # =========================
        # Map params
        # =========================
        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 1.0  # 1px=1cm
        self.off_x, self.off_y = 200, 150

        self.car_zone = ((200 + self.off_x, 180 + self.off_y),
                         (400 + self.off_x, 540 + self.off_y))

        # wheelchair size (cm)
        self.wc_w_cm, self.wc_l_cm = 55.0, 66.0

        # marker height (cm)
        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0

        # marker->center offset (cm)
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # =========================
        # Tracking smoothing (anti-jitter)
        # =========================
        self.alpha = 0.30
        self.bufN = 5
        self.buf_center = deque(maxlen=self.bufN)
        self.buf_sin = deque(maxlen=self.bufN)
        self.buf_cos = deque(maxlen=self.bufN)

        self.turn_fast_deg = 35.0
        self.alpha_fast = 0.85
        self.heading_mag_min = 0.25

        self.lost_count = 0
        self.lost_reset_frames = 8

        # =========================
        # Quality (for display / saving)
        # =========================
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # =========================
        # Camera configs (dist_gain fixed 0.90)
        # =========================
        self.cams = {
            "rear": CamCfg(
                key="rear", index=0,
                pos_px=np.array([301.4 + self.off_x, 540.0 + self.off_y], dtype=np.float32),
                h_cm=105.5,
                map_angle_deg=90.0,
                sens=1.6,
                install_angle=0.0,
                install_offset=0.0,
                yaw_trim_deg=3.0,
                dist_gain=0.90
            ),
            "left": CamCfg(
                key="left", index=1,
                pos_px=np.array([200.0 + self.off_x, 270.0 + self.off_y], dtype=np.float32),
                h_cm=110.0,
                map_angle_deg=157.0,
                sens=1.6,
                install_angle=113.0,
                install_offset=50.84,
                yaw_trim_deg=8.0,
                dist_gain=0.90
            )
        }

        # =========================
        # Versioned calibration / raw data
        # =========================
        self.active_version = self._find_or_create_latest_calib()
        self.calib = self._load_calib(self.active_version)
        self.raw_samples = self._load_or_create_raw(self.active_version)

        # =========================
        # ✅ Open inputs (MP4 or cameras)
        # =========================
        if USE_VIDEO_FILES:
            if not os.path.exists(REAR_VIDEO):
                raise FileNotFoundError(f"rear video not found: {REAR_VIDEO}")
            if not os.path.exists(LEFT_VIDEO):
                raise FileNotFoundError(f"left video not found: {LEFT_VIDEO}")

            self.cap_rear = cv2.VideoCapture(REAR_VIDEO)
            self.cap_left = cv2.VideoCapture(LEFT_VIDEO)
            if not self.cap_rear.isOpened() or not self.cap_left.isOpened():
                raise RuntimeError("MP4 오픈 실패: 경로/코덱 확인")

            print(f"[INPUT] MP4 rear={REAR_VIDEO}")
            print(f"[INPUT] MP4 left={LEFT_VIDEO}")
        else:
            self.cap_rear = cv2.VideoCapture(self.cams["rear"].index)
            self.cap_left = cv2.VideoCapture(self.cams["left"].index)
            if not self.cap_rear.isOpened() or not self.cap_left.isOpened():
                raise RuntimeError("카메라 오픈 실패: 인덱스(0/1) 확인")
            print("[INPUT] Cameras 0/1")

        # fused state (no calib applied here)
        self.raw_center = None
        self.raw_heading = 0.0
        self.is_initialized = False

        # per-frame caches (for click saving)
        self.latest_raw_center = None
        self.latest_raw_heading = 0.0
        self.latest_corr_center = None
        self.latest_corr_heading = 0.0
        self.latest_fused_d = None
        self.latest_cam_obs = {
            "rear": {"seen": False, "ground_m": None, "quality": None},
            "left": {"seen": False, "ground_m": None, "quality": None},
        }

        # UI
        self.win_map = "minimap"
        self.win_mon = "monitor(rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_map, self._on_mouse)

        cv2.createTrackbar("Smooth(0-100)", self.win_map, int(self.alpha * 100), 100, lambda v: None)

        # click mode
        self.click_mode = None  # None | record_center | record_yaw_center | record_yaw_dir
        self.tmp_yaw_center = None

        print("[Versioned Calib]")
        print(f"  ACTIVE calib_v{self.active_version}.json  (raw -> raw_v{self.active_version}.json)")
        print("[Keys]")
        print("  q/ESC : quit")
        print("  r     : reset tracking filter (not calib)")
        print("  c     : record CENTER sample (click true center on map)")
        print("  a     : record YAW sample (click center then direction)")
        print("  s     : fit from raw_vN -> make calib_v(N+1) and switch")

    # ============================================================
    # Versioned file helpers
    # ============================================================
    @staticmethod
    def _calib_path(ver: int) -> str:
        return f"calib_v{ver}.json"

    @staticmethod
    def _raw_path(ver: int) -> str:
        return f"raw_v{ver}.json"

    def _find_or_create_latest_calib(self) -> int:
        files = glob.glob("calib_v*.json")
        vers = []
        for f in files:
            base = os.path.basename(f)
            try:
                n = int(base.replace("calib_v", "").replace(".json", ""))
                vers.append(n)
            except Exception:
                pass
        if vers:
            return max(vers)

        # create v1 default
        v1 = 1
        default = CalibModel(
            version=v1,
            A0=np.eye(2, dtype=np.float32),
            A1=np.zeros((2, 2), dtype=np.float32),
            b0=np.zeros((2,), dtype=np.float32),
            b1=np.zeros((2,), dtype=np.float32),
            yaw_a=0.0,
            yaw_k=0.0,
            created_at=time.time(),
            parent_version=0,
            made_from_raw_version=0,
        )
        self._save_calib(default)
        print("[INIT] created calib_v1.json (identity / zero)")
        return v1

    def _load_calib(self, ver: int) -> CalibModel:
        path = self._calib_path(ver)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        A0 = np.array(data["model"]["A0"], dtype=np.float32)
        A1 = np.array(data["model"]["A1"], dtype=np.float32)
        b0 = np.array(data["model"]["b0"], dtype=np.float32)
        b1 = np.array(data["model"]["b1"], dtype=np.float32)
        yaw_a = float(data["model"].get("yaw_a", 0.0))
        yaw_k = float(data["model"].get("yaw_k", 0.0))

        return CalibModel(
            version=int(data.get("version", ver)),
            A0=A0, A1=A1, b0=b0, b1=b1,
            yaw_a=yaw_a, yaw_k=yaw_k,
            created_at=float(data.get("created_at", time.time())),
            parent_version=int(data.get("parent_version", 0)),
            made_from_raw_version=int(data.get("made_from_raw_version", 0)),
        )

    def _save_calib(self, calib: CalibModel, fit_stats: dict | None = None, params_used: dict | None = None):
        path = self._calib_path(calib.version)
        data = {
            "schema_version": 1,
            "version": calib.version,
            "parent_version": calib.parent_version,
            "made_from_raw_version": calib.made_from_raw_version,
            "created_at": calib.created_at,
            "dist_gain_fixed": 0.90,
            "model": {
                "type": "affine_distance",
                "A0": calib.A0.tolist(),
                "A1": calib.A1.tolist(),
                "b0": calib.b0.tolist(),
                "b1": calib.b1.tolist(),
                "yaw_a": float(calib.yaw_a),
                "yaw_k": float(calib.yaw_k),
            },
            "fit_stats": fit_stats or {},
            "params_used": params_used or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {path}")

    def _load_or_create_raw(self, ver: int) -> list:
        path = self._raw_path(ver)
        if not os.path.exists(path):
            data = {"schema_version": 1, "calib_version": ver, "created_at": time.time(), "samples": []}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[INIT] created {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", []) or []

    def _flush_raw(self):
        path = self._raw_path(self.active_version)
        data = {
            "schema_version": 1,
            "calib_version": self.active_version,
            "updated_at": time.time(),
            "samples": self.raw_samples
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Trackbars
    # ============================================================
    def update_from_trackbars(self):
        self.alpha = max(0.01, cv2.getTrackbarPos("Smooth(0-100)", self.win_map) / 100.0)

    # ============================================================
    # Geometry helpers
    # ============================================================
    def marker_to_center(self, marker_pos_px: np.ndarray, heading_map_rad: float, marker_id: int) -> np.ndarray:
        offset_cm = float(self.center_offset_cm_by_id.get(marker_id, 23.0))
        offset_px = offset_cm * self.map_scale
        dx = offset_px * math.cos(heading_map_rad)
        dy = offset_px * math.sin(heading_map_rad)
        if marker_id == 0:
            return marker_pos_px - np.array([dx, dy], dtype=np.float32)
        else:
            return marker_pos_px + np.array([dx, dy], dtype=np.float32)

    @staticmethod
    def smooth01(x, x0, x1):
        if x <= x0:
            return 1.0
        if x >= x1:
            return 0.0
        t = (x - x0) / (x1 - x0)
        return float(1.0 - t)

    # ============================================================
    # Detection / estimation
    # ============================================================
    def estimate_from_frame(self, frame, cam: CamCfg):
        dets = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return dets

        pnp_flag = cv2.SOLVEPNP_ITERATIVE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE

        for i, mid_arr in enumerate(ids):
            mid = int(mid_arr[0])
            if mid not in (0, 1):
                continue

            c2 = corners[i].reshape(4, 2).astype(np.float32)
            und = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)

            ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=pnp_flag)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            dist_m = float(np.linalg.norm(tvec))

            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))

            ground_cm = ground_m * 100.0 * cam.dist_gain  # ✅ fixed gain

            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
            ray_deg = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cam.pos_px + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset

            if mid == 1:
                yaw_compass = wrap360(yaw_compass + 180.0)
            else:
                yaw_compass = wrap360(yaw_compass)

            heading_map = compass_deg_to_map_rad(yaw_compass)
            center_pos = self.marker_to_center(marker_pos, heading_map, mid)

            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            z = float(tvec[2])
            z_score = 1.0 if z > 0.05 else 0.0

            s_area = self.smooth01(area, self.area_good_px2, self.area_bad_px2)
            s_err = self.smooth01(reproj_err, self.reproj_good_px, self.reproj_bad_px)
            quality = max(self.min_quality_w, (0.45 * s_err + 0.45 * s_area + 0.10 * z_score))

            cx = float(np.mean(c2[:, 0]))
            rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            w_base = float(max(0.05, w_center * w_dist))
            w = float(w_base * quality)

            dets.append({
                "marker_id": mid,
                "marker_pos": marker_pos,
                "center_pos": center_pos,
                "heading": heading_map,
                "weight": w,
                "cam_key": cam.key,
                "dbg_quality": {
                    "quality": float(quality),
                    "area": float(area),
                    "reproj": float(reproj_err),
                    "ground_m": float(ground_m),
                }
            })

        return dets

    def fuse(self, dets):
        if not dets:
            return None
        total_w = sum(d["weight"] for d in dets)
        if total_w <= 1e-9:
            return None
        center = sum(d["center_pos"] * d["weight"] for d in dets) / total_w
        s = sum(math.sin(d["heading"]) * d["weight"] for d in dets) / total_w
        c = sum(math.cos(d["heading"]) * d["weight"] for d in dets) / total_w
        heading = math.atan2(s, c)

        d_fused = None
        sw = 0.0
        sd = 0.0
        for d in dets:
            gm = float(d["dbg_quality"]["ground_m"])
            w = float(d["weight"])
            sd += gm * w
            sw += w
        if sw > 1e-9:
            d_fused = sd / sw

        return center, heading, d_fused

    # ============================================================
    # Robust buffer helpers
    # ============================================================
    def _robust_measurement_from_buffer(self, latest_center, latest_heading):
        xs = [float(p[0]) for p in self.buf_center]
        ys = [float(p[1]) for p in self.buf_center]
        med_center = np.array([np.median(xs), np.median(ys)], dtype=np.float32)

        s = float(np.mean(self.buf_sin)) if len(self.buf_sin) else math.sin(latest_heading)
        c = float(np.mean(self.buf_cos)) if len(self.buf_cos) else math.cos(latest_heading)
        mag = math.hypot(s, c)
        if mag < self.heading_mag_min:
            med_heading = latest_heading
        else:
            med_heading = math.atan2(s, c)
        return med_center, med_heading, mag

    # ============================================================
    # Calibration application
    # ============================================================
    def corrected_pose(self):
        if self.raw_center is None:
            return None
        d = self.latest_fused_d if self.latest_fused_d is not None else 0.0
        c = self.calib.transform_pos(self.raw_center, d)
        h = self.calib.transform_yaw(self.raw_heading, d)
        return c, h

    # ============================================================
    # UI drawing
    # ============================================================
    def _draw_grid(self, img, x0, y0, w, h, step, col_minor, col_major, major_step):
        for x in range(0, w + 1, step):
            col = col_major if (x % major_step) == 0 else col_minor
            cv2.line(img, (x0 + x, y0), (x0 + x, y0 + h), col, 1)
        for y in range(0, h + 1, step):
            col = col_major if (y % major_step) == 0 else col_minor
            cv2.line(img, (x0, y0 + y), (x0 + w, y0 + y), col, 1)

    def draw_static_map(self, img):
        grid_step_cm = 20
        step = max(1, int(grid_step_cm * self.map_scale))
        major_step = max(1, int(100 * self.map_scale))

        self._draw_grid(img, 0, 0, self.map_w - 1, self.map_h - 1,
                        step, (25, 25, 25), (45, 45, 45), major_step)

        self._draw_grid(img, self.off_x, self.off_y, self.grid_w, self.grid_h,
                        step, (45, 45, 45), (80, 80, 80), major_step)

        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h), (200, 200, 200), 2)

        cv2.rectangle(img, self.car_zone[0], self.car_zone[1], (35, 35, 45), -1)

        for cam in self.cams.values():
            cp = tuple(cam.pos_px.astype(int))
            cv2.circle(img, cp, 6, (220, 220, 220), -1)
            cv2.putText(img, cam.key, (cp[0] + 8, cp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    def draw_wheelchair(self, img, center, heading, color=(0, 255, 0), label="CORR"):
        w_px = (self.wc_w_cm * self.map_scale) / 2.0
        l_px = (self.wc_l_cm * self.map_scale) / 2.0

        base = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]], dtype=np.float32)
        rot = np.array([[math.cos(heading), -math.sin(heading)],
                        [math.sin(heading),  math.cos(heading)]], dtype=np.float32)
        pts = (base @ rot.T) + center

        cv2.polylines(img, [pts.astype(np.int32)], True, color, 2, cv2.LINE_AA)

        arrow_len_cm = 55.0
        arrow_len_px = arrow_len_cm * self.map_scale
        cv2.arrowedLine(img, tuple(center.astype(int)),
                        (int(center[0] + arrow_len_px * math.cos(heading)),
                         int(center[1] + arrow_len_px * math.sin(heading))),
                        color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (int(center[0]) + 10, int(center[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def draw_hud(self, img, dets):
        y = 20
        cv2.putText(img, f"ACTIVE: calib_v{self.active_version}.json   RECORD: raw_v{self.active_version}.json (samples={len(self.raw_samples)})",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 22
        cv2.putText(img, f"alpha={self.alpha:.2f}  bufN={self.bufN}  lost={self.lost_count}/{self.lost_reset_frames}  dist_gain(fixed)=0.90",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 22

        if self.latest_fused_d is not None:
            cv2.putText(img, f"fused_d={self.latest_fused_d:.2f}m  rear={self.latest_cam_obs['rear']['ground_m']}  left={self.latest_cam_obs['left']['ground_m']}",
                        (10, y), 0, 0.5, (200, 200, 200), 2, cv2.LINE_AA); y += 20
        else:
            cv2.putText(img, f"fused_d=None  rear_seen={self.latest_cam_obs['rear']['seen']} left_seen={self.latest_cam_obs['left']['seen']}",
                        (10, y), 0, 0.5, (200, 200, 200), 2, cv2.LINE_AA); y += 20

        if self.click_mode == "record_center":
            cv2.putText(img, "MODE: center -> click TRUE center point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "record_yaw_center":
            cv2.putText(img, "MODE: yaw -> click CENTER point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "record_yaw_dir":
            cv2.putText(img, "MODE: yaw -> click DIRECTION point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24

        if dets:
            best = max(dets, key=lambda d: d["weight"])
            q = best.get("dbg_quality", {})
            cv2.putText(img, f"best[{best['cam_key']}/ID{best['marker_id']}] q={q.get('quality',0):.2f} area={q.get('area',0):.0f} reproj={q.get('reproj',0):.2f}",
                        (10, y), 0, 0.5, (200, 200, 200), 2, cv2.LINE_AA)

    # ============================================================
    # Mouse callback (record only)
    # ============================================================
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.is_initialized or self.latest_raw_center is None or self.latest_corr_center is None:
            print("[CLICK] ignored: tracker not initialized")
            self.click_mode = None
            self.tmp_yaw_center = None
            return

        pt = np.array([float(x), float(y)], dtype=np.float32)
        now = time.time()

        obs = {
            "rear": dict(self.latest_cam_obs["rear"]),
            "left": dict(self.latest_cam_obs["left"]),
        }
        fused_d = self.latest_fused_d

        if self.click_mode == "record_center":
            sample = {
                "type": "center",
                "t": now,
                "clicked_xy": [float(pt[0]), float(pt[1])],
                "raw_center": [float(self.latest_raw_center[0]), float(self.latest_raw_center[1])],
                "corr_center": [float(self.latest_corr_center[0]), float(self.latest_corr_center[1])],
                "raw_heading": float(self.latest_raw_heading),
                "corr_heading": float(self.latest_corr_heading),
                "fused_d": None if fused_d is None else float(fused_d),
                "cam_obs": obs
            }
            self.raw_samples.append(sample)
            self._flush_raw()
            self.click_mode = None
            print(f"[REC center] +1 -> raw_v{self.active_version}.json (total={len(self.raw_samples)})")
            return

        if self.click_mode == "record_yaw_center":
            self.tmp_yaw_center = pt
            self.click_mode = "record_yaw_dir"
            print("[REC yaw] center picked. Now click direction point.")
            return

        if self.click_mode == "record_yaw_dir":
            if self.tmp_yaw_center is None:
                self.click_mode = None
                return

            v = (pt - self.tmp_yaw_center).astype(np.float32)
            if float(np.linalg.norm(v)) < 1e-3:
                print("[REC yaw] direction too small. retry.")
                return

            clicked_heading = float(math.atan2(float(v[1]), float(v[0])))

            sample = {
                "type": "yaw",
                "t": now,
                "clicked_heading": float(clicked_heading),
                "raw_heading": float(self.latest_raw_heading),
                "corr_heading": float(self.latest_corr_heading),
                "raw_center": [float(self.latest_raw_center[0]), float(self.latest_raw_center[1])],
                "corr_center": [float(self.latest_corr_center[0]), float(self.latest_corr_center[1])],
                "fused_d": None if fused_d is None else float(fused_d),
                "cam_obs": obs
            }
            self.raw_samples.append(sample)
            self._flush_raw()

            self.click_mode = None
            self.tmp_yaw_center = None
            print(f"[REC yaw] +1 -> raw_v{self.active_version}.json (total={len(self.raw_samples)})")
            return

    # ============================================================
    # Fitting (s key)  -- (여기 아래는 이전 코드 그대로)
    # ============================================================
    @staticmethod
    def _safe_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    def _sample_quality_and_d(self, s: dict):
        d = self._safe_float(s.get("fused_d"), None)

        rear = s.get("cam_obs", {}).get("rear", {})
        left = s.get("cam_obs", {}).get("left", {})

        dr = self._safe_float(rear.get("ground_m"), None)
        dl = self._safe_float(left.get("ground_m"), None)

        qr = self._safe_float(rear.get("quality"), None)
        ql = self._safe_float(left.get("quality"), None)

        seen_r = bool(rear.get("seen", False))
        seen_l = bool(left.get("seen", False))

        if d is None:
            if seen_r and seen_l and (dr is not None) and (dl is not None):
                wr = (qr if qr is not None else 0.5)
                wl = (ql if ql is not None else 0.5)
                if (wr + wl) > 1e-9:
                    d = (wr * dr + wl * dl) / (wr + wl)
                else:
                    d = 0.5 * (dr + dl)
            elif seen_r and (dr is not None):
                d = dr
            elif seen_l and (dl is not None):
                d = dl

        q = 0.5
        qs = []
        if qr is not None: qs.append(qr)
        if ql is not None: qs.append(ql)
        if qs:
            q = float(np.clip(float(np.mean(qs)), 0.05, 1.0))
        cams_seen = int(seen_r) + int(seen_l)
        cam_factor = 1.0 if cams_seen == 2 else (0.7 if cams_seen == 1 else 0.2)

        return d, q * cam_factor, cams_seen

    def _wls_ridge_prior(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, theta0: np.ndarray):
        W = w.reshape(-1, 1)
        XtWX = X.T @ (W * X)
        XtWy = X.T @ (w * y)
        A = XtWX + lam * np.eye(X.shape[1], dtype=np.float64)
        b = XtWy + lam * theta0
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _fit_affine_distance(self, center_samples: list, prior: CalibModel):
        X_list, Yx, Yy, W = [], [], [], []

        for s in center_samples:
            clicked = s.get("clicked_xy", None)
            rawc = s.get("raw_center", None)
            if clicked is None or rawc is None:
                continue

            d, wq, _ = self._sample_quality_and_d(s)
            if d is None:
                continue

            x, y = float(rawc[0]), float(rawc[1])
            x_star, y_star = float(clicked[0]), float(clicked[1])

            X_list.append([x, y, 1.0, d * x, d * y, d])
            Yx.append(x_star)
            Yy.append(y_star)
            W.append(max(0.02, float(wq)))

        if len(X_list) < 6:
            return None, {"msg": f"not enough center samples for affine (need>=6, got={len(X_list)})"}

        X = np.array(X_list, dtype=np.float64)
        yx = np.array(Yx, dtype=np.float64)
        yy = np.array(Yy, dtype=np.float64)
        w = np.array(W, dtype=np.float64)

        theta0_x = np.array([
            float(prior.A0[0, 0]), float(prior.A0[0, 1]), float(prior.b0[0]),
            float(prior.A1[0, 0]), float(prior.A1[0, 1]), float(prior.b1[0]),
        ], dtype=np.float64)
        theta0_y = np.array([
            float(prior.A0[1, 0]), float(prior.A0[1, 1]), float(prior.b0[1]),
            float(prior.A1[1, 0]), float(prior.A1[1, 1]), float(prior.b1[1]),
        ], dtype=np.float64)

        Wcol = w.reshape(-1, 1)
        XtWX = X.T @ (Wcol * X)
        base = float(np.trace(XtWX) / max(1, X.shape[1]))
        lam = max(1e-6, 1e-3 * base)

        theta_x = self._wls_ridge_prior(X, yx, w, lam, theta0_x)
        theta_y = self._wls_ridge_prior(X, yy, w, lam, theta0_y)

        for _ in range(2):
            pred_x = X @ theta_x
            pred_y = X @ theta_y
            rx = yx - pred_x
            ry = yy - pred_y
            r = np.sqrt(rx * rx + ry * ry)

            d_vals = X[:, 5]
            sigma = 15.0 + 10.0 * np.clip(d_vals, 0.0, 10.0)
            robust = 1.0 / (1.0 + (r / sigma) ** 2)
            w2 = w * robust

            theta_x = self._wls_ridge_prior(X, yx, w2, lam, theta_x)
            theta_y = self._wls_ridge_prior(X, yy, w2, lam, theta_y)

        A0 = np.array([[theta_x[0], theta_x[1]],
                       [theta_y[0], theta_y[1]]], dtype=np.float32)
        A1 = np.array([[theta_x[3], theta_x[4]],
                       [theta_y[3], theta_y[4]]], dtype=np.float32)
        b0 = np.array([theta_x[2], theta_y[2]], dtype=np.float32)
        b1 = np.array([theta_x[5], theta_y[5]], dtype=np.float32)

        pred_x = X @ theta_x
        pred_y = X @ theta_y
        rmse = float(np.sqrt(np.mean((yx - pred_x) ** 2 + (yy - pred_y) ** 2)))

        info = {"n": int(len(X_list)), "rmse_pos_px": rmse, "lam": lam}
        return (A0, A1, b0, b1), info

    def _fit_yaw_distance(self, yaw_samples: list, prior: CalibModel):
        D_list, Off_list, W_list = [], [], []

        for s in yaw_samples:
            d, wq, _ = self._sample_quality_and_d(s)
            if d is None:
                continue
            th_raw = self._safe_float(s.get("raw_heading"), None)
            th_star = self._safe_float(s.get("clicked_heading"), None)
            if th_raw is None or th_star is None:
                continue
            off = wrap_pi(float(th_star) - float(th_raw))
            D_list.append([1.0, float(d)])
            Off_list.append(float(off))
            W_list.append(max(0.02, float(wq)))

        if len(D_list) < 2:
            return None, {"msg": f"not enough yaw samples (need>=2, got={len(D_list)})"}

        X = np.array(D_list, dtype=np.float64)
        y = np.array(Off_list, dtype=np.float64)
        w = np.array(W_list, dtype=np.float64)

        theta0 = np.array([float(prior.yaw_a), float(prior.yaw_k)], dtype=np.float64)

        Wcol = w.reshape(-1, 1)
        XtWX = X.T @ (Wcol * X)
        base = float(np.trace(XtWX) / max(1, X.shape[1]))
        lam = max(1e-8, 1e-3 * base)

        theta = self._wls_ridge_prior(X, y, w, lam, theta0)

        for _ in range(2):
            pred = X @ theta
            res = np.array([wrap_pi(float(yi - pi)) for yi, pi in zip(y, pred)], dtype=np.float64)
            d_vals = X[:, 1]
            sigma = np.deg2rad(8.0 + 5.0 * np.clip(d_vals, 0.0, 10.0))
            robust = 1.0 / (1.0 + (np.abs(res) / sigma) ** 2)
            w2 = w * robust
            theta = self._wls_ridge_prior(X, y, w2, lam, theta)

        pred = X @ theta
        res = np.array([wrap_pi(float(yi - pi)) for yi, pi in zip(y, pred)], dtype=np.float64)
        rmse_deg = float(np.sqrt(np.mean(res * res)) * (180.0 / math.pi))

        info = {"n": int(len(D_list)), "rmse_yaw_deg": rmse_deg, "lam": lam}
        return (float(theta[0]), float(theta[1])), info

    def build_next_calib_from_raw(self):
        vN = self.active_version
        vNp1 = vN + 1

        centers = [s for s in self.raw_samples if s.get("type") == "center"]
        yaws = [s for s in self.raw_samples if s.get("type") == "yaw"]

        pos_fit, pos_info = self._fit_affine_distance(centers, self.calib)
        yaw_fit, yaw_info = self._fit_yaw_distance(yaws, self.calib)

        if pos_fit is None and yaw_fit is None:
            print("[FIT] nothing to fit (need center>=6 or yaw>=2).")
            return False

        A0, A1, b0, b1 = self.calib.A0.copy(), self.calib.A1.copy(), self.calib.b0.copy(), self.calib.b1.copy()
        yaw_a, yaw_k = self.calib.yaw_a, self.calib.yaw_k

        if pos_fit is not None:
            A0, A1, b0, b1 = pos_fit
        if yaw_fit is not None:
            yaw_a, yaw_k = yaw_fit

        new_calib = CalibModel(
            version=vNp1,
            A0=A0, A1=A1, b0=b0, b1=b1,
            yaw_a=yaw_a, yaw_k=yaw_k,
            created_at=time.time(),
            parent_version=vN,
            made_from_raw_version=vN,
        )

        fit_stats = {"center_fit": pos_info, "yaw_fit": yaw_info}
        params_used = {
            "dist_gain_fixed": 0.90,
            "model": "p*=(A0+dA1)p+(b0+db1), yaw=a+kd",
            "robust": "soft IRLS 2 rounds",
        }
        self._save_calib(new_calib, fit_stats=fit_stats, params_used=params_used)

        # switch
        self.active_version = vNp1
        self.calib = new_calib
        self.raw_samples = self._load_or_create_raw(self.active_version)
        self._flush_raw()

        print(f"[SWITCH] ACTIVE calib_v{self.active_version}.json, RECORD raw_v{self.active_version}.json")
        return True

    # ============================================================
    # ✅ Video read helper (loop)
    # ============================================================
    def _read_pair(self):
        ok0, fr0 = self.cap_rear.read()
        ok1, fr1 = self.cap_left.read()

        if USE_VIDEO_FILES and LOOP_VIDEO:
            if not ok0:
                self.cap_rear.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok0, fr0 = self.cap_rear.read()
            if not ok1:
                self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok1, fr1 = self.cap_left.read()

        return ok0, fr0, ok1, fr1

    # ============================================================
    # Main loop
    # ============================================================
    def run(self):
        while True:
            self.update_from_trackbars()

            ok0, fr0, ok1, fr1 = self._read_pair()
            if not ok0 or not ok1:
                print("[END] input stream ended")
                break

            dets = []
            dets += self.estimate_from_frame(fr0, self.cams["rear"])
            dets += self.estimate_from_frame(fr1, self.cams["left"])

            # per-cam best
            cam_best = {"rear": None, "left": None}
            for d in dets:
                ck = d["cam_key"]
                if cam_best[ck] is None or d["weight"] > cam_best[ck]["weight"]:
                    cam_best[ck] = d

            for ck in ("rear", "left"):
                if cam_best[ck] is None:
                    self.latest_cam_obs[ck] = {"seen": False, "ground_m": None, "quality": None}
                else:
                    q = float(cam_best[ck]["dbg_quality"]["quality"])
                    gm = float(cam_best[ck]["dbg_quality"]["ground_m"])
                    self.latest_cam_obs[ck] = {"seen": True, "ground_m": gm, "quality": q}

            fused = self.fuse(dets)

            if fused is not None:
                self.lost_count = 0
                center_meas, heading_meas, d_fused = fused
                self.latest_fused_d = d_fused

                self.buf_center.append(center_meas.copy())
                self.buf_sin.append(math.sin(heading_meas))
                self.buf_cos.append(math.cos(heading_meas))

                robust_center, robust_heading, _ = self._robust_measurement_from_buffer(center_meas, heading_meas)

                if self.is_initialized:
                    dyaw = abs(wrap_pi(robust_heading - self.raw_heading))
                    a = self.alpha_fast if math.degrees(dyaw) > self.turn_fast_deg else self.alpha
                    self.raw_center = self.raw_center * (1.0 - a) + robust_center * a
                    self.raw_heading = wrap_pi(self.raw_heading + wrap_pi(robust_heading - self.raw_heading) * a)
                else:
                    self.raw_center = robust_center
                    self.raw_heading = robust_heading
                    self.is_initialized = True
            else:
                self.latest_fused_d = None
                self.lost_count += 1
                if self.lost_count >= self.lost_reset_frames:
                    self.buf_center.clear()
                    self.buf_sin.clear()
                    self.buf_cos.clear()
                    self.is_initialized = False
                    self.lost_count = 0

            # update snapshot
            self.latest_raw_center = None if self.raw_center is None else self.raw_center.copy()
            self.latest_raw_heading = float(self.raw_heading)
            corr = self.corrected_pose()
            if corr is not None:
                self.latest_corr_center = corr[0].copy()
                self.latest_corr_heading = float(corr[1])
            else:
                self.latest_corr_center = None
                self.latest_corr_heading = 0.0

            # draw
            m = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m)

            for d in dets:
                mp = tuple(d["marker_pos"].astype(int))
                cc = tuple(d["center_pos"].astype(int))
                cv2.circle(m, mp, 4, (0, 255, 255), -1)
                cv2.circle(m, cc, 3, (255, 180, 0), -1)
                cv2.putText(m, f"ID{d['marker_id']}/{d['cam_key']}", (mp[0] + 6, mp[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            if corr is not None:
                cpos, chead = corr
                self.draw_wheelchair(m, cpos, chead, color=(0, 255, 0), label=f"CORR v{self.active_version}")

            self.draw_hud(m, dets)

            mon0 = cv2.resize(fr0, (640, 360))
            mon1 = cv2.resize(fr1, (640, 360))
            cv2.imshow(self.win_mon, np.hstack([mon0, mon1]))
            cv2.imshow(self.win_map, m)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            if key == ord('r'):
                self.is_initialized = False
                self.raw_center = None
                self.raw_heading = 0.0
                self.buf_center.clear()
                self.buf_sin.clear()
                self.buf_cos.clear()
                self.lost_count = 0
                print("[RESET] tracking filter reset (calib unchanged)")

            if key == ord('c'):
                self.click_mode = "record_center"
                self.tmp_yaw_center = None
                print(f"[MODE] record center -> raw_v{self.active_version}.json")

            if key == ord('a'):
                self.click_mode = "record_yaw_center"
                self.tmp_yaw_center = None
                print(f"[MODE] record yaw -> raw_v{self.active_version}.json (center then direction)")

            if key == ord('s'):
                self.build_next_calib_from_raw()

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    WheelchairTracker().run()
