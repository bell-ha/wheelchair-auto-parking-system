#!/usr/bin/env python3
# wc_tracker_with_click_calib_accum_errweight_px1cm_distbin_full.py

import cv2
import numpy as np
import math
import time
import json
import os
from dataclasses import dataclass

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
    dist_gain: float = 1.0


class WheelchairTracker:
    def __init__(self):
        # =========================
        # Map params
        # =========================
        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720

        # ✅ 1px = 1cm
        self.map_scale = 1.0

        self.off_x, self.off_y = 200, 150

        # car zone (그대로 유지)
        self.car_zone = ((200 + self.off_x, 180 + self.off_y),
                         (400 + self.off_x, 540 + self.off_y))

        # wheelchair size (cm)
        self.wc_w_cm, self.wc_l_cm = 55.0, 66.0

        # marker height (cm)
        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0

        # marker->center offset (cm)
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # smoothing
        self.alpha = 0.30

        # quality-based weight params
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # =========================
        # Click sample err-based weight
        # =========================
        self.center_sigma_px = 60.0
        self.yaw_sigma_deg = 20.0
        self.min_click_w = 0.02
        self.last_click_info = None

        # =========================
        # NEW: 거리 구간별 center calibration
        # =========================
        # m 단위: near(0~2), mid(2~4), far(4~)
        self.dist_bins_m = [0.0, 2.0, 4.0, 999.0]
        self.dist_blend_m = 0.4  # 경계 부드럽게 보간할 폭

        self.calib_dxdy_bins = {
            "near": np.array([0.0, 0.0], dtype=np.float32),
            "mid":  np.array([0.0, 0.0], dtype=np.float32),
            "far":  np.array([0.0, 0.0], dtype=np.float32),
        }

        # 최신 거리/최고 det (HUD/보정 적용 기준)
        self.last_best_ground_m = None
        self.last_best_det = None
        self.last_best_quality = None

        # =========================
        # Camera configs
        # =========================
        # ✅ dist_gain은 네가 원한 값 0.9로 고정(트랙바로 조절도 가능)
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
        # Calibration storage
        # =========================
        self.calib_path = "wc_calibration.json"
        self.samples = []

        # 구버전 호환용(남겨둠)
        self.calib_dxdy = np.array([0.0, 0.0], dtype=np.float32)  # pixels
        self.calib_yaw_offset = 0.0  # radians

        self._load_calibration()

        # =========================
        # Camera open
        # =========================
        self.cap_rear = cv2.VideoCapture(self.cams["rear"].index)
        self.cap_left = cv2.VideoCapture(self.cams["left"].index)
        if not self.cap_rear.isOpened() or not self.cap_left.isOpened():
            raise RuntimeError("카메라 오픈 실패: 인덱스(0/1) 확인")

        # fused (raw)
        self.raw_center = None
        self.raw_heading = 0.0
        self.is_initialized = False

        # =========================
        # UI
        # =========================
        self.win_map = "minimap"
        self.win_mon = "monitor(rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)

        # ✅ 반드시 존재하도록 보장(너가 겪은 AttributeError 방지)
        self.click_mode = None  # None | set_center | set_yaw_center | set_yaw_dir
        self.tmp_yaw_center = None

        cv2.setMouseCallback(self.win_map, self._on_mouse)

        cv2.createTrackbar("Smooth(0-100)", self.win_map, int(self.alpha * 100), 100, lambda v: None)
        cv2.createTrackbar("Rear_DistGain(x100)", self.win_map, int(self.cams["rear"].dist_gain * 100), 500, lambda v: None)
        cv2.createTrackbar("Left_DistGain(x100)", self.win_map, int(self.cams["left"].dist_gain * 100), 500, lambda v: None)
        cv2.createTrackbar("CenterSigma(px)", self.win_map, int(self.center_sigma_px), 300, lambda v: None)
        cv2.createTrackbar("YawSigma(deg)", self.win_map, int(self.yaw_sigma_deg), 90, lambda v: None)

        print("[Keys]")
        print("  q/ESC : quit")
        print("  r     : reset fuse (raw state reset)")
        print("  c     : center calibration (click 1 point)  [거리 구간별 저장]")
        print("  a     : yaw calibration (click center then direction)")
        print("  s     : save (recompute weighted mean + write file)")

    # -------------------------
    # Calibration file I/O
    # -------------------------
    def _load_calibration(self):
        if not os.path.exists(self.calib_path):
            return
        try:
            with open(self.calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.samples = data.get("samples", []) or []

            # 구버전 필드
            dx = float(data.get("calib_dx", 0.0))
            dy = float(data.get("calib_dy", 0.0))
            yo = float(data.get("calib_yaw_offset", 0.0))
            self.calib_dxdy = np.array([dx, dy], dtype=np.float32)
            self.calib_yaw_offset = yo

            # 신버전 필드(있으면 읽기)
            bins = data.get("calib_dxdy_bins", None)
            if isinstance(bins, dict):
                for k in ("near", "mid", "far"):
                    if k in bins and isinstance(bins[k], (list, tuple)) and len(bins[k]) == 2:
                        self.calib_dxdy_bins[k] = np.array([float(bins[k][0]), float(bins[k][1])], dtype=np.float32)

            # 저장된 dist bin 경계도 있으면 읽기
            db = data.get("dist_bins_m", None)
            if isinstance(db, list) and len(db) >= 4:
                self.dist_bins_m = [float(db[0]), float(db[1]), float(db[2]), float(db[3])]

            print(f"[LOAD] {self.calib_path} samples={len(self.samples)} yaw_off={math.degrees(self.calib_yaw_offset):.1f}deg")
            print(f"       calib_dxdy_bins near={self.calib_dxdy_bins['near']} mid={self.calib_dxdy_bins['mid']} far={self.calib_dxdy_bins['far']}")
        except Exception as e:
            print("[WARN] calibration load failed:", e)

    def _save_calibration(self):
        data = {
            "saved_at": time.time(),
            "calib_yaw_offset": float(self.calib_yaw_offset),

            # 구버전 호환용
            "calib_dx": float(self.calib_dxdy[0]),
            "calib_dy": float(self.calib_dxdy[1]),

            # 신버전: 거리별 dxdy
            "calib_dxdy_bins": {
                "near": [float(self.calib_dxdy_bins["near"][0]), float(self.calib_dxdy_bins["near"][1])],
                "mid":  [float(self.calib_dxdy_bins["mid"][0]),  float(self.calib_dxdy_bins["mid"][1])],
                "far":  [float(self.calib_dxdy_bins["far"][0]),  float(self.calib_dxdy_bins["far"][1])],
            },
            "dist_bins_m": self.dist_bins_m,
            "samples": self.samples,
        }
        with open(self.calib_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SAVE] {self.calib_path} samples={len(self.samples)} yaw_off={math.degrees(self.calib_yaw_offset):.1f}deg")
        print(f"       bins near={self.calib_dxdy_bins['near']} mid={self.calib_dxdy_bins['mid']} far={self.calib_dxdy_bins['far']}")

    # -------------------------
    # Trackbars
    # -------------------------
    def update_from_trackbars(self):
        self.alpha = max(0.01, cv2.getTrackbarPos("Smooth(0-100)", self.win_map) / 100.0)
        rdg = cv2.getTrackbarPos("Rear_DistGain(x100)", self.win_map) / 100.0
        ldg = cv2.getTrackbarPos("Left_DistGain(x100)", self.win_map) / 100.0
        self.cams["rear"].dist_gain = max(0.01, rdg)
        self.cams["left"].dist_gain = max(0.01, ldg)

        self.center_sigma_px = max(5.0, float(cv2.getTrackbarPos("CenterSigma(px)", self.win_map)))
        self.yaw_sigma_deg = max(1.0, float(cv2.getTrackbarPos("YawSigma(deg)", self.win_map)))

    # -------------------------
    # Helpers
    # -------------------------
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

    def _click_quality_weight(self):
        if self.last_best_quality is None:
            return 1.0
        return float(max(0.05, min(1.0, self.last_best_quality)))

    def _err_weight_center(self, err_px: float) -> float:
        sigma = float(self.center_sigma_px)
        w = math.exp(-(err_px * err_px) / (2.0 * sigma * sigma))
        return float(max(self.min_click_w, min(1.0, w)))

    def _err_weight_yaw(self, err_rad: float) -> float:
        sigma = math.radians(float(self.yaw_sigma_deg))
        w = math.exp(-(err_rad * err_rad) / (2.0 * sigma * sigma))
        return float(max(self.min_click_w, min(1.0, w)))

    def _bin_name_for_dist(self, r_m: float) -> str:
        if r_m < self.dist_bins_m[1]:
            return "near"
        if r_m < self.dist_bins_m[2]:
            return "mid"
        return "far"

    def _dxdy_for_dist(self, r_m: float) -> np.ndarray:
        """
        현재 거리 r_m에 따라 dxdy를 선택.
        - bin 값 사용
        - bin 경계 근처에서는 선형 보간(부드럽게)
        """
        near_end = float(self.dist_bins_m[1])
        mid_end = float(self.dist_bins_m[2])
        b = float(self.dist_blend_m)

        if r_m < near_end - b:
            return self.calib_dxdy_bins["near"]
        if r_m > mid_end + b:
            return self.calib_dxdy_bins["far"]
        if (near_end + b) <= r_m <= (mid_end - b):
            return self.calib_dxdy_bins["mid"]

        # near <-> mid blend
        if (near_end - b) <= r_m <= (near_end + b):
            t = (r_m - (near_end - b)) / (2.0 * b)
            return (1 - t) * self.calib_dxdy_bins["near"] + t * self.calib_dxdy_bins["mid"]

        # mid <-> far blend
        if (mid_end - b) <= r_m <= (mid_end + b):
            t = (r_m - (mid_end - b)) / (2.0 * b)
            return (1 - t) * self.calib_dxdy_bins["mid"] + t * self.calib_dxdy_bins["far"]

        return self.calib_dxdy_bins[self._bin_name_for_dist(r_m)]

    def corrected_pose(self):
        if self.raw_center is None:
            return None
        r = float(self.last_best_ground_m) if self.last_best_ground_m is not None else 0.0
        dxdy = self._dxdy_for_dist(r)
        c = self.raw_center + dxdy
        h = wrap_pi(self.raw_heading + self.calib_yaw_offset)
        return c, h

    # -------------------------
    # Click sampling + recompute
    # -------------------------
    def add_center_sample(self, clicked_xy: np.ndarray, weight: float):
        if self.raw_center is None:
            return
        r = float(self.last_best_ground_m) if self.last_best_ground_m is not None else 0.0
        bin_name = self._bin_name_for_dist(r)
        dxdy_need = (clicked_xy - self.raw_center).astype(np.float32)

        self.samples.append({
            "type": "center",
            "bin": bin_name,
            "r_m": float(r),
            "dx": float(dxdy_need[0]),
            "dy": float(dxdy_need[1]),
            "w": float(max(0.001, weight)),
            "t": time.time(),
        })

    def add_yaw_sample(self, clicked_heading_rad: float, weight: float):
        if not self.is_initialized:
            return
        off = wrap_pi(float(clicked_heading_rad) - float(self.raw_heading))
        self.samples.append({
            "type": "yaw",
            "off": float(off),
            "w": float(max(0.001, weight)),
            "t": time.time(),
        })

    def recompute_calibration_from_samples(self):
        # center bins weighted mean
        acc = {
            "near": {"sx": 0.0, "sy": 0.0, "sw": 0.0},
            "mid":  {"sx": 0.0, "sy": 0.0, "sw": 0.0},
            "far":  {"sx": 0.0, "sy": 0.0, "sw": 0.0},
        }

        for s in self.samples:
            if s.get("type") != "center":
                continue
            w = float(s.get("w", 1.0))
            b = s.get("bin", "mid")
            if b not in acc:
                b = "mid"
            acc[b]["sx"] += float(s["dx"]) * w
            acc[b]["sy"] += float(s["dy"]) * w
            acc[b]["sw"] += w

        for b in ("near", "mid", "far"):
            if acc[b]["sw"] > 1e-9:
                self.calib_dxdy_bins[b] = np.array([acc[b]["sx"] / acc[b]["sw"],
                                                    acc[b]["sy"] / acc[b]["sw"]],
                                                   dtype=np.float32)

        # yaw weighted circular mean
        ss = cc = sw2 = 0.0
        for s in self.samples:
            if s.get("type") != "yaw":
                continue
            w = float(s.get("w", 1.0))
            off = float(s["off"])
            ss += math.sin(off) * w
            cc += math.cos(off) * w
            sw2 += w
        if sw2 > 1e-9:
            self.calib_yaw_offset = math.atan2(ss, cc)

        # 구버전용 전역 dxdy는 mid를 복사
        self.calib_dxdy = self.calib_dxdy_bins["mid"].copy()

    # -------------------------
    # Mouse callback
    # -------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pt = np.array([float(x), float(y)], dtype=np.float32)
        wq = self._click_quality_weight()

        # center mode
        if self.click_mode == "set_center":
            corr = self.corrected_pose()
            if corr is None:
                self.click_mode = None
                return

            corr_center, _ = corr
            err_px = float(np.linalg.norm(pt - corr_center))
            we = self._err_weight_center(err_px)
            w = float(wq * we)

            # ✅ "현재 거리 bin"의 dxdy만 즉시 업데이트
            r = float(self.last_best_ground_m) if self.last_best_ground_m is not None else 0.0
            bname = self._bin_name_for_dist(r)
            delta = (pt - corr_center).astype(np.float32)
            self.calib_dxdy_bins[bname] = self.calib_dxdy_bins[bname] + delta

            # 샘플 누적(raw 기준)
            self.add_center_sample(pt, w)

            self.last_click_info = {
                "mode": "center",
                "err_px": err_px,
                "wq": wq,
                "we": we,
                "w": w,
                "r_m": r,
                "bin": bname
            }

            self.click_mode = None
            print(f"[CENTER CLICK] bin={bname} r={r:.2f}m err={err_px:.1f}px w={w:.2f} samples={len(self.samples)}")
            return

        # yaw step1
        if self.click_mode == "set_yaw_center":
            self.tmp_yaw_center = pt
            self.click_mode = "set_yaw_dir"
            print("[YAW CLICK] center picked. Now click direction point.")
            return

        # yaw step2
        if self.click_mode == "set_yaw_dir":
            if self.tmp_yaw_center is None:
                self.click_mode = None
                return

            v = (pt - self.tmp_yaw_center).astype(np.float32)
            if float(np.linalg.norm(v)) < 1e-3:
                print("[YAW CLICK] direction too small. retry.")
                return

            clicked_heading = float(math.atan2(float(v[1]), float(v[0])))

            corr = self.corrected_pose()
            if corr is None:
                self.click_mode = None
                self.tmp_yaw_center = None
                return

            _, corr_heading = corr
            err_rad = float(wrap_pi(clicked_heading - corr_heading))
            we = self._err_weight_yaw(abs(err_rad))
            w = float(wq * we)

            d = wrap_pi(clicked_heading - corr_heading)
            self.calib_yaw_offset = wrap_pi(self.calib_yaw_offset + d)

            self.add_yaw_sample(clicked_heading, w)

            self.last_click_info = {
                "mode": "yaw",
                "err_deg": float(abs(math.degrees(err_rad))),
                "wq": wq,
                "we": we,
                "w": w
            }

            self.click_mode = None
            self.tmp_yaw_center = None
            print(f"[YAW CLICK] err={abs(math.degrees(err_rad)):.1f}deg w={w:.2f} samples={len(self.samples)}")
            return

    # -------------------------
    # Detection / estimation
    # -------------------------
    def estimate_from_frame(self, frame, cam: CamCfg):
        dets = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return dets

        for i, mid_arr in enumerate(ids):
            mid = int(mid_arr[0])
            if mid not in (0, 1):
                continue

            c2 = corners[i].reshape(4, 2).astype(np.float32)

            # fisheye undistort
            und = cv2.fisheye.undistortPoints(
                corners[i].reshape(-1, 1, 2),
                K, D, P=K
            )

            ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            dist_m = float(np.linalg.norm(tvec))

            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * cam.dist_gain

            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
            ray_deg = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cam.pos_px + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            # yaw from rvec
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

            # quality calc
            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(proj - und.reshape(-1, 2).astype(np.float32), axis=1)))

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
        return center, heading

    # -------------------------
    # Drawing
    # -------------------------
    def draw_static_map(self, img):
        # ✅ 1000x1000 전체 격자도 그리기 + 600x720 유효영역 격자 강조
        grid_step_cm = 20   # 20cm 간격
        step = max(1, int(grid_step_cm * self.map_scale))     # 1px=1cm -> 20px
        major_step = max(1, int(100 * self.map_scale))        # 1m 간격(굵게)

        # 전체 1000x1000 그리드(연하게)
        for x in range(0, self.map_w + 1, step):
            col = (20, 20, 20) if (x % major_step) != 0 else (40, 40, 40)
            cv2.line(img, (x, 0), (x, self.map_h), col, 1)
        for y in range(0, self.map_h + 1, step):
            col = (20, 20, 20) if (y % major_step) != 0 else (40, 40, 40)
            cv2.line(img, (0, y), (self.map_w, y), col, 1)

        # 유효영역 600x720 격자(조금 진하게)
        for x in range(0, self.grid_w + 1, step):
            col = (45, 45, 45) if (x % major_step) != 0 else (85, 85, 85)
            cv2.line(img, (self.off_x + x, self.off_y),
                     (self.off_x + x, self.off_y + self.grid_h), col, 1)
        for y in range(0, self.grid_h + 1, step):
            col = (45, 45, 45) if (y % major_step) != 0 else (85, 85, 85)
            cv2.line(img, (self.off_x, self.off_y + y),
                     (self.off_x + self.grid_w, self.off_y + y), col, 1)

        # 유효영역 경계
        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h), (200, 200, 200), 2)

        # car zone
        cv2.rectangle(img, self.car_zone[0], self.car_zone[1], (35, 35, 45), -1)

        # cameras
        for cam in self.cams.values():
            cp = tuple(cam.pos_px.astype(int))
            cv2.circle(img, cp, 6, (220, 220, 220), -1)
            cv2.putText(img, cam.key, (cp[0] + 8, cp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    def draw_wheelchair(self, img, center, heading, color=(0, 255, 0), label="FUSED(corr)"):
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
        # 방어: 혹시 합치다 누락돼도 죽지 않게
        if not hasattr(self, "click_mode"):
            self.click_mode = None

        y = 20
        cv2.putText(img, f"alpha={self.alpha:.2f}", (10, y), 0, 0.6, (220, 220, 220), 2, cv2.LINE_AA); y += 22
        cv2.putText(img, f"rear_gain={self.cams['rear'].dist_gain:.2f}  left_gain={self.cams['left'].dist_gain:.2f}",
                    (10, y), 0, 0.6, (220, 220, 220), 2, cv2.LINE_AA); y += 22

        r = float(self.last_best_ground_m) if self.last_best_ground_m is not None else 0.0
        bname = self._bin_name_for_dist(r)
        dxdy_now = self._dxdy_for_dist(r)
        cv2.putText(img, f"dist={r:.2f}m bin={bname} dxdy=({dxdy_now[0]:.1f},{dxdy_now[1]:.1f})",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 20

        cv2.putText(img, f"CenterSigma={self.center_sigma_px:.0f}px  YawSigma={self.yaw_sigma_deg:.0f}deg",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 20

        cv2.putText(img, f"yaw_off={math.degrees(self.calib_yaw_offset):.1f}deg  samples={len(self.samples)} dets={len(dets)}",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 20

        if self.click_mode == "set_center":
            cv2.putText(img, "MODE: center -> click TRUE center point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "set_yaw_center":
            cv2.putText(img, "MODE: yaw -> click CENTER point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "set_yaw_dir":
            cv2.putText(img, "MODE: yaw -> click DIRECTION point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24

        if dets:
            best = max(dets, key=lambda d: d["weight"])
            q = best.get("dbg_quality", {})
            self.last_best_quality = float(q.get("quality", 0.5))
            cv2.putText(img, f"best[{best['cam_key']}/ID{best['marker_id']}] q={q.get('quality',0):.2f} area={q.get('area',0):.0f} reproj={q.get('reproj',0):.2f}",
                        (10, y), 0, 0.5, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            self.last_best_quality = None

        if self.last_click_info is not None:
            y += 18
            if self.last_click_info.get("mode") == "center":
                cv2.putText(img,
                            f"lastClick(center): bin={self.last_click_info.get('bin','?')} r={self.last_click_info.get('r_m',0):.2f}m "
                            f"err={self.last_click_info['err_px']:.1f}px w={self.last_click_info['w']:.2f}",
                            (10, y), 0, 0.5, (180, 180, 180), 2, cv2.LINE_AA)
            else:
                cv2.putText(img,
                            f"lastClick(yaw): err={self.last_click_info.get('err_deg',0):.1f}deg w={self.last_click_info.get('w',0):.2f}",
                            (10, y), 0, 0.5, (180, 180, 180), 2, cv2.LINE_AA)

    # -------------------------
    # Loop
    # -------------------------
    def run(self):
        while True:
            self.update_from_trackbars()

            ok0, fr0 = self.cap_rear.read()
            ok1, fr1 = self.cap_left.read()
            if not ok0 or not ok1:
                break

            dets = []
            dets += self.estimate_from_frame(fr0, self.cams["rear"])
            dets += self.estimate_from_frame(fr1, self.cams["left"])

            # best det info
            if dets:
                best = max(dets, key=lambda d: d["weight"])
                self.last_best_det = best
                self.last_best_ground_m = float(best.get("dbg_quality", {}).get("ground_m", 0.0))
            else:
                self.last_best_det = None
                self.last_best_ground_m = None

            fused = self.fuse(dets)
            if fused is not None:
                center, heading = fused
                if self.is_initialized:
                    self.raw_center = self.raw_center * (1 - self.alpha) + center * self.alpha
                    diff = wrap_pi(heading - self.raw_heading)
                    self.raw_heading = wrap_pi(self.raw_heading + diff * self.alpha)
                else:
                    self.raw_center = center
                    self.raw_heading = heading
                    self.is_initialized = True

            m = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m)

            for d in dets:
                mp = tuple(d["marker_pos"].astype(int))
                cc = tuple(d["center_pos"].astype(int))
                cv2.circle(m, mp, 4, (0, 255, 255), -1)
                cv2.circle(m, cc, 3, (255, 180, 0), -1)
                cv2.putText(m, f"ID{d['marker_id']}/{d['cam_key']}", (mp[0] + 6, mp[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            corr = self.corrected_pose()
            if corr is not None:
                cpos, chead = corr
                self.draw_wheelchair(m, cpos, chead, color=(0, 255, 0), label="FUSED(corr)")

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
                print("[RESET] raw fused state reset")

            if key == ord('c'):
                self.click_mode = "set_center"
                self.tmp_yaw_center = None
                print("[MODE] center calibration: click TRUE center point (거리별 bin 저장)")

            if key == ord('a'):
                self.click_mode = "set_yaw_center"
                self.tmp_yaw_center = None
                print("[MODE] yaw calibration: click CENTER then DIRECTION")

            if key == ord('s'):
                self.recompute_calibration_from_samples()
                self._save_calibration()
                print("[APPLY] recomputed from samples and saved (distance-bins)")

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    WheelchairTracker().run()
