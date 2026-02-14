#!/usr/bin/env python3
# wc_tracker_versioned_calib_dtheta_nosmooth.py
#
# 목표:
#  - smoothing(EMA/버퍼) 없이 "프레임별 추정 raw"를 그대로 사용해서 오차를 관찰
#  - c/a는 raw 샘플만 기록(즉시 보정 반영 X)
#  - s를 누르면 raw_vN으로부터 calib_v(N+1) 생성 + 즉시 적용(버전 스위치)
#  - raw_vN.json은 "해당 버전에서 처음 c/a로 기록할 때" 생성됨
#
# 보정 모델(거리 d + heading θ 의존):
#  - 좌표: p_corr = p_raw + Δp(d,θ)
#        Δx = bx · [1, d, cosθ, sinθ, dcosθ, dsinθ]
#        Δy = by · [1, d, cosθ, sinθ, dcosθ, dsinθ]
#  - 각도: θ_corr = wrap( θ_raw + Δθ(d,θ) )
#        Δθ = bt · [1, d, cosθ, sinθ, dcosθ, dsinθ]
#
# Keys:
#   q/ESC: quit
#   r    : reset tracking (pose만)
#   c    : record CENTER sample (click true center)
#   a    : record YAW sample (click center then direction)
#   s    : fit -> save calib_v(N+1) and switch to it
#
# Optional:
#   --rear 0 --left 1              (camera indices)
#   --rear ../command/1_rear.mp4   (video file)
#   --left ../command/1_left.mp4

import cv2
import numpy as np
import math
import time
import json
import os
import glob
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, List, Optional


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


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg


def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(아래)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)


def _parse_src(s: str) -> Any:
    try:
        if s.strip().isdigit():
            return int(s.strip())
    except Exception:
        pass
    return s


def _open_capture(src: Any) -> cv2.VideoCapture:
    return cv2.VideoCapture(src)


@dataclass
class CamCfg:
    key: str
    index: Any  # int or str path
    pos_px: np.ndarray
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90  # fixed


@dataclass
class CalibModel:
    version: int
    # Δx = bx·phi, Δy = by·phi, Δθ = bt·phi
    bx: np.ndarray  # (6,)
    by: np.ndarray  # (6,)
    bt: np.ndarray  # (6,)
    created_at: float
    parent_version: int = 0
    made_from_raw_version: int = 0

    @staticmethod
    def phi(d: float, theta: float) -> np.ndarray:
        d = float(d if d is not None else 0.0)
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        return np.array([1.0, d, c, s, d * c, d * s], dtype=np.float32)

    def transform_pos(self, p_raw: np.ndarray, d: float, theta: float) -> np.ndarray:
        ph = self.phi(d, theta)
        dx = float(np.dot(self.bx, ph))
        dy = float(np.dot(self.by, ph))
        return p_raw.astype(np.float32).reshape(2,) + np.array([dx, dy], dtype=np.float32)

    def transform_yaw(self, yaw_raw: float, d: float, theta: float) -> float:
        ph = self.phi(d, theta)
        dth = float(np.dot(self.bt, ph))
        return wrap_pi(float(yaw_raw) + dth)


@dataclass
class CamGateState:
    gate_hist: deque
    stab_hist: deque
    active: bool = False
    last_det: Optional[Dict[str, Any]] = None
    last_seen_frame: int = -10_000
    prev_seen: bool = False
    ramp_seen: int = 0


class WheelchairTracker:
    def __init__(self, rear_src: Any = 0, left_src: Any = 1):
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
        # Flicker handling (gate/hold)
        # =========================
        self.gate_W = 5
        self.gate_H = 2
        self.hold_decay = 0.80
        self.hold_max_age = 10
        self.stab_N = 20
        self.stab_min = 0.30
        self.reentry_ramp_len = 3
        self.consistency_min = 0.15

        # =========================
        # Tilt(view angle) weighting
        # =========================
        self.tilt_good_deg = 15.0
        self.tilt_bad_deg = 60.0
        self.view_min = 0.20

        # =========================
        # Quality
        # =========================
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # =========================
        # Camera configs (dist_gain fixed)
        # =========================
        self.cams = {
            "rear": CamCfg(
                key="rear", index=rear_src,
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
                key="left", index=left_src,
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

        self.cam_gate: Dict[str, CamGateState] = {
            "rear": CamGateState(gate_hist=deque(maxlen=self.gate_W), stab_hist=deque(maxlen=self.stab_N)),
            "left": CamGateState(gate_hist=deque(maxlen=self.gate_W), stab_hist=deque(maxlen=self.stab_N)),
        }

        # =========================
        # Versioned calibration / raw data
        # =========================
        self.active_version = self._find_or_create_latest_calib()
        self.calib = self._load_calib(self.active_version)

        # raw samples: 파일이 있으면 로드, 없으면 [] (파일 생성은 첫 기록 시)
        self.raw_samples: List[dict] = self._load_raw_if_exists(self.active_version)
        self.raw_file_created = os.path.exists(self._raw_path(self.active_version))

        # =========================
        # Camera open
        # =========================
        self.cap_rear = _open_capture(self.cams["rear"].index)
        self.cap_left = _open_capture(self.cams["left"].index)
        if not self.cap_rear.isOpened() or not self.cap_left.isOpened():
            raise RuntimeError(
                "카메라/비디오 오픈 실패.\n"
                "예) --rear 0 --left 1  또는  --rear ../command/1_rear.mp4 --left ../command/1_left.mp4"
            )

        # =========================
        # NO SMOOTHING: raw pose == fused pose (프레임마다)
        # =========================
        self.raw_center: Optional[np.ndarray] = None
        self.raw_heading: float = 0.0
        self.is_initialized = False

        # 거리(보정용) - FREEZE에서도 마지막 값 유지
        self.latest_fused_d: Optional[float] = None

        # per-frame caches (for click saving)
        self.latest_raw_center: Optional[np.ndarray] = None
        self.latest_raw_heading: float = 0.0
        self.latest_corr_center: Optional[np.ndarray] = None
        self.latest_corr_heading: float = 0.0

        self.latest_cam_obs = {
            "rear": {"seen": False, "active": False, "held": False, "ground_m": None, "quality": None, "stability": None, "tilt_deg": None},
            "left": {"seen": False, "active": False, "held": False, "ground_m": None, "quality": None, "stability": None, "tilt_deg": None},
        }

        self.lost_count = 0
        self.lost_reset_frames = 12

        # UI
        self.win_map = "minimap"
        self.win_mon = "monitor(rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_map, self._on_mouse)

        # click mode
        self.click_mode = None  # None | record_center | record_yaw_center | record_yaw_dir
        self.tmp_yaw_center = None

        self.frame_idx = 0

        print("[NO-SMOOTH Versioned Calib]")
        print(f"  ACTIVE calib_v{self.active_version}.json")
        print(f"  RAW    raw_v{self.active_version}.json  (created={self.raw_file_created}, samples={len(self.raw_samples)})")
        print("[Keys] q/ESC quit | r reset pose | c record center | a record yaw | s fit->next version")
        print("[Note] raw 파일은 '첫 c/a 기록' 때 생성됩니다.")
        print("[UI] RAW(주황 틀)는 제거됨. CORR(초록)만 표시합니다.")

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
        vers: List[int] = []
        for f in files:
            base = os.path.basename(f)
            try:
                n = int(base.replace("calib_v", "").replace(".json", ""))
                vers.append(n)
            except Exception:
                pass
        if vers:
            return max(vers)

        # create v1 default (all zeros => no correction)
        v1 = 1
        default = CalibModel(
            version=v1,
            bx=np.zeros((6,), dtype=np.float32),
            by=np.zeros((6,), dtype=np.float32),
            bt=np.zeros((6,), dtype=np.float32),
            created_at=time.time(),
            parent_version=0,
            made_from_raw_version=0,
        )
        self._save_calib(default, fit_stats={"msg": "init zero model"}, params_used={"model": "delta(d,theta) linear"})
        print("[INIT] created calib_v1.json (zero correction)")
        return v1

    def _load_calib(self, ver: int) -> CalibModel:
        path = self._calib_path(ver)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data.get("model", {})
        bx = np.array(m.get("bx", [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        by = np.array(m.get("by", [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        bt = np.array(m.get("bt", [0, 0, 0, 0, 0, 0]), dtype=np.float32)

        return CalibModel(
            version=int(data.get("version", ver)),
            bx=bx, by=by, bt=bt,
            created_at=float(data.get("created_at", time.time())),
            parent_version=int(data.get("parent_version", 0)),
            made_from_raw_version=int(data.get("made_from_raw_version", 0)),
        )

    def _save_calib(self, calib: CalibModel, fit_stats: Optional[dict] = None, params_used: Optional[dict] = None):
        path = self._calib_path(calib.version)
        data = {
            "schema_version": 1,
            "version": calib.version,
            "parent_version": calib.parent_version,
            "made_from_raw_version": calib.made_from_raw_version,
            "created_at": calib.created_at,
            "dist_gain_fixed": 0.90,
            "model": {
                "type": "delta_linear_d_theta",
                "phi": ["1", "d", "cos(theta)", "sin(theta)", "d*cos(theta)", "d*sin(theta)"],
                "bx": calib.bx.tolist(),
                "by": calib.by.tolist(),
                "bt": calib.bt.tolist(),
            },
            "fit_stats": fit_stats or {},
            "params_used": params_used or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {path}")

    def _load_raw_if_exists(self, ver: int) -> List[dict]:
        path = self._raw_path(ver)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("samples", []) or []
        except Exception as e:
            print("[WARN] raw load failed:", e)
            return []

    def _ensure_raw_file(self):
        if self.raw_file_created:
            return
        path = self._raw_path(self.active_version)
        data = {
            "schema_version": 1,
            "calib_version": self.active_version,
            "created_at": time.time(),
            "samples": []
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.raw_file_created = True
        print(f"[INIT] created {path}")

    def _flush_raw(self):
        self._ensure_raw_file()
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
    def smooth01_small_is_good(x: float, x_good: float, x_bad: float) -> float:
        # small is good: 1 at <=good, 0 at >=bad
        if x <= x_good:
            return 1.0
        if x >= x_bad:
            return 0.0
        t = (x - x_good) / (x_bad - x_good)
        return float(1.0 - t)

    @staticmethod
    def smooth01_large_is_good(x: float, x_bad: float, x_good: float) -> float:
        # large is good: 0 at <=bad, 1 at >=good
        if x <= x_bad:
            return 0.0
        if x >= x_good:
            return 1.0
        t = (x - x_bad) / (x_good - x_bad)
        return float(t)

    def tilt_view_weight(self, tilt_deg: float) -> float:
        # tilt 0 좋음, 90 나쁨
        w = self.smooth01_small_is_good(float(tilt_deg), self.tilt_good_deg, self.tilt_bad_deg)
        return float(max(self.view_min, min(1.0, w)))

    # ============================================================
    # Detection / estimation
    # ============================================================
    def estimate_from_frame(self, frame, cam: CamCfg) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
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
            ground_cm = ground_m * 100.0 * cam.dist_gain  # fixed

            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
            ray_deg = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cam.pos_px + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            rmat, _ = cv2.Rodrigues(rvec)

            # tilt
            nz = float(abs(rmat[2, 2]))
            nz = clamp(nz, 0.0, 1.0)
            tilt_deg = float(math.degrees(math.acos(nz)))
            w_view = self.tilt_view_weight(tilt_deg)

            # yaw
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

            # quality
            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            z = float(tvec[2])
            z_score = 1.0 if z > 0.05 else 0.0

            s_area = self.smooth01_large_is_good(area, self.area_bad_px2, self.area_good_px2)
            s_err = self.smooth01_small_is_good(reproj_err, self.reproj_good_px, self.reproj_bad_px)
            quality = max(self.min_quality_w, (0.45 * s_err + 0.45 * s_area + 0.10 * z_score))
            quality = float(max(self.min_quality_w, quality * (0.6 + 0.4 * w_view)))

            # base weight
            cx = float(np.mean(c2[:, 0]))
            rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            w_base = float(max(0.05, w_center * w_dist))

            w = float(w_base * quality * w_view)

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
                    "tilt_deg": float(tilt_deg),
                    "view_w": float(w_view),
                },
                "held": False,
            })

        return dets

    @staticmethod
    def _clone_det(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            elif isinstance(v, dict):
                out[k] = dict(v)
            else:
                out[k] = v
        return out

    def _apply_cam_gate_and_hold(self, cam_best: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        any_seen = (cam_best["rear"] is not None) or (cam_best["left"] is not None)
        dets_fuse: List[Dict[str, Any]] = []

        for ck in ("rear", "left"):
            g = self.cam_gate[ck]
            seen = cam_best[ck] is not None

            g.gate_hist.append(1 if seen else 0)
            g.stab_hist.append(1 if seen else 0)

            if sum(g.gate_hist) >= self.gate_H:
                g.active = True
            elif (len(g.gate_hist) == self.gate_W) and (sum(g.gate_hist) == 0):
                g.active = False

            det_rate = float(sum(g.stab_hist) / max(1, len(g.stab_hist)))
            stability = float(np.clip(det_rate, self.stab_min, 1.0))

            det_eff: Optional[Dict[str, Any]] = None
            held = False
            ramp = 0.0

            if seen:
                if not g.prev_seen:
                    g.ramp_seen = 0
                g.prev_seen = True

                det_eff = self._clone_det(cam_best[ck])
                g.last_det = self._clone_det(cam_best[ck])
                g.last_seen_frame = self.frame_idx
                held = False

                g.ramp_seen = min(self.reentry_ramp_len, g.ramp_seen + 1)
                ramp = float(g.ramp_seen) / float(self.reentry_ramp_len)
            else:
                g.prev_seen = False
                if not any_seen:
                    det_eff = None
                else:
                    age = self.frame_idx - g.last_seen_frame
                    if g.active and (g.last_det is not None) and (age <= self.hold_max_age):
                        det_eff = self._clone_det(g.last_det)
                        det_eff["weight"] = float(det_eff["weight"]) * (self.hold_decay ** max(0, age))
                        det_eff["held"] = True
                        held = True
                    else:
                        det_eff = None

            if det_eff is None:
                self.latest_cam_obs[ck] = {
                    "seen": bool(seen),
                    "active": bool(g.active),
                    "held": False,
                    "ground_m": None,
                    "quality": None,
                    "stability": stability,
                    "tilt_deg": None,
                }
                continue

            q = float(det_eff.get("dbg_quality", {}).get("quality", 0.5))
            gm = float(det_eff.get("dbg_quality", {}).get("ground_m", 0.0))
            tilt_deg = det_eff.get("dbg_quality", {}).get("tilt_deg", None)

            self.latest_cam_obs[ck] = {
                "seen": bool(seen),
                "active": bool(g.active),
                "held": bool(held),
                "ground_m": gm,
                "quality": q,
                "stability": stability,
                "tilt_deg": tilt_deg,
            }

            det_eff["weight"] = float(det_eff["weight"]) * stability

            if seen:
                det_eff["weight"] = float(det_eff["weight"]) * float(max(0.2, min(1.0, ramp)))

            # consistency soft-gate
            if self.raw_center is not None:
                diff = float(np.linalg.norm(det_eff["center_pos"] - self.raw_center))
                sigma_px = 120.0 + 80.0 * clamp(gm, 0.0, 5.0)
                w_cons = 1.0 / (1.0 + (diff / sigma_px) ** 2)
                w_cons = float(max(self.consistency_min, min(1.0, w_cons)))
                det_eff["weight"] = float(det_eff["weight"]) * w_cons
                det_eff.setdefault("dbg_quality", {})
                det_eff["dbg_quality"]["consistency_w"] = w_cons
                det_eff["dbg_quality"]["diff_px"] = diff

            dets_fuse.append(det_eff)

        return dets_fuse

    def fuse(self, dets: List[Dict[str, Any]]):
        if not dets:
            return None
        total_w = sum(float(d["weight"]) for d in dets)
        if total_w <= 1e-9:
            return None

        center = sum(d["center_pos"] * float(d["weight"]) for d in dets) / total_w
        s = sum(math.sin(float(d["heading"])) * float(d["weight"]) for d in dets) / total_w
        c = sum(math.cos(float(d["heading"])) * float(d["weight"]) for d in dets) / total_w
        heading = math.atan2(s, c)

        sd = 0.0
        sw = 0.0
        for dct in dets:
            gm = float(dct["dbg_quality"]["ground_m"])
            w = float(dct["weight"])
            sd += gm * w
            sw += w
        d_fused = (sd / sw) if sw > 1e-9 else None

        return center, heading, d_fused

    # ============================================================
    # Calibration application
    # ============================================================
    def corrected_pose(self):
        if self.raw_center is None:
            return None
        d = self.latest_fused_d if self.latest_fused_d is not None else 0.0
        theta = self.raw_heading
        c = self.calib.transform_pos(self.raw_center, d, theta)
        h = self.calib.transform_yaw(self.raw_heading, d, theta)
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

    def draw_wheelchair(self, img, center, heading, color, label):
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

    def draw_hud(self, img, dets_fuse: List[Dict[str, Any]]):
        y = 20
        cv2.putText(img, f"ACTIVE calib_v{self.active_version}.json | RAW raw_v{self.active_version}.json created={self.raw_file_created} samples={len(self.raw_samples)}",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 22

        r = self.latest_cam_obs["rear"]
        l = self.latest_cam_obs["left"]
        fd = self.latest_fused_d
        cv2.putText(img, f"fused_d={None if fd is None else f'{fd:.2f}m'} | rear seen={r['seen']} held={r['held']} stab={r['stability']:.2f} tilt={r['tilt_deg']} | left seen={l['seen']} held={l['held']} stab={l['stability']:.2f} tilt={l['tilt_deg']}",
                    (10, y), 0, 0.45, (200, 200, 200), 2, cv2.LINE_AA); y += 20

        if self.lost_count > 0:
            cv2.putText(img, f"STATE: FREEZE (no detections this frame) lost={self.lost_count}",
                        (10, y), 0, 0.6, (0, 140, 255), 2, cv2.LINE_AA); y += 22

        if self.click_mode == "record_center":
            cv2.putText(img, "MODE: center -> click TRUE center point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "record_yaw_center":
            cv2.putText(img, "MODE: yaw -> click CENTER point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24
        elif self.click_mode == "record_yaw_dir":
            cv2.putText(img, "MODE: yaw -> click DIRECTION point", (10, y), 0, 0.65, (0, 255, 255), 2, cv2.LINE_AA); y += 24

        if dets_fuse:
            tags = []
            for d in dets_fuse:
                t = f"{d['cam_key']}{'(H)' if d.get('held', False) else ''}"
                tags.append(t)
            cv2.putText(img, f"fuse uses: {', '.join(tags)}", (10, y), 0, 0.5, (180, 180, 180), 2, cv2.LINE_AA)

    # ============================================================
    # Mouse callback (record only; no corr 저장)
    # ============================================================
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.lost_count > 0:
            print("[CLICK] ignored: currently FREEZE (no fresh detection)")
            self.click_mode = None
            self.tmp_yaw_center = None
            return

        if not self.is_initialized or self.latest_raw_center is None:
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
        theta = float(self.latest_raw_heading)

        if self.click_mode == "record_center":
            self._ensure_raw_file()
            sample = {
                "type": "center",
                "t": now,
                "clicked_xy": [float(pt[0]), float(pt[1])],
                "raw_center": [float(self.latest_raw_center[0]), float(self.latest_raw_center[1])],
                "raw_heading": float(theta),
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

            self._ensure_raw_file()
            sample = {
                "type": "yaw",
                "t": now,
                "clicked_heading": float(clicked_heading),
                "raw_heading": float(theta),
                "raw_center": [float(self.latest_raw_center[0]), float(self.latest_raw_center[1])],
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
    # Fitting (s key)
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

        tr = self._safe_float(rear.get("tilt_deg"), None)
        tl = self._safe_float(left.get("tilt_deg"), None)

        seen_r = bool(rear.get("seen", False))
        seen_l = bool(left.get("seen", False))

        # build d if missing
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

        qs = []
        if qr is not None: qs.append(qr)
        if ql is not None: qs.append(ql)
        q = float(np.clip(float(np.mean(qs)) if qs else 0.5, 0.05, 1.0))

        # tilt(view) factor
        vs = []
        if tr is not None: vs.append(self.tilt_view_weight(tr))
        if tl is not None: vs.append(self.tilt_view_weight(tl))
        v = float(np.mean(vs)) if vs else 1.0

        cams_seen = int(seen_r) + int(seen_l)
        cam_factor = 1.0 if cams_seen == 2 else (0.7 if cams_seen == 1 else 0.2)

        return d, q * v * cam_factor, cams_seen

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

    def _phi_row(self, d: float, theta: float) -> List[float]:
        d = float(d)
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        return [1.0, d, c, s, d * c, d * s]

    def _fit_pos_delta(self, center_samples: list, prior: CalibModel):
        X_list, Ydx, Ydy, W = [], [], [], []
        for s in center_samples:
            clicked = s.get("clicked_xy", None)
            rawc = s.get("raw_center", None)
            theta = self._safe_float(s.get("raw_heading"), None)
            if clicked is None or rawc is None or theta is None:
                continue

            d, wq, _ = self._sample_quality_and_d(s)
            if d is None:
                continue

            x, y = float(rawc[0]), float(rawc[1])
            x_star, y_star = float(clicked[0]), float(clicked[1])

            X_list.append(self._phi_row(d, theta))
            Ydx.append(x_star - x)
            Ydy.append(y_star - y)
            W.append(max(0.02, float(wq)))

        if len(X_list) < 8:
            return None, {"msg": f"not enough center samples (need>=8, got={len(X_list)})"}

        X = np.array(X_list, dtype=np.float64)  # (N,6)
        yx = np.array(Ydx, dtype=np.float64)
        yy = np.array(Ydy, dtype=np.float64)
        w = np.array(W, dtype=np.float64)

        theta0_x = np.array(prior.bx.tolist(), dtype=np.float64)
        theta0_y = np.array(prior.by.tolist(), dtype=np.float64)

        Wcol = w.reshape(-1, 1)
        XtWX = X.T @ (Wcol * X)
        base = float(np.trace(XtWX) / max(1, X.shape[1]))
        lam = max(1e-6, 1e-3 * base)

        bx = self._wls_ridge_prior(X, yx, w, lam, theta0_x)
        by = self._wls_ridge_prior(X, yy, w, lam, theta0_y)

        # robust reweight (2 rounds), sigma grows with d
        for _ in range(2):
            pred_x = X @ bx
            pred_y = X @ by
            rx = yx - pred_x
            ry = yy - pred_y
            r = np.sqrt(rx * rx + ry * ry)

            d_vals = X[:, 1]
            sigma = 10.0 + 8.0 * np.clip(d_vals, 0.0, 10.0)  # px
            robust = 1.0 / (1.0 + (r / sigma) ** 2)
            w2 = w * robust

            bx = self._wls_ridge_prior(X, yx, w2, lam, bx)
            by = self._wls_ridge_prior(X, yy, w2, lam, by)

        pred_x = X @ bx
        pred_y = X @ by
        rmse = float(np.sqrt(np.mean((yx - pred_x) ** 2 + (yy - pred_y) ** 2)))

        info = {"n": int(len(X_list)), "rmse_pos_px": rmse, "lam": lam}
        return (bx.astype(np.float32), by.astype(np.float32)), info

    def _fit_yaw_delta(self, yaw_samples: list, prior: CalibModel):
        X_list, Y, W = [], [], []
        for s in yaw_samples:
            theta = self._safe_float(s.get("raw_heading"), None)
            th_star = self._safe_float(s.get("clicked_heading"), None)
            if theta is None or th_star is None:
                continue

            d, wq, _ = self._sample_quality_and_d(s)
            if d is None:
                continue

            off = wrap_pi(float(th_star) - float(theta))
            X_list.append(self._phi_row(d, theta))
            Y.append(float(off))
            W.append(max(0.02, float(wq)))

        if len(X_list) < 4:
            return None, {"msg": f"not enough yaw samples (need>=4, got={len(X_list)})"}

        X = np.array(X_list, dtype=np.float64)  # (N,6)
        y = np.array(Y, dtype=np.float64)
        w = np.array(W, dtype=np.float64)

        theta0 = np.array(prior.bt.tolist(), dtype=np.float64)

        Wcol = w.reshape(-1, 1)
        XtWX = X.T @ (Wcol * X)
        base = float(np.trace(XtWX) / max(1, X.shape[1]))
        lam = max(1e-8, 1e-3 * base)

        bt = self._wls_ridge_prior(X, y, w, lam, theta0)

        for _ in range(2):
            pred = X @ bt
            res = np.array([wrap_pi(float(yi - pi)) for yi, pi in zip(y, pred)], dtype=np.float64)
            d_vals = X[:, 1]
            sigma = np.deg2rad(6.0 + 4.0 * np.clip(d_vals, 0.0, 10.0))
            robust = 1.0 / (1.0 + (np.abs(res) / sigma) ** 2)
            w2 = w * robust
            bt = self._wls_ridge_prior(X, y, w2, lam, bt)

        pred = X @ bt
        res = np.array([wrap_pi(float(yi - pi)) for yi, pi in zip(y, pred)], dtype=np.float64)
        rmse_deg = float(np.sqrt(np.mean(res * res)) * (180.0 / math.pi))

        info = {"n": int(len(X_list)), "rmse_yaw_deg": rmse_deg, "lam": lam}
        return bt.astype(np.float32), info

    def build_next_calib_from_raw(self):
        vN = self.active_version
        vNp1 = vN + 1

        if not self.raw_file_created:
            print("[FIT] raw file not created yet. Press 'c' or 'a' to collect first.")
            return False

        centers = [s for s in self.raw_samples if s.get("type") == "center"]
        yaws = [s for s in self.raw_samples if s.get("type") == "yaw"]

        pos_fit, pos_info = self._fit_pos_delta(centers, self.calib)
        yaw_fit, yaw_info = self._fit_yaw_delta(yaws, self.calib)

        if pos_fit is None and yaw_fit is None:
            print("[FIT] nothing to fit (need center>=8 or yaw>=4).")
            if pos_info: print("  center:", pos_info)
            if yaw_info: print("  yaw:", yaw_info)
            return False

        bx = self.calib.bx.copy()
        by = self.calib.by.copy()
        bt = self.calib.bt.copy()

        if pos_fit is not None:
            bx, by = pos_fit
        if yaw_fit is not None:
            bt = yaw_fit

        new_calib = CalibModel(
            version=vNp1,
            bx=bx, by=by, bt=bt,
            created_at=time.time(),
            parent_version=vN,
            made_from_raw_version=vN,
        )

        fit_stats = {"center_fit": pos_info, "yaw_fit": yaw_info}
        params_used = {
            "dist_gain_fixed": 0.90,
            "model": "p_corr=p_raw+Δ(d,θ), θ_corr=θ_raw+Δθ(d,θ)",
            "phi": ["1", "d", "cosθ", "sinθ", "dcosθ", "dsinθ"],
            "robust": "soft IRLS 2 rounds (sigma grows with d)",
            "note": "NO SMOOTHING in runtime pose; only calib fit uses robust",
        }
        self._save_calib(new_calib, fit_stats=fit_stats, params_used=params_used)

        # switch
        self.active_version = vNp1
        self.calib = new_calib

        # 새로운 버전에서는 raw 파일을 아직 만들지 않음
        self.raw_samples = []
        self.raw_file_created = False

        print(f"[SWITCH] ACTIVE calib_v{self.active_version}.json")
        print(f"         Next raw will be raw_v{self.active_version}.json (created on first c/a)")
        return True

    # ============================================================
    # Main loop
    # ============================================================
    def run(self):
        while True:
            self.frame_idx += 1

            ok0, fr0 = self.cap_rear.read()
            ok1, fr1 = self.cap_left.read()
            if not ok0 or not ok1:
                break

            dets_all: List[Dict[str, Any]] = []
            dets_all += self.estimate_from_frame(fr0, self.cams["rear"])
            dets_all += self.estimate_from_frame(fr1, self.cams["left"])

            cam_best: Dict[str, Optional[Dict[str, Any]]] = {"rear": None, "left": None}
            for d in dets_all:
                ck = d["cam_key"]
                if cam_best[ck] is None or float(d["weight"]) > float(cam_best[ck]["weight"]):
                    cam_best[ck] = d

            dets_fuse = self._apply_cam_gate_and_hold(cam_best)
            fused = self.fuse(dets_fuse)

            if fused is not None:
                self.lost_count = 0
                center_meas, heading_meas, d_fused = fused

                # NO SMOOTHING: 바로 사용
                self.raw_center = center_meas.astype(np.float32)
                self.raw_heading = float(heading_meas)
                self.is_initialized = True

                # d는 마지막값 유지(없으면 유지)
                if d_fused is not None:
                    self.latest_fused_d = float(d_fused)
            else:
                # FREEZE
                self.lost_count += 1

            # snapshot
            self.latest_raw_center = None if self.raw_center is None else self.raw_center.copy()
            self.latest_raw_heading = float(self.raw_heading)

            corr = self.corrected_pose()
            if corr is not None:
                self.latest_corr_center = corr[0].copy()
                self.latest_corr_heading = float(corr[1])
            else:
                self.latest_corr_center = None
                self.latest_corr_heading = 0.0

            # draw map
            m = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m)

            # all detections markers
            for d in dets_all:
                mp = tuple(d["marker_pos"].astype(int))
                cc = tuple(d["center_pos"].astype(int))
                cv2.circle(m, mp, 4, (0, 255, 255), -1)
                cv2.circle(m, cc, 3, (255, 180, 0), -1)
                tilt = d.get("dbg_quality", {}).get("tilt_deg", None)
                txt = f"ID{d['marker_id']}/{d['cam_key']}"
                if tilt is not None:
                    txt += f" tilt={tilt:.0f}"
                cv2.putText(m, txt, (mp[0] + 6, mp[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            # ✅ RAW(주황 틀) 표시 제거: CORR만 그림
            if corr is not None:
                cpos, chead = corr
                self.draw_wheelchair(m, cpos, chead, color=(0, 255, 0), label=f"CORR v{self.active_version}")

            self.draw_hud(m, dets_fuse)

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
                self.lost_count = 0
                print("[RESET] pose reset (calib/versions unchanged)")

            if key == ord('c'):
                self.click_mode = "record_center"
                self.tmp_yaw_center = None
                print(f"[MODE] record center -> will write raw_v{self.active_version}.json")

            if key == ord('a'):
                self.click_mode = "record_yaw_center"
                self.tmp_yaw_center = None
                print(f"[MODE] record yaw -> will write raw_v{self.active_version}.json (center then direction)")

            if key == ord('s'):
                self.build_next_calib_from_raw()

        self.cap_rear.release()
        self.cap_left.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rear", type=str, default="0", help="rear source: camera index (e.g. 0) or video path")
    ap.add_argument("--left", type=str, default="1", help="left source: camera index (e.g. 1) or video path")
    args = ap.parse_args()

    rear_src = _parse_src(args.rear)
    left_src = _parse_src(args.left)

    WheelchairTracker(rear_src=rear_src, left_src=left_src).run()


if __name__ == "__main__":
    main()
