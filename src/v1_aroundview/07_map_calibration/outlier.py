#!/usr/bin/env python3
# wc_tracker_stable_measure_nocalib_nosmooth.py
#
# ✅ 보정(calib/raw json, c/a/s) 전부 제거
# ✅ smoothing(EMA/버퍼) 없음: 프레임 단위 업데이트
# ✅ 둘 다 못 잡으면 FREEZE(마지막 pose 유지)
#
# ✅ 정지 중 튐 줄이기 3종 세트:
#   (A) ArUco 코너 Subpixel refinement
#   (B) reprojection error 컷(나쁜 프레임 버림)
#   (C) 두 카메라 동시 검출 시 "섞기" 대신 카메라 선택(hysteresis)
#
# ✅ yaw 튐:
#   - flip-flop lock (짧은 시간 내 큰 점프가 좌/우 번갈아 나오면 lock)
#   - lost 후 reacquire 안정하면 즉시 스냅
#
# Keys:
#   q/ESC : quit
#   r     : reset pose

import cv2
import numpy as np
import math
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


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

# --- (A) Subpixel refinement (가능한 경우 켬) ---
# OpenCV 버전에 따라 이름이 다를 수 있어서 hasattr로 안전 처리
if hasattr(aruco_params, "cornerRefinementMethod"):
    # CORNER_REFINE_SUBPIX이 있으면 사용
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # 파라미터들(없으면 무시됨)
    if hasattr(aruco_params, "cornerRefinementWinSize"):
        aruco_params.cornerRefinementWinSize = 5
    if hasattr(aruco_params, "cornerRefinementMaxIterations"):
        aruco_params.cornerRefinementMaxIterations = 30
    if hasattr(aruco_params, "cornerRefinementMinAccuracy"):
        aruco_params.cornerRefinementMinAccuracy = 0.01

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


def circ_mean(angles: List[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    return math.atan2(s, c)


def circ_spread_max(angles: List[float], mean_angle: float) -> float:
    if not angles:
        return 0.0
    return max(abs(wrap_pi(a - mean_angle)) for a in angles)


@dataclass
class CamCfg:
    key: str
    index: Any
    pos_px: np.ndarray
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90  # 고정


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

        self.wc_w_cm, self.wc_l_cm = 55.0, 66.0

        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0
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
        # Tilt weighting
        # =========================
        self.tilt_good_deg = 15.0
        self.tilt_bad_deg = 60.0
        self.view_min = 0.20

        # =========================
        # Quality params
        # =========================
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # =========================
        # (B) reproj 컷 (정지 튐 줄이는 핵심)
        #   - 정지 중 튀는 프레임 대부분 reproj가 순간 튀는 경우가 많음
        # =========================
        self.reproj_cut_px = 7.5     # 이보다 크면 det 버림
        self.area_cut_px2 = 350.0    # 너무 작은 마커(먼 거리/노이즈) 버림

        # =========================
        # (C) 두 카메라 동시 검출 시 카메라 선택 히스테리시스
        # =========================
        self.enable_cam_hysteresis = True
        self.pref_cam: Optional[str] = "rear"    # rear가 기본적으로 낫다면 bias
        self.keep_ratio = 1.15                   # pref가 other보다 이 비율 이상이면 유지
        self.switch_ratio = 1.35                 # other가 pref보다 이 비율 이상이면 스위치
        self.min_score_single = 0.06             # 너무 낮으면 단일 선택하지 않고 둘 다 섞음

        # =========================
        # yaw flip / reacquire
        # =========================
        self.theta_good_map = compass_deg_to_map_rad(0.0)  # 북쪽이 안정적이라는 가정

        self.flip_window = 10
        self.flip_deg = 25.0
        self.flip_needed = 3
        self.yaw_lock_frames = 12
        self._last_large_sign: Optional[int] = None
        self._last_large_frame: int = -10_000
        self._flip_count: int = 0
        self.yaw_unstable_until: int = -1

        self.reacq_confirm_frames = 2
        self.reacq_spread_deg = 12.0
        self.reacq_buf: deque = deque(maxlen=3)
        self.reacq_active: bool = False
        self.reacq_consec_seen: int = 0

        # =========================
        # Camera configs
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
        # Pose state (NO smoothing)
        # =========================
        self.pose_center: Optional[np.ndarray] = None
        self.pose_heading: float = 0.0
        self.is_initialized = False
        self.latest_fused_d: Optional[float] = None
        self.lost_count = 0

        self.latest_cam_obs = {
            "rear": {"seen": False, "active": False, "held": False, "ground_m": None, "quality": None, "stability": None, "tilt_deg": None, "reproj": None, "area": None},
            "left": {"seen": False, "active": False, "held": False, "ground_m": None, "quality": None, "stability": None, "tilt_deg": None, "reproj": None, "area": None},
        }

        # UI
        self.win_map = "minimap"
        self.win_mon = "monitor(rear|left)"
        cv2.namedWindow(self.win_map, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mon, cv2.WINDOW_NORMAL)

        self.frame_idx = 0

        print("[NO CALIB / NO SMOOTH + Stable Measurement]")
        print(" - (A) subpixel corners, (B) reproj cut, (C) cam hysteresis")
        print(" - miss면 FREEZE 유지")
        print("[Keys] q/ESC quit | r reset pose")

    # =========================================================
    # Geometry helpers
    # =========================================================
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
        if x <= x_good:
            return 1.0
        if x >= x_bad:
            return 0.0
        t = (x - x_good) / (x_bad - x_good)
        return float(1.0 - t)

    @staticmethod
    def smooth01_large_is_good(x: float, x_bad: float, x_good: float) -> float:
        if x <= x_bad:
            return 0.0
        if x >= x_good:
            return 1.0
        t = (x - x_bad) / (x_good - x_bad)
        return float(t)

    def tilt_view_weight(self, tilt_deg: float) -> float:
        w = self.smooth01_small_is_good(float(tilt_deg), self.tilt_good_deg, self.tilt_bad_deg)
        return float(max(self.view_min, min(1.0, w)))

    # =========================================================
    # yaw flip / reacquire helpers
    # =========================================================
    def _update_flipflop_detector(self, dy: float) -> None:
        flip_rad = math.radians(self.flip_deg)
        if abs(dy) <= flip_rad:
            self._flip_count = max(0, self._flip_count - 1)
            return

        sign = 1 if dy > 0 else -1
        if (self.frame_idx - self._last_large_frame) > self.flip_window:
            self._flip_count = 0
            self._last_large_sign = None

        if self._last_large_sign is not None and sign != self._last_large_sign:
            self._flip_count += 1
        else:
            self._flip_count = max(0, self._flip_count - 1)

        self._last_large_sign = sign
        self._last_large_frame = self.frame_idx

        if self._flip_count >= self.flip_needed:
            self.yaw_unstable_until = self.frame_idx + self.yaw_lock_frames
            self._flip_count = 0
            self._last_large_sign = None

    def _yaw_locked(self) -> bool:
        return self.frame_idx <= self.yaw_unstable_until

    def _reacq_update_and_maybe_snap(self, yaw_meas: float) -> Optional[float]:
        self.reacq_buf.append(float(yaw_meas))
        self.reacq_consec_seen += 1
        if self.reacq_consec_seen < self.reacq_confirm_frames:
            return None

        angles = list(self.reacq_buf)[-self.reacq_confirm_frames:]
        mu = circ_mean(angles)
        spread = circ_spread_max(angles, mu)
        if spread <= math.radians(self.reacq_spread_deg):
            self.reacq_active = False
            self.reacq_consec_seen = 0
            self.reacq_buf.clear()
            return float(mu)
        return None

    # =========================================================
    # Detection / estimation
    # =========================================================
    def estimate_from_frame(self, frame, cam: CamCfg) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return dets

        # (A) 수동 cornerSubPix를 한 번 더(버전에 따라 detector가 안할 때가 있음)
        # -> 이건 "스무딩"이 아니라 "측정값 정밀화"
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            for i in range(len(corners)):
                # corners[i] shape: (1,4,2) float
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)
        except Exception:
            pass

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
            ground_cm = ground_m * 100.0 * cam.dist_gain

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
            yaw_compass = wrap360(yaw_compass + (180.0 if mid == 1 else 0.0))
            heading_map = compass_deg_to_map_rad(yaw_compass)

            center_pos = self.marker_to_center(marker_pos, heading_map, mid)

            # quality metrics
            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            # (B) 하드 컷: reproj/area가 나쁘면 "그 프레임 측정은 버림"
            if (reproj_err > self.reproj_cut_px) or (area < self.area_cut_px2):
                continue

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
                "dbg": {
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

    # =========================================================
    # Gate/Hold/Stability/Reentry/Consistency + FREEZE on both-miss
    # =========================================================
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

                g.ramp_seen = min(self.reentry_ramp_len, g.ramp_seen + 1)
                ramp = float(g.ramp_seen) / float(self.reentry_ramp_len)
            else:
                g.prev_seen = False
                if not any_seen:
                    det_eff = None  # 둘 다 miss면 hold도 안씀 -> FREEZE
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
                    "reproj": None,
                    "area": None,
                }
                continue

            gm = float(det_eff["dbg"]["ground_m"])
            q = float(det_eff["dbg"]["quality"])
            tilt_deg = float(det_eff["dbg"]["tilt_deg"])
            reproj = float(det_eff["dbg"]["reproj"])
            area = float(det_eff["dbg"]["area"])

            self.latest_cam_obs[ck] = {
                "seen": bool(seen),
                "active": bool(g.active),
                "held": bool(held),
                "ground_m": gm,
                "quality": q,
                "stability": stability,
                "tilt_deg": tilt_deg,
                "reproj": reproj,
                "area": area,
            }

            # stability / ramp
            det_eff["weight"] = float(det_eff["weight"]) * stability
            if seen:
                det_eff["weight"] = float(det_eff["weight"]) * float(max(0.2, min(1.0, ramp)))

            # consistency soft gate (현재 pose와 너무 멀면 감쇠)
            if self.pose_center is not None:
                diff = float(np.linalg.norm(det_eff["center_pos"] - self.pose_center))
                sigma_px = 120.0 + 80.0 * clamp(gm, 0.0, 5.0)
                w_cons = 1.0 / (1.0 + (diff / sigma_px) ** 2)
                w_cons = float(max(self.consistency_min, min(1.0, w_cons)))
                det_eff["weight"] = float(det_eff["weight"]) * w_cons

            dets_fuse.append(det_eff)

        return dets_fuse

    # =========================================================
    # (C) Cam hysteresis selector
    # =========================================================
    def _select_for_fusion(self, dets_fuse: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enable_cam_hysteresis:
            return dets_fuse
        if len(dets_fuse) < 2:
            return dets_fuse

        by = {d["cam_key"]: d for d in dets_fuse}
        if ("rear" not in by) or ("left" not in by):
            return dets_fuse

        dr = by["rear"]
        dl = by["left"]

        # 둘 다 "실제 seen"일 때만 강하게 단일 선택(held 섞이면 이상해질 수 있음)
        if dr.get("held", False) or dl.get("held", False):
            return dets_fuse

        sr = float(dr["weight"])
        sl = float(dl["weight"])

        # 둘 다 너무 낮으면 굳이 단일 선택 X
        if max(sr, sl) < self.min_score_single:
            return dets_fuse

        # 초기/비슷하면 rear bias
        if self.pref_cam is None:
            self.pref_cam = "rear" if (sr >= 0.95 * sl) else ("rear" if sr > sl else "left")

        pref = self.pref_cam
        other = "left" if pref == "rear" else "rear"
        sp = sr if pref == "rear" else sl
        so = sl if pref == "rear" else sr

        # 유지/스위치 조건
        if sp >= self.keep_ratio * so:
            return [by[pref]]
        if so >= self.switch_ratio * sp:
            self.pref_cam = other
            return [by[other]]

        # 비슷하면 둘 다 섞되, pref를 살짝 더 믿고 싶으면 여기서 weight bias를 줄 수도 있음(지금은 그대로)
        return dets_fuse

    # =========================================================
    # Fuse
    # =========================================================
    def fuse(self, dets: List[Dict[str, Any]]):
        if not dets:
            return None

        # (C) cam 선택 적용
        dets_use = self._select_for_fusion(dets)

        total_w = sum(float(d["weight"]) for d in dets_use)
        if total_w <= 1e-9:
            return None

        center = sum(d["center_pos"] * float(d["weight"]) for d in dets_use) / total_w

        # heading은 held 제외 우선
        dets_h = [d for d in dets_use if not d.get("held", False)]
        if not dets_h:
            dets_h = dets_use
        tw_h = sum(float(d["weight"]) for d in dets_h)
        if tw_h <= 1e-9:
            return None

        s = sum(math.sin(float(d["heading"])) * float(d["weight"]) for d in dets_h) / tw_h
        c = sum(math.cos(float(d["heading"])) * float(d["weight"]) for d in dets_h) / tw_h
        heading = math.atan2(s, c)

        sd, sw = 0.0, 0.0
        for dct in dets_h:
            gm = float(dct["dbg"]["ground_m"])
            w = float(dct["weight"])
            sd += gm * w
            sw += w
        d_fused = (sd / sw) if sw > 1e-9 else None

        return center, heading, d_fused, dets_use

    # =========================================================
    # UI
    # =========================================================
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

    def draw_wheelchair(self, img, center, heading, color=(0, 255, 0), label="POSE"):
        w_px = (self.wc_w_cm * self.map_scale) / 2.0
        l_px = (self.wc_l_cm * self.map_scale) / 2.0

        base = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]], dtype=np.float32)
        rot = np.array([[math.cos(heading), -math.sin(heading)],
                        [math.sin(heading),  math.cos(heading)]], dtype=np.float32)
        pts = (base @ rot.T) + center

        cv2.polylines(img, [pts.astype(np.int32)], True, color, 2, cv2.LINE_AA)
        arrow_len_px = 55.0 * self.map_scale
        cv2.arrowedLine(img, tuple(center.astype(int)),
                        (int(center[0] + arrow_len_px * math.cos(heading)),
                         int(center[1] + arrow_len_px * math.sin(heading))),
                        color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (int(center[0]) + 10, int(center[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def draw_hud(self, img, dets_use: List[Dict[str, Any]]):
        y = 20
        r = self.latest_cam_obs["rear"]
        l = self.latest_cam_obs["left"]
        fd = self.latest_fused_d

        flags = []
        if self.lost_count > 0:
            flags.append(f"FREEZE({self.lost_count})")
        if self._yaw_locked():
            flags.append("YAW_LOCK")
        if self.reacq_active:
            flags.append("REACQ_WAIT")
        if self.enable_cam_hysteresis:
            flags.append(f"PREF={self.pref_cam}")

        cv2.putText(img, f"StableMeasure | flags={','.join(flags) if flags else 'none'}",
                    (10, y), 0, 0.55, (220, 220, 220), 2, cv2.LINE_AA); y += 22

        cv2.putText(img,
                    f"fused_d={None if fd is None else f'{fd:.2f}m'} | "
                    f"rear seen={r['seen']} held={r['held']} reproj={r['reproj']} area={r['area']} | "
                    f"left seen={l['seen']} held={l['held']} reproj={l['reproj']} area={l['area']}",
                    (10, y), 0, 0.42, (200, 200, 200), 2, cv2.LINE_AA); y += 20

        if dets_use:
            tags = []
            for d in dets_use:
                tags.append(f"{d['cam_key']}{'(H)' if d.get('held', False) else ''}")
            cv2.putText(img, f"use: {', '.join(tags)}",
                        (10, y), 0, 0.5, (180, 180, 180), 2, cv2.LINE_AA)

    # =========================================================
    # Main loop
    # =========================================================
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

            # best per cam
            cam_best: Dict[str, Optional[Dict[str, Any]]] = {"rear": None, "left": None}
            for d in dets_all:
                ck = d["cam_key"]
                if cam_best[ck] is None or float(d["weight"]) > float(cam_best[ck]["weight"]):
                    cam_best[ck] = d

            dets_fuse = self._apply_cam_gate_and_hold(cam_best)
            fused = self.fuse(dets_fuse)

            if fused is not None:
                center_meas, heading_meas, d_fused, dets_use = fused

                # lost였다가 다시 잡히면 reacquire 모드
                if self.lost_count > 0:
                    self.reacq_active = True
                    self.reacq_consec_seen = 0
                    self.reacq_buf.clear()
                    self.yaw_unstable_until = -1
                    self._flip_count = 0
                    self._last_large_sign = None

                self.lost_count = 0
                if d_fused is not None:
                    self.latest_fused_d = float(d_fused)

                # yaw 처리: reacquire 스냅/락
                yaw_new = float(heading_meas)
                if self.pose_center is None:
                    pass
                else:
                    # reacq면 안정하면 스냅, 아니면 동결
                    if self.reacq_active:
                        snapped = self._reacq_update_and_maybe_snap(yaw_new)
                        if snapped is None:
                            yaw_new = float(self.pose_heading)
                        else:
                            yaw_new = float(snapped)
                    else:
                        dy_ref = wrap_pi(float(yaw_new) - float(self.pose_heading))
                        self._update_flipflop_detector(dy_ref)
                        if self._yaw_locked():
                            yaw_new = float(self.pose_heading)

                # position은 reproj 컷 + cam hyst로 이미 많이 안정화됨 -> 그대로 사용
                self.pose_center = center_meas.astype(np.float32)
                self.pose_heading = float(yaw_new)
                self.is_initialized = True

            else:
                # miss -> FREEZE 유지
                self.lost_count += 1
                self.reacq_consec_seen = 0
                self.reacq_buf.clear()

            # draw
            m = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m)

            # det 점 표시(참고용) — 박스(주황) 같은 "원본 표시"는 따로 안 그려줌
            for d in dets_all:
                mp = tuple(d["marker_pos"].astype(int))
                cc = tuple(d["center_pos"].astype(int))
                cv2.circle(m, mp, 4, (0, 255, 255), -1)
                cv2.circle(m, cc, 3, (255, 180, 0), -1)

            if self.pose_center is not None:
                self.draw_wheelchair(m, self.pose_center, self.pose_heading, color=(0, 255, 0), label="POSE")

            # HUD는 fuse에서 실제 사용한 cam만 보여주고 싶어서 dets_use를 만들자
            # fused가 None이면 빈 리스트
            dets_use = []
            if dets_fuse:
                # fuse 내부에서 선택이 적용되지만 fused=None이면 여기 못옴
                pass

            # 간단히: 현재 프레임 dets_fuse를 다시 selector로 통과시켜 표시
            dets_use = self._select_for_fusion(dets_fuse) if dets_fuse else []
            self.draw_hud(m, dets_use)

            mon0 = cv2.resize(fr0, (640, 360))
            mon1 = cv2.resize(fr1, (640, 360))
            cv2.imshow(self.win_mon, np.hstack([mon0, mon1]))
            cv2.imshow(self.win_map, m)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                self.is_initialized = False
                self.pose_center = None
                self.pose_heading = 0.0
                self.latest_fused_d = None
                self.lost_count = 0

                self.reacq_active = False
                self.reacq_buf.clear()
                self.reacq_consec_seen = 0

                self.yaw_unstable_until = -1
                self._flip_count = 0
                self._last_large_sign = None
                print("[RESET] pose reset")

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
