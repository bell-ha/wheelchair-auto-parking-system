import cv2
import numpy as np
import math
import json
import os


def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg


def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(아래)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)


class PoseEstimator:
    """ArUco 마커 기반 위치 및 방향 추정 모듈 (map_calibration 반영)"""

    def __init__(self, K, D, cams, marker_size_m, marker_h_cm, dist_gain, alpha, calib_path=None):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size_m = marker_size_m
        self.marker_h_cm_default = marker_h_cm
        self.dist_gain = dist_gain
        self.alpha = alpha

        self.marker_h_cm_by_id = {0: float(marker_h_cm), 1: float(marker_h_cm)}
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        # calibration
        self.calib_path = calib_path or os.path.join(os.path.dirname(__file__), "wc_calibration.json")
        self.calib_dxdy = np.array([0.0, 0.0], dtype=np.float32)
        self.calib_yaw_offset = 0.0
        self.dist_bins_m = [0.0, 2.0, 4.0, 999.0]
        self.dist_blend_m = 0.4
        self.calib_dxdy_bins = {
            "near": np.array([0.0, 0.0], dtype=np.float32),
            "mid": np.array([0.0, 0.0], dtype=np.float32),
            "far": np.array([0.0, 0.0], dtype=np.float32),
        }
        self._load_calibration()

        # quality params
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # state
        self.raw_center = None
        self.raw_heading = 0.0
        self.marker_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False
        self.last_best_ground_m = None
        self.last_det_by_cam = {}
        self.last_det_all = []

    def _load_calibration(self):
        if not os.path.exists(self.calib_path):
            return
        try:
            with open(self.calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            dx = float(data.get("calib_dx", 0.0))
            dy = float(data.get("calib_dy", 0.0))
            self.calib_dxdy = np.array([dx, dy], dtype=np.float32)
            self.calib_yaw_offset = float(data.get("calib_yaw_offset", 0.0))

            bins = data.get("calib_dxdy_bins", None)
            if isinstance(bins, dict):
                for k in ("near", "mid", "far"):
                    if k in bins and isinstance(bins[k], (list, tuple)) and len(bins[k]) == 2:
                        self.calib_dxdy_bins[k] = np.array([
                            float(bins[k][0]),
                            float(bins[k][1])
                        ], dtype=np.float32)

            db = data.get("dist_bins_m", None)
            if isinstance(db, list) and len(db) >= 4:
                self.dist_bins_m = [float(db[0]), float(db[1]), float(db[2]), float(db[3])]
        except Exception:
            pass

    def _bin_name_for_dist(self, r_m: float) -> str:
        if r_m < self.dist_bins_m[1]:
            return "near"
        if r_m < self.dist_bins_m[2]:
            return "mid"
        return "far"

    def _dxdy_for_dist(self, r_m: float) -> np.ndarray:
        near_end = float(self.dist_bins_m[1])
        mid_end = float(self.dist_bins_m[2])
        b = float(self.dist_blend_m)

        if r_m < near_end - b:
            return self.calib_dxdy_bins["near"]
        if r_m > mid_end + b:
            return self.calib_dxdy_bins["far"]
        if (near_end + b) <= r_m <= (mid_end - b):
            return self.calib_dxdy_bins["mid"]

        if (near_end - b) <= r_m <= (near_end + b):
            t = (r_m - (near_end - b)) / (2.0 * b)
            return (1 - t) * self.calib_dxdy_bins["near"] + t * self.calib_dxdy_bins["mid"]

        if (mid_end - b) <= r_m <= (mid_end + b):
            t = (r_m - (mid_end - b)) / (2.0 * b)
            return (1 - t) * self.calib_dxdy_bins["mid"] + t * self.calib_dxdy_bins["far"]

        return self.calib_dxdy_bins[self._bin_name_for_dist(r_m)]

    @staticmethod
    def _smooth01(x, x0, x1):
        if x <= x0:
            return 1.0
        if x >= x1:
            return 0.0
        t = (x - x0) / (x1 - x0)
        return float(1.0 - t)

    def _marker_to_center(self, marker_pos_px: np.ndarray, heading_map_rad: float, marker_id: int) -> np.ndarray:
        offset_cm = float(self.center_offset_cm_by_id.get(marker_id, 23.0))
        offset_px = offset_cm
        dx = offset_px * math.cos(heading_map_rad)
        dy = offset_px * math.sin(heading_map_rad)
        if marker_id == 0:
            return marker_pos_px - np.array([dx, dy], dtype=np.float32)
        return marker_pos_px + np.array([dx, dy], dtype=np.float32)

    def _estimate_from_frame(self, frame, cam_name, cfg):
        dets = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return dets

        for i, mid_arr in enumerate(ids):
            mid = int(mid_arr[0])
            if mid not in (0, 1):
                continue

            c2 = corners[i].reshape(4, 2).astype(np.float32)

            und = cv2.fisheye.undistortPoints(
                corners[i].reshape(-1, 1, 2),
                self.K, self.D, P=self.K
            )

            ms = float(self.marker_size_m)
            obj_points = np.array([
                [-ms / 2,  ms / 2, 0],
                [ ms / 2,  ms / 2, 0],
                [ ms / 2, -ms / 2, 0],
                [-ms / 2, -ms / 2, 0]
            ], dtype=np.float32)

            ok, rvec, tvec = cv2.solvePnP(obj_points, und, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            dist_m = float(np.linalg.norm(tvec))
            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(float(cfg["h_cm"]) - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            dist_gain = float(cfg.get("dist_gain", self.dist_gain))
            ground_cm = ground_m * 100.0 * dist_gain

            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
            ray_deg = float(cfg["map_angle_deg"]) + float(cfg.get("yaw_trim_deg", 0.0)) + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cfg["pos_px"] + np.array([
                ground_cm * float(cfg.get("map_scale", 1.0)) * math.cos(ray_rad),
                ground_cm * float(cfg.get("map_scale", 1.0)) * math.sin(ray_rad)
            ], dtype=np.float32)

            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * float(cfg.get("sens", 1.0))) + float(cfg.get("install_angle", 0.0))
            yaw_compass = total - float(cfg.get("install_offset", 0.0))
            if mid == 1:
                yaw_compass = wrap360(yaw_compass + 180.0)
            else:
                yaw_compass = wrap360(yaw_compass)

            heading_map = compass_deg_to_map_rad(yaw_compass)
            center_pos = self._marker_to_center(marker_pos, heading_map, mid)

            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(obj_points, rvec, tvec, self.K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(proj - und.reshape(-1, 2).astype(np.float32), axis=1)))

            z = float(tvec[2])
            z_score = 1.0 if z > 0.05 else 0.0
            s_area = self._smooth01(area, self.area_good_px2, self.area_bad_px2)
            s_err = self._smooth01(reproj_err, self.reproj_good_px, self.reproj_bad_px)
            quality = max(self.min_quality_w, (0.45 * s_err + 0.45 * s_area + 0.10 * z_score))

            cx = float(np.mean(c2[:, 0]))
            rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            w = float(max(0.05, w_center * w_dist) * quality)

            dets.append({
                "marker_id": mid,
                "marker_pos": marker_pos,
                "center_pos": center_pos,
                "heading": heading_map,
                "weight": w,
                "cam_key": cam_name,
                "ground_m": float(ground_m)
            })

        return dets

    def detect_and_estimate(self, frames):
        """프레임에서 마커를 감지하고 위치/방향 추정"""
        self.last_det_by_cam = {}
        self.last_det_all = []

        dets = []
        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue
            cfg = self.cams[cam_name]
            cam_dets = self._estimate_from_frame(frame, cam_name, cfg)
            if cam_dets:
                dets.extend(cam_dets)
            self.last_det_by_cam[cam_name] = cam_dets

        self.last_det_all = dets

        if dets:
            best = max(dets, key=lambda d: d["weight"])
            self.last_best_ground_m = float(best.get("ground_m", 0.0))

        if dets:
            total_w = sum(d["weight"] for d in dets)
            center = sum(d["center_pos"] * d["weight"] for d in dets) / total_w
            s = sum(math.sin(d["heading"]) * d["weight"] for d in dets) / total_w
            c = sum(math.cos(d["heading"]) * d["weight"] for d in dets) / total_w
            heading = math.atan2(s, c)

            if not self.is_initialized:
                self.raw_center = center
                self.raw_heading = heading
                self.is_initialized = True
            else:
                self.raw_center = self.raw_center * (1 - self.alpha) + center * self.alpha
                diff = wrap_pi(heading - self.raw_heading)
                self.raw_heading = wrap_pi(self.raw_heading + diff * self.alpha)

            r = float(self.last_best_ground_m) if self.last_best_ground_m is not None else 0.0
            dxdy = self._dxdy_for_dist(r)
            self.marker_pos = self.raw_center + dxdy
            self.heading_angle = wrap_pi(self.raw_heading + self.calib_yaw_offset)

        return self.marker_pos, self.heading_angle, self.is_initialized

    def get_center_position(self, wc_l, map_scale):
        """휠체어 중심 위치 계산 (보정된 center 반환)"""
        return self.marker_pos