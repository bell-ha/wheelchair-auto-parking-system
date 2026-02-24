#!/usr/bin/env python3
# make_tocsv_south0_with_view_angles.py
#
# 목적
# - snapshots_rear_left 폴더의 이미지들을 읽어서(파일명에 GT 포함)
# - ArUco/PnP 기반으로 좌표/각도(남쪽=0도 기준)를 추정하고
# - analyze/tocsv.csv "한 파일"로 저장한다.
#
# 파일명 형식 (예)
#   left_x100_y360_+45_3.jpg
#   rear_x-10_y800_-45_12.png
#
# =========================
# 출력 CSV 컬럼 설명
# =========================
# cam                  : "left" 또는 "rear" (어떤 카메라 이미지인지)
# gt_x, gt_y           : 정답 좌표 (맵 좌표, 너가 파일명에 적은 값 그대로)
# gt_ang               : 정답 방향 (남쪽=0 기준, deg; 파일명 각도)
# seq                  : 같은 위치/각도에서 찍은 번호(파일명 마지막 숫자)
# status               : ok / no_marker / read_fail
#
# pred_x, pred_y        : 예측 좌표 (맵 좌표, cm==px 가정)
# pred_ang              : 예측 방향 (남쪽=0 기준, deg)
#
# marker_id             : 사용된 마커 ID (0 또는 1)
# num_dets              : 해당 이미지에서 검출된 후보 마커 개수(0이면 못잡음)
#
# area                  : 마커 사각형 면적(px^2) - 클수록 보통 더 잘 잡힘
# reproj                : 재투영 오차(px) - 작을수록 PnP가 안정적
# ground_m              : 카메라-마커 바닥 거리(m) - 멀수록 흔들리기 쉬움
# bearing_deg           : 카메라 기준 좌/우 방향각(deg) = atan2(tvec.x, tvec.z) // 양수=오른쪽, 음수=왼쪽, 중앙=0
#
# ray_ang               : "카메라->마커" 직선 방향(남쪽=0 기준 deg)  카메라가 마커를 바라볼 때 지도 위에서 어느 방향을 향하는지
# view_rel_deg          : 상대 관측각 = wrap(pred_ang - ray_ang)  ([-180,180))
#                         - 0이면 정면, ±180이면 반대정면(뒷마커 정면도 포함)
# view_sym_deg          : 정면성 지표(0~90) = min(|view_rel|, |180-|view_rel||)
#                         - 0이면 (정면 또는 반대정면), 90이면 옆면
#
# quality               : (0~1) 정도의 품질 점수(간단 휴리스틱: reproj/area/z)
# weight                : 최종 가중치(quality * 거리/중앙 가중)
#
# pos_err_cm            : 위치오차(cm) = sqrt((pred_x-gt_x)^2 + (pred_y-gt_y)^2)   (ok일 때만)
# ang_err_deg           : 각도오차(deg) = wrap(pred_ang - gt_ang)                  (ok일 때만)
#
# =========================
# 주의
# - "보정(calibration)"은 전혀 적용하지 않는다. (no_calib)
# - 각도는 전부 "남쪽=0도(south0)" 기준으로 통일한다.
# - 정면(0)뿐 아니라 반대정면(180)도 잘 보일 수 있으니 view_sym_deg를 함께 제공한다.
#
import os, glob, math, re, csv
from dataclasses import dataclass

import cv2
import numpy as np


# =========================
# Intrinsic / Distortion
# =========================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE_M = 0.25
OBJ_POINTS = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0]
], dtype=np.float32)

# =========================
# ArUco
# =========================
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
    # [-180,180)
    return (deg + 180.0) % 360.0 - 180.0

def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(down)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)

def map_deg_to_south0_deg(map_deg: float) -> float:
    # map SOUTH(down)=90 -> south0 SOUTH=0
    return wrap360(map_deg - 90.0)

def south0_deg_to_map_deg(south0_deg: float) -> float:
    # south0 SOUTH=0 -> map SOUTH=90
    return wrap360(south0_deg + 90.0)

def view_sym_from_rel(rel_deg: float) -> float:
    # rel in [-180,180)
    a = abs(rel_deg)
    return float(min(a, abs(180.0 - a)))  # 0~90

def clamp(x, a, b):
    return max(a, min(b, x))

@dataclass
class CamCfg:
    key: str
    pos_px: np.ndarray        # map coords (px==cm)
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90

class EstimatorNoCalibSameCoords:
    def __init__(self):
        self.map_scale = 1.0  # 1px=1cm

        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # quality heuristics
        self.reproj_good_px = 2.0
        self.reproj_bad_px  = 8.0
        self.area_good_px2  = 2500.0
        self.area_bad_px2   = 600.0
        self.min_quality_w  = 0.08

        # ✅ 좌표계는 GT(파일명)와 동일하게 사용
        self.cams = {
            "rear": CamCfg("rear", np.array([301.4, 540.0], np.float32),
                           105.5, 90.0, 1.6, 0.0, 0.0, yaw_trim_deg=3.0, dist_gain=0.90),
            "left": CamCfg("left", np.array([200.0, 270.0], np.float32),
                           110.0, 157.0, 1.6, 113.0, 50.84, yaw_trim_deg=8.0, dist_gain=0.90),
        }

    @staticmethod
    def smooth01(x, x0, x1):
        if x <= x0: return 1.0
        if x >= x1: return 0.0
        t = (x - x0) / (x1 - x0)
        return float(1.0 - t)

    def marker_to_center(self, marker_pos_px, heading_map_rad, marker_id):
        offset_cm = float(self.center_offset_cm_by_id.get(marker_id, 23.0))
        dx = offset_cm * math.cos(heading_map_rad)
        dy = offset_cm * math.sin(heading_map_rad)
        if marker_id == 0:
            return marker_pos_px - np.array([dx, dy], dtype=np.float32)
        else:
            return marker_pos_px + np.array([dx, dy], dtype=np.float32)

    def estimate_from_image(self, img_bgr, cam_key: str):
        """
        returns:
          best (dict) or None
          all_dets (list[dict])
        best includes: pred_x,pred_y,pred_ang, area,reproj,ground_m,bearing_deg,ray_ang,view_rel_deg,view_sym_deg,quality,weight,marker_id
        """
        cam = self.cams[cam_key]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return None, []

        dets = []
        for i, mid_arr in enumerate(ids):
            mid = int(mid_arr[0])
            if mid not in (0, 1):
                continue

            c2 = corners[i].reshape(4, 2).astype(np.float32)
            und = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)

            ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            if float(tvec[2]) <= 0.01:
                continue

            dist_m = float(np.linalg.norm(tvec))

            # ground distance
            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * cam.dist_gain

            # bearing (camera 좌/우)
            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))

            # ray direction on map
            ray_deg_map = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad_map = math.radians(ray_deg_map)

            # marker map position
            marker_pos = cam.pos_px + np.array([
                ground_cm * math.cos(ray_rad_map),
                ground_cm * math.sin(ray_rad_map)
            ], dtype=np.float32)

            # yaw -> heading (south0)
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset
            yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)

            heading_map_rad = compass_deg_to_map_rad(yaw_compass)
            heading_map_deg = wrap360(math.degrees(heading_map_rad))
            pred_ang_south0 = map_deg_to_south0_deg(heading_map_deg)

            # center position
            center_pos = self.marker_to_center(marker_pos, heading_map_rad, mid)

            # quality metrics
            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(
                proj - und.reshape(-1, 2).astype(np.float32), axis=1
            )))

            z = float(tvec[2])
            z_score = 1.0 if z > 0.05 else 0.0
            s_area = self.smooth01(area, self.area_good_px2, self.area_bad_px2)
            s_err  = self.smooth01(reproj_err, self.reproj_good_px, self.reproj_bad_px)
            quality = max(self.min_quality_w, (0.45 * s_err + 0.45 * s_area + 0.10 * z_score))

            # weight (거리 멀수록↓ + 화면 중앙에 가까울수록↑ + quality)
            cx = float(np.mean(c2[:, 0]))
            rel_x = (cx - img_bgr.shape[1] / 2) / (img_bgr.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            w_base = float(max(0.05, w_center * w_dist))
            weight = float(w_base * quality)

            # ray angle in south0 too
            ray_ang_south0 = map_deg_to_south0_deg(wrap360(ray_deg_map))
            view_rel_deg = wrap180(pred_ang_south0 - ray_ang_south0)   # [-180,180)
            view_sym_deg = view_sym_from_rel(view_rel_deg)             # [0,90]

            dets.append({
                "pred_x": float(center_pos[0]),
                "pred_y": float(center_pos[1]),
                "pred_ang": float(pred_ang_south0),
                "marker_id": mid,
                "area": area,
                "reproj": reproj_err,
                "ground_m": float(ground_m),
                "bearing_deg": float(bearing_deg),
                "ray_ang": float(ray_ang_south0),
                "view_rel_deg": float(view_rel_deg),
                "view_sym_deg": float(view_sym_deg),
                "quality": float(quality),
                "weight": float(weight),
            })

        if not dets:
            return None, dets

        best = max(dets, key=lambda d: d["weight"])
        return best, dets


# =========================
# Filename parsing
# =========================
PAT = re.compile(
    r"^(left|rear)_x([+-]?\d+)_y([+-]?\d+)_([+-]?\d+)_([0-9]+)\.(jpg|jpeg|png)$",
    re.IGNORECASE
)

def parse_meta(path: str):
    b = os.path.basename(path)
    m = PAT.match(b)
    if not m:
        return None
    return {
        "cam": m.group(1).lower(),
        "gt_x": int(m.group(2)),
        "gt_y": int(m.group(3)),
        "gt_ang": int(m.group(4)),     # south0 deg
        "seq": int(m.group(5)),
        "file": b,
    }


def main():
    # input folder
    folder = "snapshots_rear_left"
    if not os.path.isdir(folder) and os.path.isdir("/snapshots_rear_left"):
        folder = "/snapshots_rear_left"
    if not os.path.isdir(folder):
        print("[ERR] folder not found:", folder)
        return

    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg")) +
        glob.glob(os.path.join(folder, "*.png"))
    )
    if not paths:
        print("[ERR] no images in:", folder)
        return

    # output (single CSV)
    analyze_dir = os.path.join(os.getcwd(), "analyze")
    os.makedirs(analyze_dir, exist_ok=True)
    out_csv = os.path.join(analyze_dir, "tocsv.csv")

    est = EstimatorNoCalibSameCoords()

    fieldnames = [
        "cam", "gt_x", "gt_y", "gt_ang", "seq", "status",
        "pred_x", "pred_y", "pred_ang",
        "marker_id", "num_dets",
        "area", "reproj", "ground_m", "bearing_deg",
        "ray_ang", "view_rel_deg", "view_sym_deg",
        "quality", "weight",
        "pos_err_cm", "ang_err_deg",
        "file"
    ]

    rows = []
    for p in paths:
        meta = parse_meta(p)
        if meta is None:
            continue

        img = cv2.imread(p)
        if img is None:
            rows.append({
                "cam": meta["cam"], "gt_x": meta["gt_x"], "gt_y": meta["gt_y"], "gt_ang": meta["gt_ang"],
                "seq": meta["seq"], "status": "read_fail", "file": meta["file"]
            })
            continue

        best, all_dets = est.estimate_from_image(img, meta["cam"])

        if best is None:
            rows.append({
                "cam": meta["cam"], "gt_x": meta["gt_x"], "gt_y": meta["gt_y"], "gt_ang": meta["gt_ang"],
                "seq": meta["seq"], "status": "no_marker", "num_dets": len(all_dets), "file": meta["file"]
            })
            continue

        # errors (analysis columns)
        pos_err = math.hypot(best["pred_x"] - meta["gt_x"], best["pred_y"] - meta["gt_y"])
        ang_err = wrap180(best["pred_ang"] - float(meta["gt_ang"]))

        rows.append({
            "cam": meta["cam"],
            "gt_x": meta["gt_x"], "gt_y": meta["gt_y"], "gt_ang": meta["gt_ang"],
            "seq": meta["seq"],
            "status": "ok",
            "pred_x": f"{best['pred_x']:.3f}",
            "pred_y": f"{best['pred_y']:.3f}",
            "pred_ang": f"{best['pred_ang']:.3f}",
            "marker_id": best["marker_id"],
            "num_dets": len(all_dets),
            "area": f"{best['area']:.1f}",
            "reproj": f"{best['reproj']:.3f}",
            "ground_m": f"{best['ground_m']:.3f}",
            "bearing_deg": f"{best['bearing_deg']:.3f}",
            "ray_ang": f"{best['ray_ang']:.3f}",
            "view_rel_deg": f"{best['view_rel_deg']:.3f}",
            "view_sym_deg": f"{best['view_sym_deg']:.3f}",
            "quality": f"{best['quality']:.3f}",
            "weight": f"{best['weight']:.6f}",
            "pos_err_cm": f"{pos_err:.3f}",
            "ang_err_deg": f"{ang_err:.3f}",
            "file": meta["file"],
        })

    # write single CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)

    print("[DONE] wrote:", out_csv)
    print("Columns:", ", ".join(fieldnames))


if __name__ == "__main__":
    main()