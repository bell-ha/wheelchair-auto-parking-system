#!/usr/bin/env python3
# offline_pose_from_snapshots.py
import os
import re
import glob
import math
import argparse
from dataclasses import dataclass

import cv2
import numpy as np

# =========================
# Intrinsic (shared fisheye) - from your tracker
# =========================
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE_M = 0.25  # 25cm

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
# Helpers (same conventions as your tracker)
# =========================
def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi

def compass_deg_to_map_rad(compass_deg: float) -> float:
    # compass: 0=N,90=E -> map: 0=+x, 90=+y(아래)
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)

def smooth01(x, x0, x1):
    if x <= x0:
        return 1.0
    if x >= x1:
        return 0.0
    t = (x - x0) / (x1 - x0)
    return float(1.0 - t)

@dataclass
class CamCfg:
    key: str
    pos_px: np.ndarray        # camera position on map (px == cm)
    h_cm: float               # camera mounting height (cm)
    map_angle_deg: float      # base ray direction on map
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90   # from your file (0.90)

class OfflinePoseEstimator:
    def __init__(self, show: bool = True):
        # Map params (same style)
        self.map_w, self.map_h = 1000, 1000
        self.off_x, self.off_y = 200, 150
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 1.0  # 1px=1cm

        # Marker height settings
        self.marker_h_cm_by_id = {0: 70.0, 1: 70.0}
        self.marker_h_cm_default = 70.0

        # marker->center offset (cm)
        self.center_offset_cm_by_id = {0: 23.0, 1: 23.0}

        # Quality params (copied spirit)
        self.reproj_good_px = 2.0
        self.reproj_bad_px = 8.0
        self.area_good_px2 = 2500.0
        self.area_bad_px2 = 600.0
        self.min_quality_w = 0.08

        # Camera configs (from your pasted code)
        self.cams = {
            "rear": CamCfg(
                key="rear",
                pos_px=np.array([301.4 + self.off_x, 540.0 + self.off_y], dtype=np.float32),
                h_cm=105.5,
                map_angle_deg=90.0,
                sens=1.6,
                install_angle=0.0,
                install_offset=0.0,
                yaw_trim_deg=3.0,
                dist_gain=0.90,
            ),
            "left": CamCfg(
                key="left",
                pos_px=np.array([200.0 + self.off_x, 270.0 + self.off_y], dtype=np.float32),
                h_cm=110.0,
                map_angle_deg=157.0,
                sens=1.6,
                install_angle=113.0,
                install_offset=50.84,
                yaw_trim_deg=8.0,
                dist_gain=0.90,
            ),
        }

        self.show = show

    def marker_to_center(self, marker_pos_px: np.ndarray, heading_map_rad: float, marker_id: int) -> np.ndarray:
        offset_cm = float(self.center_offset_cm_by_id.get(marker_id, 23.0))
        offset_px = offset_cm * self.map_scale
        dx = offset_px * math.cos(heading_map_rad)
        dy = offset_px * math.sin(heading_map_rad)
        # same rule as your code
        if marker_id == 0:
            return marker_pos_px - np.array([dx, dy], dtype=np.float32)
        else:
            return marker_pos_px + np.array([dx, dy], dtype=np.float32)

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
            und = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)

            ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            tvec = tvec.reshape(3).astype(np.float32)
            dist_m = float(np.linalg.norm(tvec))

            # ground distance (height compensation)
            mh = float(self.marker_h_cm_by_id.get(mid, self.marker_h_cm_default))
            dh_m = abs(cam.h_cm - mh) / 100.0
            ground_m = math.sqrt(max(0.0, dist_m * dist_m - dh_m * dh_m))
            ground_cm = ground_m * 100.0 * cam.dist_gain

            # bearing -> map ray
            bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
            ray_deg = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
            ray_rad = math.radians(ray_deg)

            marker_pos = cam.pos_px + np.array([
                ground_cm * self.map_scale * math.cos(ray_rad),
                ground_cm * self.map_scale * math.sin(ray_rad)
            ], dtype=np.float32)

            # yaw from rvec -> compass -> map heading
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            raw_yaw_deg = math.degrees(math.atan2(-rmat[2, 0], sy))

            total = (raw_yaw_deg * cam.sens) + cam.install_angle
            yaw_compass = total - cam.install_offset
            yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)

            heading_map = compass_deg_to_map_rad(yaw_compass)
            center_pos = self.marker_to_center(marker_pos, heading_map, mid)

            # quality + weight
            area = float(abs(cv2.contourArea(c2)))
            proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(np.float32)
            reproj_err = float(np.mean(np.linalg.norm(proj - und.reshape(-1, 2).astype(np.float32), axis=1)))

            z = float(tvec[2])
            z_score = 1.0 if z > 0.05 else 0.0

            s_area = smooth01(area, self.area_good_px2, self.area_bad_px2)
            s_err = smooth01(reproj_err, self.reproj_good_px, self.reproj_bad_px)
            quality = max(self.min_quality_w, (0.45 * s_err + 0.45 * s_area + 0.10 * z_score))

            cx = float(np.mean(c2[:, 0]))
            rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
            w_center = max(0.1, 1.0 - abs(rel_x))
            w_dist = 1.0 / (1.0 + ground_m)
            w_base = float(max(0.05, w_center * w_dist))

            w = float(w_base * quality)

            dets.append({
                "cam_key": cam.key,
                "marker_id": mid,
                "marker_pos": marker_pos,
                "center_pos": center_pos,
                "heading": heading_map,
                "weight": w,
                "dbg": {
                    "quality": quality,
                    "area": area,
                    "reproj": reproj_err,
                    "ground_m": ground_m,
                    "bearing_deg": bearing_deg,
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
        return center, heading, total_w

    def draw_map(self, dets, fused):
        m = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15

        # grid (simple)
        for y in range(0, self.map_h, 50):
            cv2.line(m, (0, y), (self.map_w, y), (30, 30, 30), 1)
        for x in range(0, self.map_w, 50):
            cv2.line(m, (x, 0), (x, self.map_h), (30, 30, 30), 1)

        # camera points
        for cam in self.cams.values():
            cp = tuple(cam.pos_px.astype(int))
            cv2.circle(m, cp, 6, (220, 220, 220), -1)
            cv2.putText(m, cam.key, (cp[0] + 8, cp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # detections
        for d in dets:
            mp = tuple(d["marker_pos"].astype(int))
            cc = tuple(d["center_pos"].astype(int))
            col = (0, 255, 255) if d["cam_key"] == "rear" else (255, 180, 0)
            cv2.circle(m, mp, 4, col, -1)
            cv2.circle(m, cc, 4, col, 1)
            cv2.putText(m, f"{d['cam_key']}/ID{d['marker_id']} w={d['weight']:.2f}",
                        (mp[0] + 6, mp[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        if fused is not None:
            center, heading, total_w = fused
            c = center.astype(int)
            cv2.circle(m, tuple(c), 7, (0, 255, 0), -1)
            L = int(60 * self.map_scale)
            hx = int(c[0] + L * math.cos(heading))
            hy = int(c[1] + L * math.sin(heading))
            cv2.arrowedLine(m, tuple(c), (hx, hy), (0, 255, 0), 2, tipLength=0.25)
            cv2.putText(m, f"FUSED wsum={total_w:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(m, f"pos=({center[0]:.1f},{center[1]:.1f})cm  heading={math.degrees(heading):.1f}deg(map)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

        return m

def pair_snapshots(folder: str):
    rear_list = sorted(glob.glob(os.path.join(folder, "rear_*.jpg")))
    pairs = []
    pat = re.compile(r"rear_(.+)\.jpg$")
    for rp in rear_list:
        m = pat.search(os.path.basename(rp))
        if not m:
            continue
        key = m.group(1)
        lp = os.path.join(folder, f"left_{key}.jpg")
        if os.path.exists(lp):
            pairs.append((key, rp, lp))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="snapshots_rear_left", help="snapshot directory")
    ap.add_argument("--start", type=int, default=0, help="start index")
    ap.add_argument("--max", type=int, default=0, help="max pairs (0=all)")
    ap.add_argument("--show", action="store_true", help="show visualization windows")
    args = ap.parse_args()

    pairs = pair_snapshots(args.dir)
    if not pairs:
        print(f"[ERR] no pairs found in {args.dir} (expect rear_*.jpg and left_*.jpg)")
        return

    pairs = pairs[args.start:]
    if args.max and args.max > 0:
        pairs = pairs[:args.max]

    est = OfflinePoseEstimator(show=args.show)

    print(f"[INFO] pairs={len(pairs)}  dir={args.dir}")
    if args.show:
        print("[Keys] n/SPACE: next | q/ESC: quit")

    for idx, (key, rp, lp) in enumerate(pairs):
        fr_r = cv2.imread(rp)
        fr_l = cv2.imread(lp)
        if fr_r is None or fr_l is None:
            print(f"[WARN] failed to read: {rp} or {lp}")
            continue

        dets = []
        dets += est.estimate_from_frame(fr_r, est.cams["rear"])
        dets += est.estimate_from_frame(fr_l, est.cams["left"])

        fused = est.fuse(dets)

        # ---- console output
        if fused is None:
            print(f"[{idx:04d}] {key}  dets={len(dets)}  => FUSED: None")
        else:
            center, heading, wsum = fused
            print(f"[{idx:04d}] {key}  dets={len(dets)}  wsum={wsum:.2f}  "
                  f"=> x={center[0]:.1f}cm y={center[1]:.1f}cm  heading={math.degrees(heading):.1f}deg(map)")

            # optional: per-det debug
            for d in sorted(dets, key=lambda x: -x["weight"])[:3]:
                dbg = d["dbg"]
                print(f"         - {d['cam_key']}/ID{d['marker_id']} w={d['weight']:.2f} "
                      f"q={dbg['quality']:.2f} area={dbg['area']:.0f} reproj={dbg['reproj']:.2f} ground={dbg['ground_m']:.2f}m")

        # ---- visualization
        if args.show:
            mon_r = cv2.resize(fr_r, (640, 360))
            mon_l = cv2.resize(fr_l, (640, 360))
            cv2.imshow("rear | left", cv2.hconcat([mon_r, mon_l]))

            m = est.draw_map(dets, fused)
            cv2.imshow("map", m)

            keycode = cv2.waitKey(0) & 0xFF
            if keycode in (27, ord('q')):
                break

    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()