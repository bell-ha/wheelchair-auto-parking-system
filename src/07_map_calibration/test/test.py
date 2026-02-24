#!/usr/bin/env python3
# test_snapshots_folder_v3params.py
#
# 목표:
# - snapshots_rear_left/ 폴더의 모든 이미지에 대해
#   ArUco+PnP로 추정 -> (저장된 calib_params_ridge.csv로) 보정 적용 -> GT와 비교
# - 학습 절대 없음(테스트만)
#
# 입력:
#   --folder snapshots_rear_left
#   --params analyze/calib_params_ridge.csv
#
# 출력:
#   analyze/test_snapshots_results.csv
#
import os, re, glob, math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


# =========================
# Intrinsic / Distortion
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
def wrap360(d): return (d % 360 + 360) % 360
def wrap180(d): return (d + 180) % 360 - 180
def angdiff_deg(a, b): return wrap180(a - b)

def south0_to_map_deg(south0_deg: float) -> float:
    return wrap360(south0_deg + 90.0)

def rayvec_from_south0(ray_ang_south0_deg: float):
    map_deg = south0_to_map_deg(ray_ang_south0_deg)
    r = math.radians(map_deg)
    return math.cos(r), math.sin(r)

def from_ray_frame(e_par, e_perp, ray_ang):
    ux, uy = rayvec_from_south0(ray_ang)
    vx, vy = -uy, ux
    dx = e_par*ux + e_perp*vx
    dy = e_par*uy + e_perp*vy
    return dx, dy

def compass_deg_to_map_rad(compass_deg: float) -> float:
    mdeg = (compass_deg + 270.0) % 360.0
    return math.radians(mdeg)

def map_deg_to_south0_deg(map_deg: float) -> float:
    return wrap360(map_deg - 90.0)

def view_sym_from_rel(rel_deg: float) -> float:
    a = abs(rel_deg)
    return float(min(a, abs(180.0 - a)))

# =========================
# Camera config
# =========================
@dataclass
class CamCfg:
    key: str
    pos_px: np.ndarray
    h_cm: float
    map_angle_deg: float
    sens: float
    install_angle: float
    install_offset: float
    yaw_trim_deg: float = 0.0
    dist_gain: float = 0.90

CAMS = {
    "rear": CamCfg("rear", np.array([301.4, 540.0], np.float32),
                   105.5, 90.0, 1.6, 0.0, 0.0, yaw_trim_deg=3.0, dist_gain=0.90),
    "left": CamCfg("left", np.array([200.0, 270.0], np.float32),
                   110.0, 157.0, 1.6, 113.0, 50.84, yaw_trim_deg=8.0, dist_gain=0.90),
}
MARKER_H_CM = {0: 70.0, 1: 70.0}
CENTER_OFFSET_CM = {0: 23.0, 1: 23.0}

# =========================
# Quality
# =========================
REPROJ_GOOD = 2.0
REPROJ_BAD  = 8.0
AREA_GOOD   = 2500.0
AREA_BAD    = 600.0
MIN_QUALITY = 0.08

def smooth01(x, x0, x1):
    if x <= x0: return 1.0
    if x >= x1: return 0.0
    t = (x - x0) / (x1 - x0)
    return float(1.0 - t)

# =========================
# Filename parsing
# =========================
PAT = re.compile(r"^(left|rear)_x([+-]?\d+)_y([+-]?\d+)_([+-]?\d+)_([0-9]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def parse_meta(path: str):
    b = os.path.basename(path)
    m = PAT.match(b)
    if not m:
        return None
    return {
        "file": b,
        "path": path,
        "cam": m.group(1).lower(),
        "gt_x": int(m.group(2)),
        "gt_y": int(m.group(3)),
        "gt_ang": int(m.group(4)),
        "seq": int(m.group(5)),
        "pair_key": f"{int(m.group(2))}_{int(m.group(3))}_{int(m.group(4))}_{int(m.group(5))}",
    }

# =========================
# Estimation (raw)
# =========================
def estimate(img_bgr, cam_key):
    cam = CAMS[cam_key]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    best = None
    best_score = -1.0

    for i, mid_arr in enumerate(ids):
        mid = int(mid_arr[0])
        if mid not in (0, 1):
            continue

        c2 = corners[i].reshape(4,2).astype(np.float32)
        und = cv2.fisheye.undistortPoints(corners[i].reshape(-1,1,2), K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(OBJ_POINTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue

        tvec = tvec.reshape(3).astype(np.float32)
        if float(tvec[2]) <= 0.01:
            continue

        dist_m = float(np.linalg.norm(tvec))
        mh = float(MARKER_H_CM.get(mid, 70.0))
        dh_m = abs(cam.h_cm - mh) / 100.0
        ground_m = math.sqrt(max(0.0, dist_m*dist_m - dh_m*dh_m))
        ground_cm = ground_m * 100.0 * cam.dist_gain

        bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
        ray_map_deg = cam.map_angle_deg + cam.yaw_trim_deg + bearing_deg
        ray_map_rad = math.radians(ray_map_deg)

        marker_x = float(cam.pos_px[0] + ground_cm * math.cos(ray_map_rad))
        marker_y = float(cam.pos_px[1] + ground_cm * math.sin(ray_map_rad))

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        raw_yaw_deg = math.degrees(math.atan2(-rmat[2,0], sy))

        total = (raw_yaw_deg * cam.sens) + cam.install_angle
        yaw_compass = total - cam.install_offset
        yaw_compass = wrap360(yaw_compass + 180.0) if mid == 1 else wrap360(yaw_compass)

        heading_map_rad = compass_deg_to_map_rad(yaw_compass)
        heading_map_deg = wrap360(math.degrees(heading_map_rad))
        pred_ang = map_deg_to_south0_deg(heading_map_deg)

        off = float(CENTER_OFFSET_CM.get(mid, 23.0))
        head_map_rad = math.radians((pred_ang + 90.0) % 360.0)
        dx = off * math.cos(head_map_rad)
        dy = off * math.sin(head_map_rad)
        sign = -1.0 if mid == 0 else +1.0
        pred_x = marker_x + sign * dx
        pred_y = marker_y + sign * dy

        proj, _ = cv2.projectPoints(OBJ_POINTS, rvec, tvec, K, None)
        proj = proj.reshape(-1,2).astype(np.float32)
        reproj = float(np.mean(np.linalg.norm(proj - und.reshape(-1,2).astype(np.float32), axis=1)))

        area = float(abs(cv2.contourArea(c2)))
        z_score = 1.0 if float(tvec[2]) > 0.05 else 0.0
        s_area = smooth01(area, AREA_GOOD, AREA_BAD)
        s_err  = smooth01(reproj, REPROJ_GOOD, REPROJ_BAD)
        quality = max(MIN_QUALITY, (0.45*s_err + 0.45*s_area + 0.10*z_score))

        ray_ang = map_deg_to_south0_deg(wrap360(ray_map_deg))
        view_rel = wrap180(pred_ang - ray_ang)
        view_sym = view_sym_from_rel(view_rel)

        score = quality / (1.0 + ground_m)

        est = dict(
            pred_x=float(pred_x), pred_y=float(pred_y), pred_ang=float(pred_ang),
            ground_m=float(ground_m), bearing_deg=float(bearing_deg), ray_ang=float(ray_ang),
            view_sym_deg=float(view_sym), reproj=float(reproj), quality=float(quality),
            marker_id=int(mid)
        )
        if score > best_score:
            best_score = score
            best = est

    return best

# =========================
# Load params and apply (v3.1 or v1)
# =========================
def load_calib_params(path: Path):
    df = pd.read_csv(path)
    cols = set(df.columns)
    v3 = any(c.startswith("wpar_") for c in cols) and any(c.startswith("wper_") for c in cols)
    v1 = ("wx_bias" in cols and "wy_bias" in cols)

    params = {}
    for _, r in df.iterrows():
        cam = str(r["cam"]).lower()
        if v3:
            feat = ["bias","g","g2","b","b2","v","g*b","reproj"]
            wpar = np.array([r[f"wpar_{f}"] for f in feat], float)
            wper = np.array([r[f"wper_{f}"] for f in feat], float)
            wtheta = None
            if any(c.startswith("wtheta_") for c in cols):
                wtheta = np.array([r[f"wtheta_{f}"] for f in feat], float)
            params[cam] = ("v3", wpar, wper, wtheta)
        elif v1:
            feat = ["bias","ground_m","abs_bearing","view_sym","ground_m*abs_bearing"]
            wx = np.array([r[f"wx_{f}"] for f in feat], float)
            wy = np.array([r[f"wy_{f}"] for f in feat], float)
            params[cam] = ("v1", wx, wy, None)
        else:
            params[cam] = ("none", None, None, None)
    return params

def X_v3(est):
    g = float(est["ground_m"])
    b = abs(float(est["bearing_deg"]))
    v = float(est["view_sym_deg"])
    r = float(est["reproj"])
    return np.array([1.0, g, g*g, b, b*b, v, g*b, r], float)

def X_v1(est):
    g = float(est["ground_m"])
    b = abs(float(est["bearing_deg"]))
    v = float(est["view_sym_deg"])
    return np.array([1.0, g, b, v, g*b], float)

def apply_calib(est, cam, params):
    kind, a, b, c = params[cam]
    x = float(est["pred_x"]); y = float(est["pred_y"]); ang = float(est["pred_ang"])
    if kind == "v3":
        wpar, wper, wtheta = a, b, c
        X = X_v3(est)
        epar = float(X @ wpar)
        eperp = float(X @ wper)
        dx, dy = from_ray_frame(epar, eperp, float(est["ray_ang"]))
        x2, y2 = x + dx, y + dy
        if wtheta is not None:
            dth = float(X @ wtheta)
            ang2 = wrap360(ang + dth)
        else:
            ang2 = ang
        return x2, y2, ang2
    if kind == "v1":
        wx, wy = a, b
        X = X_v1(est)
        dx = float(X @ wx)
        dy = float(X @ wy)
        return x+dx, y+dy, ang
    return x, y, ang

# =========================
# Simple per-pair fuse (no temporal)
# =========================
def fuse_two(mr, ml):
    if mr is None and ml is None:
        return None
    if mr is None:
        return {"x": ml["x"], "y": ml["y"], "ang": ml["ang"]}
    if ml is None:
        return {"x": mr["x"], "y": mr["y"], "ang": mr["ang"]}

    def score(m):
        q = float(m.get("quality", 0.5))
        gm = float(m.get("ground_m", 2.0))
        vs = float(m.get("view_sym_deg", 45.0))
        return max(1e-6, q * (1.0/(1.0+gm)) * (1.0/(1.0+(vs/45.0)**2)))

    sr, sl = score(mr), score(ml)
    sw = sr + sl
    x = (sr*mr["x"] + sl*ml["x"]) / sw
    y = (sr*mr["y"] + sl*ml["y"]) / sw

    # angle average via sin/cos
    cr, srn = math.cos(math.radians(mr["ang"])), math.sin(math.radians(mr["ang"]))
    cl, sln = math.cos(math.radians(ml["ang"])), math.sin(math.radians(ml["ang"]))
    c = (sr*cr + sl*cl) / sw
    s = (sr*srn + sl*sln) / sw
    ang = wrap360(math.degrees(math.atan2(s, c)))
    return {"x": x, "y": y, "ang": ang}

# =========================
# Main
# =========================
def main():
    folder = "snapshots_rear_left"
    if not os.path.isdir(folder) and os.path.isdir("/snapshots_rear_left"):
        folder = "/snapshots_rear_left"

    params_path = Path("analyze/calib_params_ridge.csv")
    if not params_path.exists():
        raise FileNotFoundError(f"missing params: {params_path}")

    out_path = Path("analyze/test_snapshots_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect images
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        paths += glob.glob(os.path.join(folder, ext))
    metas = [parse_meta(p) for p in sorted(paths)]
    metas = [m for m in metas if m is not None]
    if not metas:
        raise RuntimeError("no snapshot images matched filename pattern")

    params = load_calib_params(params_path)

    # group by pair_key
    pairs = {}
    for m in metas:
        pairs.setdefault(m["pair_key"], {})[m["cam"]] = m

    rows = []
    for k, d in pairs.items():
        rear_m = d.get("rear")
        left_m = d.get("left")

        # GT shared
        gt_x = rear_m["gt_x"] if rear_m else left_m["gt_x"]
        gt_y = rear_m["gt_y"] if rear_m else left_m["gt_y"]
        gt_ang = rear_m["gt_ang"] if rear_m else left_m["gt_ang"]
        seq = rear_m["seq"] if rear_m else left_m["seq"]

        # rear
        mr = None
        if rear_m:
            img = cv2.imread(rear_m["path"])
            est = None if img is None else estimate(img, "rear")
            if est:
                x,y,a = apply_calib(est, "rear", params)
                mr = {"x": x, "y": y, "ang": a,
                      "quality": est["quality"], "ground_m": est["ground_m"], "view_sym_deg": est["view_sym_deg"]}
                rows.append({
                    "pair_key": k, "cam": "rear",
                    "file": rear_m["file"], "seq": seq,
                    "gt_x": gt_x, "gt_y": gt_y, "gt_ang": gt_ang,
                    "status": "ok",
                    "pred_x": x, "pred_y": y, "pred_ang": a,
                    "pos_err_cm": math.hypot(x-gt_x, y-gt_y),
                    "ang_err_deg": abs(angdiff_deg(a, gt_ang)),
                })
            else:
                rows.append({
                    "pair_key": k, "cam": "rear",
                    "file": rear_m["file"], "seq": seq,
                    "gt_x": gt_x, "gt_y": gt_y, "gt_ang": gt_ang,
                    "status": "no_marker",
                })

        # left
        ml = None
        if left_m:
            img = cv2.imread(left_m["path"])
            est = None if img is None else estimate(img, "left")
            if est:
                x,y,a = apply_calib(est, "left", params)
                ml = {"x": x, "y": y, "ang": a,
                      "quality": est["quality"], "ground_m": est["ground_m"], "view_sym_deg": est["view_sym_deg"]}
                rows.append({
                    "pair_key": k, "cam": "left",
                    "file": left_m["file"], "seq": seq,
                    "gt_x": gt_x, "gt_y": gt_y, "gt_ang": gt_ang,
                    "status": "ok",
                    "pred_x": x, "pred_y": y, "pred_ang": a,
                    "pos_err_cm": math.hypot(x-gt_x, y-gt_y),
                    "ang_err_deg": abs(angdiff_deg(a, gt_ang)),
                })
            else:
                rows.append({
                    "pair_key": k, "cam": "left",
                    "file": left_m["file"], "seq": seq,
                    "gt_x": gt_x, "gt_y": gt_y, "gt_ang": gt_ang,
                    "status": "no_marker",
                })

        # fused (per pair)
        fused = fuse_two(mr, ml)
        if fused is not None:
            rows.append({
                "pair_key": k, "cam": "fused",
                "file": "", "seq": seq,
                "gt_x": gt_x, "gt_y": gt_y, "gt_ang": gt_ang,
                "status": "ok",
                "pred_x": fused["x"], "pred_y": fused["y"], "pred_ang": fused["ang"],
                "pos_err_cm": math.hypot(fused["x"]-gt_x, fused["y"]-gt_y),
                "ang_err_deg": abs(angdiff_deg(fused["ang"], gt_ang)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("[DONE] wrote:", out_path)

    ok = df[(df["status"]=="ok") & df["cam"].isin(["rear","left","fused"])].copy()
    for cam in ["rear","left","fused"]:
        sub = ok[ok["cam"]==cam]
        if len(sub)==0:
            continue
        print(f"[{cam}] n={len(sub)} pos_med={sub['pos_err_cm'].median():.2f} pos_p90={sub['pos_err_cm'].quantile(0.9):.2f}  ang_med={sub['ang_err_deg'].median():.2f} ang_p90={sub['ang_err_deg'].quantile(0.9):.2f}")

if __name__ == "__main__":
    main()