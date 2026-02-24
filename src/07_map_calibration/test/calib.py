#!/usr/bin/env python3
# calib_v3_1_rayframe_anchor_rear_with_heading.py
#
# INPUT :
#   analyze/tocsv.csv
#   required columns:
#     cam,status,gt_x,gt_y,gt_ang, pred_ang,
#     ground_m,bearing_deg,ray_ang, view_sym_deg,reproj,quality,marker_id,seq
#
# OUTPUT:
#   analyze/tocsv_calib.csv
#   analyze/toimage_calib.png
#   analyze/calib_params_ridge.csv
#
# v3.1 핵심:
# - 위치 보정: ray frame residuals (e_par, e_perp) 학습 -> xy로 변환 적용
# - 각도 보정: dtheta residual 학습 -> pred_ang_calib 적용 (wrap360)
# - feature: [1, g, g^2, b, b^2, v, g*b, reproj]
# - rear는 GT 타깃
# - left는 (1-a)*GT + a*rear_phys 타깃 (rear 신뢰도 기반 a, cap 적용)
# - 튐 억제: residual clipping
#
import math
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


# ---------------- angle helpers ----------------
def wrap360(d): return (d % 360 + 360) % 360
def wrap180(d): return (d + 180) % 360 - 180
def angdiff_deg(a, b):  # a-b wrapped to [-180,180)
    return wrap180(a - b)

def south0_to_map_deg(south0_deg: float) -> float:
    return wrap360(south0_deg + 90.0)

def rayvec_from_south0(ray_ang_south0_deg: float):
    """south0 angle -> unit vector in map coords (x right, y down)"""
    map_deg = south0_to_map_deg(ray_ang_south0_deg)
    r = math.radians(map_deg)
    return math.cos(r), math.sin(r)

def ang_to_vec_deg(a):
    r = math.radians(a)
    return math.cos(r), math.sin(r)

def vec_to_ang_deg(c, s):
    return (math.degrees(math.atan2(s, c)) + 360) % 360

def circular_blend_deg(a_deg, b_deg, alpha):
    """
    blend two angles on unit circle:
      result = normalize((1-alpha)*va + alpha*vb)
    alpha in [0,1]
    """
    ca, sa = ang_to_vec_deg(a_deg)
    cb, sb = ang_to_vec_deg(b_deg)
    c = (1.0 - alpha) * ca + alpha * cb
    s = (1.0 - alpha) * sa + alpha * sb
    # handle near-zero vector
    if abs(c) < 1e-9 and abs(s) < 1e-9:
        return float(a_deg)
    return vec_to_ang_deg(c, s)


# ---------------- base physical model (same as before) ----------------
def predict_phys(rows, cam_defaults, dist_gain, heading_offset_deg, center_offset_cm=23.0):
    """
    returns: pred_x_phys, pred_y_phys, pred_ang_phys(south0)
    NOTE: pred_ang_phys here is just pred_ang + heading_offset (not recomputing from rvec; we only adjust constant)
    """
    cam_x = cam_defaults["cam_x"]
    cam_y = cam_defaults["cam_y"]
    map_angle = cam_defaults["map_angle_deg"]
    yaw_trim = cam_defaults["yaw_trim_deg"]

    ground_m = rows["ground_m"].to_numpy(dtype=float)
    bearing = rows["bearing_deg"].to_numpy(dtype=float)
    pred_ang = rows["pred_ang"].to_numpy(dtype=float)
    mid = rows["marker_id"].to_numpy(dtype=int)

    ray_deg = map_angle + yaw_trim + bearing
    ray_rad = np.deg2rad(ray_deg)

    ground_cm = ground_m * 100.0 * float(dist_gain)
    marker_x = cam_x + ground_cm * np.cos(ray_rad)
    marker_y = cam_y + ground_cm * np.sin(ray_rad)

    head = (pred_ang + float(heading_offset_deg)) % 360.0

    # marker -> center offset in map
    head_map_rad = np.deg2rad((head + 90.0) % 360.0)  # south0->map
    dx = center_offset_cm * np.cos(head_map_rad)
    dy = center_offset_cm * np.sin(head_map_rad)

    sign = np.where(mid == 0, -1.0, +1.0)
    center_x = marker_x + sign * dx
    center_y = marker_y + sign * dy
    return center_x, center_y, head


# ---------------- Ridge regression ----------------
def ridge_fit(X, y, lam):
    D = X.shape[1]
    A = X.T @ X + lam * np.eye(D)
    b = X.T @ y
    return np.linalg.solve(A, b)

def ridge_predict(X, w):
    return X @ w

def make_features(df):
    """
    X = [1, g, g^2, b, b^2, v, g*b, reproj]
      g = ground_m
      b = abs(bearing_deg)
      v = view_sym_deg
    """
    g = df["ground_m"].to_numpy(dtype=float)
    b = np.abs(df["bearing_deg"].to_numpy(dtype=float))
    v = df["view_sym_deg"].to_numpy(dtype=float)
    r = df["reproj"].to_numpy(dtype=float)

    g = np.nan_to_num(g, nan=float(np.nanmedian(g)))
    b = np.nan_to_num(b, nan=float(np.nanmedian(b)))
    v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
    r = np.nan_to_num(r, nan=float(np.nanmedian(r)))

    X = np.stack([
        np.ones_like(g),
        g,
        g*g,
        b,
        b*b,
        v,
        g*b,
        r,
    ], axis=1).astype(float)
    return X


# ---------------- rear anchor alpha ----------------
def rear_anchor_alpha(rear_rows: pd.DataFrame):
    """
    rear 신뢰도 기반 a in [0.05, 0.65]
    """
    q = rear_rows["quality"].to_numpy(dtype=float)
    r = rear_rows["reproj"].to_numpy(dtype=float)
    g = rear_rows["ground_m"].to_numpy(dtype=float)
    v = rear_rows["view_sym_deg"].to_numpy(dtype=float)

    q = np.nan_to_num(q, nan=float(np.nanmedian(q)))
    r = np.nan_to_num(r, nan=float(np.nanmedian(r)))
    g = np.nan_to_num(g, nan=float(np.nanmedian(g)))
    v = np.nan_to_num(v, nan=float(np.nanmedian(v)))

    s_reproj = np.clip((8.0 - r) / (8.0 - 1.5), 0.0, 1.0)
    s_dist   = 1.0 / (1.0 + g)
    s_view   = 1.0 / (1.0 + (v/45.0)**2)
    s = np.clip(q, 0.0, 1.0) * s_reproj * s_dist * s_view

    a = 0.05 + 0.60 * np.clip(s / 0.35, 0.0, 1.0)
    return a


# ---------------- ray frame conversions ----------------
def to_ray_frame_errors(gt_x, gt_y, pred_x, pred_y, ray_ang_south0_deg):
    dx = gt_x - pred_x
    dy = gt_y - pred_y
    ux, uy = rayvec_from_south0(float(ray_ang_south0_deg))
    vx, vy = -uy, ux
    e_par = dx * ux + dy * uy
    e_perp = dx * vx + dy * vy
    return e_par, e_perp

def from_ray_frame_errors(pred_x, pred_y, e_par, e_perp, ray_ang_south0_deg):
    ux, uy = rayvec_from_south0(float(ray_ang_south0_deg))
    vx, vy = -uy, ux
    dx = e_par * ux + e_perp * vx
    dy = e_par * uy + e_perp * vy
    return pred_x + dx, pred_y + dy


# ---------------- Drawing ----------------
def draw_grid(img, step=50):
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (30, 30, 30), 1)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (30, 30, 30), 1)

def draw_arrow(img, x, y, deg_map, color, length=26, thickness=2):
    rad = math.radians(deg_map)
    x2 = int(round(x + length * math.cos(rad)))
    y2 = int(round(y + length * math.sin(rad)))
    cv2.arrowedLine(img, (int(round(x)), int(round(y))), (x2, y2),
                    color, thickness, tipLength=0.25)

def draw_x(img, x, y, color, size=10, thickness=2):
    xi = int(round(x)); yi = int(round(y))
    cv2.line(img, (xi - size, yi - size), (xi + size, yi + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (xi - size, yi + size), (xi + size, yi - size), color, thickness, cv2.LINE_AA)

def alpha_blend_circle(dst, center, radius, bgr, alpha):
    overlay = dst.copy()
    cv2.circle(overlay, center, radius, bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), dst, float(1.0-alpha), 0.0, dst)

def draw_3panel(df, out_png: Path):
    MAP_W, MAP_H = 1000, 1000
    gap, top_pad = 30, 70
    canvas_w = MAP_W * 3 + gap * 4
    canvas_h = MAP_H + top_pad + gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)

    panel_x = [gap + i * (MAP_W + gap) for i in range(3)]
    panel_y = top_pad
    buckets = [0, 45, -45]
    title = {0:"GT heading=0 [CALIB v3.1]", 45:"GT heading=+45 [CALIB v3.1]", -45:"GT heading=-45 [CALIB v3.1]"}

    CAR_TL = (200, 180)
    CAR_BR = (400, 540)
    CAM_REAR = (301, 540)
    CAM_LEFT = (200, 270)

    COL_PRED = {"rear": (0,255,0), "left": (255,0,0)}

    panels = []
    for _ in range(3):
        p = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        p[:] = (18,18,18)
        draw_grid(p, 50)
        cv2.rectangle(p, CAR_TL, CAR_BR, (80,80,220), 2)
        cv2.circle(p, CAM_REAR, 7, (0,255,255), -1)
        cv2.putText(p, "rear", (CAM_REAR[0]+10, CAM_REAR[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.circle(p, CAM_LEFT, 7, (0,255,255), -1)
        cv2.putText(p, "left", (CAM_LEFT[0]+10, CAM_LEFT[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        panels.append(p)

    processed = {0:0,45:0,-45:0}
    ok_by_cam = {0:{"rear":0,"left":0},45:{"rear":0,"left":0},-45:{"rear":0,"left":0}}
    miss = {0:0,45:0,-45:0}

    for _, r in df.iterrows():
        if pd.isna(r["gt_ang"]) or int(r["gt_ang"]) not in buckets:
            continue
        ang = int(r["gt_ang"])
        idx = buckets.index(ang)
        processed[ang] += 1

        gt_x, gt_y = r["gt_x"], r["gt_y"]
        if pd.isna(gt_x) or pd.isna(gt_y):
            continue
        gt_x, gt_y = float(gt_x), float(gt_y)

        gt_map = south0_to_map_deg(float(ang))
        cv2.circle(panels[idx], (int(round(gt_x)), int(round(gt_y))), 3, (200,200,200), -1)
        draw_arrow(panels[idx], gt_x, gt_y, gt_map, (200,200,200), length=26, thickness=1)

        if r["status"] != "ok":
            miss[ang] += 1
            draw_x(panels[idx], gt_x, gt_y, (0,0,255), size=9, thickness=2)
            continue

        cam = str(r["cam"]).lower()
        if cam not in ("rear","left"): cam = "rear"
        ok_by_cam[ang][cam] += 1

        px, py, pa = r["pred_x_calib"], r["pred_y_calib"], r["pred_ang_calib"]
        if pd.isna(px) or pd.isna(py) or pd.isna(pa):
            miss[ang] += 1
            draw_x(panels[idx], gt_x, gt_y, (0,0,255), size=9, thickness=2)
            continue

        px, py, pa = float(px), float(py), float(pa)
        cv2.line(panels[idx], (int(round(gt_x)), int(round(gt_y))), (int(round(px)), int(round(py))), (120,120,120), 1)

        pe = r.get("pos_err_calib_cm", np.nan)
        if pd.isna(pe):
            pe = math.hypot(px-gt_x, py-gt_y)
        rad = int(max(3, min(10, 3 + float(pe)/15.0)))

        gm = r.get("ground_m", np.nan)
        alpha = 0.75 if pd.isna(gm) else float(max(0.25, min(1.0, 1.2 - 0.2*float(gm))))

        col = COL_PRED[cam]
        alpha_blend_circle(panels[idx], (int(round(px)), int(round(py))), rad, col, alpha)

        vs = r.get("view_sym_deg", np.nan)
        if not pd.isna(vs) and float(vs) >= 60.0:
            cv2.circle(panels[idx], (int(round(px)), int(round(py))), rad+2, (0,255,255), 2, cv2.LINE_AA)

        pred_map = south0_to_map_deg(pa)
        draw_arrow(panels[idx], px, py, pred_map, col, length=26, thickness=2)

    for i, ang in enumerate(buckets):
        x0, y0 = panel_x[i], panel_y
        canvas[y0:y0+MAP_H, x0:x0+MAP_W] = panels[i]
        cv2.putText(canvas, title[ang], (x0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 2)
        cv2.putText(canvas, f"imgs={processed[ang]} rear_ok={ok_by_cam[ang]['rear']} left_ok={ok_by_cam[ang]['left']} miss={miss[ang]}",
                    (x0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)

    cv2.imwrite(str(out_png), canvas)


def main():
    in_csv = Path("analyze/tocsv.csv")
    if not in_csv.exists():
        raise FileNotFoundError("analyze/tocsv.csv not found")

    out_csv = Path("analyze/tocsv_calib.csv")
    out_png = Path("analyze/toimage_calib.png")
    out_params = Path("analyze/calib_params_ridge.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df["status"] = df["status"].astype(str).str.lower()
    df["cam"] = df["cam"].astype(str).str.lower()

    need = ["gt_x","gt_y","gt_ang","pred_ang","ground_m","bearing_deg","ray_ang","marker_id","view_sym_deg","reproj","quality","seq"]
    for c in need + ["pred_x","pred_y"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ok = df[df["status"].eq("ok")].copy()
    ok = ok.dropna(subset=["cam","gt_x","gt_y","gt_ang","pred_ang","ground_m","bearing_deg","ray_ang","marker_id","view_sym_deg","reproj","quality","seq"])
    if ok.empty:
        raise RuntimeError("No ok rows to train on")

    ok["pair_key"] = ok["gt_x"].astype(int).astype(str) + "_" + ok["gt_y"].astype(int).astype(str) + "_" + ok["gt_ang"].astype(int).astype(str) + "_" + ok["seq"].astype(int).astype(str)
    ok["pos_key"] = ok["gt_x"].astype(int).astype(str) + "_" + ok["gt_y"].astype(int).astype(str)

    # base camera defaults (your current config)
    base = {
        "rear": dict(cam_x=301.4, cam_y=540.0, map_angle_deg=90.0,  yaw_trim_deg=3.0),
        "left": dict(cam_x=200.0, cam_y=270.0, map_angle_deg=157.0, yaw_trim_deg=8.0),
    }
    dist_gain = {"rear": 0.90, "left": 0.90}
    heading_off = {"rear": 0.0, "left": 0.0}

    lambdas = [0.1, 1.0, 5.0, 20.0, 100.0, 300.0, 1000.0]

    # clipping (cm, deg)
    CLIP_PAR = 80.0
    CLIP_PERP = 80.0
    CLIP_DTH = 35.0   # degrees

    # learned params
    chosen = {}     # cam -> (lam_xy, lam_theta)
    weights = {}    # cam -> (wpar, wper, wtheta)

    # ---------- REAR train ----------
    rear_data = ok[ok["cam"] == "rear"].copy()
    if rear_data.empty:
        raise RuntimeError("No rear ok rows.")

    rx, ry, rang = predict_phys(rear_data, base["rear"], dist_gain["rear"], heading_off["rear"])
    rear_data["pred_x_phys"] = rx
    rear_data["pred_y_phys"] = ry
    rear_data["pred_ang_phys"] = rang

    # residuals to GT in ray frame + angle residual
    epar = []
    eperp = []
    dth = []
    for gx, gy, ga, px0, py0, pa0, ra in zip(
        rear_data["gt_x"], rear_data["gt_y"], rear_data["gt_ang"],
        rear_data["pred_x_phys"], rear_data["pred_y_phys"], rear_data["pred_ang_phys"],
        rear_data["ray_ang"]
    ):
        ep, er = to_ray_frame_errors(gx, gy, px0, py0, ra)
        epar.append(ep); eperp.append(er)
        dth.append(angdiff_deg(ga, pa0))  # target - pred
    rear_data["e_par"] = np.clip(np.array(epar, float), -CLIP_PAR, CLIP_PAR)
    rear_data["e_perp"] = np.clip(np.array(eperp, float), -CLIP_PERP, CLIP_PERP)
    rear_data["dtheta"] = np.clip(np.array(dth, float), -CLIP_DTH, CLIP_DTH)

    # choose lambda for xy
    best_lam_xy, best_score_xy = None, None
    for lam in lambdas:
        fold_meds = []
        for posk in rear_data["pos_key"].unique():
            tr = rear_data[rear_data["pos_key"] != posk]
            te = rear_data[rear_data["pos_key"] == posk]
            if len(tr) < 10 or len(te) == 0:
                continue
            Xtr = make_features(tr); Xte = make_features(te)
            wpar = ridge_fit(Xtr, tr["e_par"].to_numpy(float), lam)
            wper = ridge_fit(Xtr, tr["e_perp"].to_numpy(float), lam)
            ep_hat = ridge_predict(Xte, wpar)
            er_hat = ridge_predict(Xte, wper)

            predx = []
            predy = []
            for px0, py0, ra, ep, er in zip(te["pred_x_phys"], te["pred_y_phys"], te["ray_ang"], ep_hat, er_hat):
                x1, y1 = from_ray_frame_errors(px0, py0, ep, er, ra)
                predx.append(x1); predy.append(y1)
            predx = np.array(predx, float); predy = np.array(predy, float)

            gt_x = te["gt_x"].to_numpy(float); gt_y = te["gt_y"].to_numpy(float)
            err = np.sqrt((predx-gt_x)**2 + (predy-gt_y)**2)
            fold_meds.append(float(np.median(err)))
        if not fold_meds:
            continue
        score = float(np.median(fold_meds))
        if best_score_xy is None or score < best_score_xy:
            best_score_xy, best_lam_xy = score, lam

    best_lam_xy = best_lam_xy if best_lam_xy is not None else 100.0

    # choose lambda for theta (separately)
    best_lam_th, best_score_th = None, None
    for lam in lambdas:
        fold_meds = []
        for posk in rear_data["pos_key"].unique():
            tr = rear_data[rear_data["pos_key"] != posk]
            te = rear_data[rear_data["pos_key"] == posk]
            if len(tr) < 10 or len(te) == 0:
                continue
            Xtr = make_features(tr); Xte = make_features(te)
            wth = ridge_fit(Xtr, tr["dtheta"].to_numpy(float), lam)
            dth_hat = ridge_predict(Xte, wth)
            # predicted calibrated angle error to GT:
            pred_ang_cal = (te["pred_ang_phys"].to_numpy(float) + dth_hat) % 360.0
            gt_ang = te["gt_ang"].to_numpy(float)
            err = np.abs(np.array([angdiff_deg(a, b) for a, b in zip(pred_ang_cal, gt_ang)], float))
            fold_meds.append(float(np.median(err)))
        if not fold_meds:
            continue
        score = float(np.median(fold_meds))
        if best_score_th is None or score < best_score_th:
            best_score_th, best_lam_th = score, lam

    best_lam_th = best_lam_th if best_lam_th is not None else 100.0

    # final rear weights
    Xr = make_features(rear_data)
    wpar_r = ridge_fit(Xr, rear_data["e_par"].to_numpy(float), best_lam_xy)
    wper_r = ridge_fit(Xr, rear_data["e_perp"].to_numpy(float), best_lam_xy)
    wth_r  = ridge_fit(Xr, rear_data["dtheta"].to_numpy(float), best_lam_th)

    chosen["rear"] = (best_lam_xy, best_lam_th)
    weights["rear"] = (wpar_r, wper_r, wth_r)

    # ---------- LEFT train (rear anchored) ----------
    left_data = ok[ok["cam"] == "left"].copy()
    if left_data.empty:
        raise RuntimeError("No left ok rows.")

    lx, ly, lang = predict_phys(left_data, base["left"], dist_gain["left"], heading_off["left"])
    left_data["pred_x_phys"] = lx
    left_data["pred_y_phys"] = ly
    left_data["pred_ang_phys"] = lang

    # rear phys per pair_key (for anchoring)
    rear_anchor = ok[ok["cam"] == "rear"][["pair_key","gt_x","gt_y","gt_ang","ray_ang","quality","reproj","ground_m","view_sym_deg","bearing_deg","pred_ang","marker_id"]].copy()
    rx, ry, rang = predict_phys(rear_anchor, base["rear"], dist_gain["rear"], heading_off["rear"])
    rear_anchor["rear_x_phys"] = rx
    rear_anchor["rear_y_phys"] = ry
    rear_anchor["rear_ang_phys"] = rang

    left_join = left_data.merge(
        rear_anchor[["pair_key","rear_x_phys","rear_y_phys","rear_ang_phys","quality","reproj","ground_m","view_sym_deg"]],
        on="pair_key", how="left", suffixes=("", "_rear")
    )

    has_rear = left_join["rear_x_phys"].notna() & left_join["rear_y_phys"].notna() & left_join["rear_ang_phys"].notna()
    alpha = np.zeros(len(left_join), dtype=float)
    if has_rear.any():
        alpha[has_rear.to_numpy()] = rear_anchor_alpha(
            left_join.loc[has_rear, ["quality_rear","reproj_rear","ground_m_rear","view_sym_deg_rear"]]
            .rename(columns={"quality_rear":"quality","reproj_rear":"reproj","ground_m_rear":"ground_m","view_sym_deg_rear":"view_sym_deg"})
        )

    # position target = (1-a)*GT + a*rear_phys
    tgt_x = (1.0 - alpha) * left_join["gt_x"].to_numpy(float) + alpha * np.nan_to_num(left_join["rear_x_phys"].to_numpy(float), nan=left_join["gt_x"].to_numpy(float))
    tgt_y = (1.0 - alpha) * left_join["gt_y"].to_numpy(float) + alpha * np.nan_to_num(left_join["rear_y_phys"].to_numpy(float), nan=left_join["gt_y"].to_numpy(float))

    # angle target = circular blend between GT and rear_ang_phys
    gt_ang_arr = left_join["gt_ang"].to_numpy(float)
    rear_ang_arr = np.nan_to_num(left_join["rear_ang_phys"].to_numpy(float), nan=gt_ang_arr)
    tgt_ang = np.array([circular_blend_deg(ga, ra, a) for ga, ra, a in zip(gt_ang_arr, rear_ang_arr, alpha)], float)

    # residuals in ray frame to target + angle residual to target
    epar = []
    eperp = []
    dth = []
    for tx, ty, ta, px0, py0, pa0, ra in zip(
        tgt_x, tgt_y, tgt_ang,
        left_join["pred_x_phys"], left_join["pred_y_phys"], left_join["pred_ang_phys"],
        left_join["ray_ang"]
    ):
        ep, er = to_ray_frame_errors(tx, ty, px0, py0, ra)
        epar.append(ep); eperp.append(er)
        dth.append(angdiff_deg(ta, pa0))
    left_join["e_par"] = np.clip(np.array(epar, float), -CLIP_PAR, CLIP_PAR)
    left_join["e_perp"] = np.clip(np.array(eperp, float), -CLIP_PERP, CLIP_PERP)
    left_join["dtheta"] = np.clip(np.array(dth, float), -CLIP_DTH, CLIP_DTH)

    # choose lambda for left xy (evaluate vs GT)
    best_lam_xy, best_score_xy = None, None
    for lam in lambdas:
        fold_meds = []
        for posk in left_join["pos_key"].unique():
            tr = left_join[left_join["pos_key"] != posk]
            te = left_join[left_join["pos_key"] == posk]
            if len(tr) < 10 or len(te) == 0:
                continue
            Xtr = make_features(tr); Xte = make_features(te)
            wpar = ridge_fit(Xtr, tr["e_par"].to_numpy(float), lam)
            wper = ridge_fit(Xtr, tr["e_perp"].to_numpy(float), lam)
            ep_hat = ridge_predict(Xte, wpar)
            er_hat = ridge_predict(Xte, wper)

            predx = []
            predy = []
            for px0, py0, ra, ep, er in zip(te["pred_x_phys"], te["pred_y_phys"], te["ray_ang"], ep_hat, er_hat):
                x1, y1 = from_ray_frame_errors(px0, py0, ep, er, ra)
                predx.append(x1); predy.append(y1)
            predx = np.array(predx, float); predy = np.array(predy, float)

            gt_x = te["gt_x"].to_numpy(float); gt_y = te["gt_y"].to_numpy(float)
            err = np.sqrt((predx-gt_x)**2 + (predy-gt_y)**2)
            fold_meds.append(float(np.median(err)))
        if not fold_meds:
            continue
        score = float(np.median(fold_meds))
        if best_score_xy is None or score < best_score_xy:
            best_score_xy, best_lam_xy = score, lam
    best_lam_xy = best_lam_xy if best_lam_xy is not None else 100.0

    # choose lambda for left theta (evaluate vs GT)
    best_lam_th, best_score_th = None, None
    for lam in lambdas:
        fold_meds = []
        for posk in left_join["pos_key"].unique():
            tr = left_join[left_join["pos_key"] != posk]
            te = left_join[left_join["pos_key"] == posk]
            if len(tr) < 10 or len(te) == 0:
                continue
            Xtr = make_features(tr); Xte = make_features(te)
            wth = ridge_fit(Xtr, tr["dtheta"].to_numpy(float), lam)
            dth_hat = ridge_predict(Xte, wth)
            pred_ang_cal = (te["pred_ang_phys"].to_numpy(float) + dth_hat) % 360.0
            gt_ang = te["gt_ang"].to_numpy(float)
            err = np.abs(np.array([angdiff_deg(a, b) for a, b in zip(pred_ang_cal, gt_ang)], float))
            fold_meds.append(float(np.median(err)))
        if not fold_meds:
            continue
        score = float(np.median(fold_meds))
        if best_score_th is None or score < best_score_th:
            best_score_th, best_lam_th = score, lam
    best_lam_th = best_lam_th if best_lam_th is not None else 100.0

    # final left weights
    Xl = make_features(left_join)
    wpar_l = ridge_fit(Xl, left_join["e_par"].to_numpy(float), best_lam_xy)
    wper_l = ridge_fit(Xl, left_join["e_perp"].to_numpy(float), best_lam_xy)
    wth_l  = ridge_fit(Xl, left_join["dtheta"].to_numpy(float), best_lam_th)

    chosen["left"] = (best_lam_xy, best_lam_th)
    weights["left"] = (wpar_l, wper_l, wth_l)

    # ---------- APPLY to full df ----------
    df["pred_x_calib"] = np.nan
    df["pred_y_calib"] = np.nan
    df["pred_ang_calib"] = np.nan
    df["pos_err_calib_cm"] = np.nan
    df["ang_err_calib_deg"] = np.nan

    for cam in ["rear","left"]:
        mask = (
            (df["cam"] == cam) &
            df["ground_m"].notna() & df["bearing_deg"].notna() & df["ray_ang"].notna() &
            df["pred_ang"].notna() & df["marker_id"].notna() &
            df["view_sym_deg"].notna() & df["reproj"].notna()
        )
        rows_apply = df.loc[mask].copy()
        if rows_apply.empty:
            continue

        px0, py0, pa0 = predict_phys(rows_apply, base[cam], dist_gain[cam], heading_off[cam])
        Xap = make_features(rows_apply)
        wpar, wper, wth = weights[cam]
        ep_hat = ridge_predict(Xap, wpar)
        er_hat = ridge_predict(Xap, wper)
        dth_hat = ridge_predict(Xap, wth)

        xs = []
        ys = []
        for x0, y0, ra, ep, er in zip(px0, py0, rows_apply["ray_ang"], ep_hat, er_hat):
            x1, y1 = from_ray_frame_errors(x0, y0, ep, er, ra)
            xs.append(x1); ys.append(y1)

        df.loc[mask, "pred_x_calib"] = np.array(xs, float)
        df.loc[mask, "pred_y_calib"] = np.array(ys, float)
        df.loc[mask, "pred_ang_calib"] = (pa0 + dth_hat) % 360.0

    # errors
    okm = (df["status"]=="ok") & df["pred_x_calib"].notna() & df["pred_y_calib"].notna() & df["gt_x"].notna() & df["gt_y"].notna()
    df.loc[okm, "pos_err_calib_cm"] = np.sqrt(
        (df.loc[okm,"pred_x_calib"] - df.loc[okm,"gt_x"])**2 +
        (df.loc[okm,"pred_y_calib"] - df.loc[okm,"gt_y"])**2
    )

    okm2 = okm & df["pred_ang_calib"].notna() & df["gt_ang"].notna()
    df.loc[okm2, "ang_err_calib_deg"] = df.loc[okm2].apply(
        lambda r: angdiff_deg(float(r["pred_ang_calib"]), float(r["gt_ang"])),
        axis=1
    )

    # ---------- SAVE PARAMS ----------
    feat_names = ["bias","g","g2","b","b2","v","g*b","reproj"]
    rows = []
    for cam in ["rear","left"]:
        lam_xy, lam_th = chosen[cam]
        wpar, wper, wth = weights[cam]
        row = {
            "cam": cam,
            "lambda_xy": lam_xy,
            "lambda_theta": lam_th,
            "feature_def": "X=[1,g,g^2,abs(b),abs(b)^2,v,g*abs(b),reproj], fit e_par/e_perp in ray frame + dtheta",
            "target": "rear->GT, left->(1-a)*GT+a*rear_phys (pos+theta)"
        }
        for i, fn in enumerate(feat_names):
            row[f"wpar_{fn}"] = float(wpar[i])
            row[f"wper_{fn}"] = float(wper[i])
            row[f"wtheta_{fn}"] = float(wth[i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_params, index=False)

    # save outputs
    df.to_csv(out_csv, index=False)
    draw_3panel(df, out_png)

    print("[DONE] wrote:", out_csv)
    print("[DONE] wrote:", out_png)
    print("[DONE] wrote:", out_params)
    print("Chosen lambdas:", {k:  {"xy": v[0], "theta": v[1]} for k,v in chosen.items()})
    print("v3.1: ray-frame xy + theta calibration (rear anchors left when rear reliable)")


if __name__ == "__main__":
    main()