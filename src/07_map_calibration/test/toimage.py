#!/usr/bin/env python3
# draw_3maps_from_csv_camcolor.py
#
# INPUT : analyze/tocsv.csv
# OUTPUT: analyze/toimage.png
#
# - 3 panels by gt_ang (south0): 0 / +45 / -45
# - GT: gray dot+arrow
# - PRED: rear=green, left=blue  (dot+arrow)
# - MISS(no_marker/read_fail): red X at GT
# - Error line: gray (GT->Pred)
# - Side view (view_sym_deg >= 60): yellow ring around Pred
# - Pred dot size ~ pos_err_cm, alpha ~ distance (ground_m)

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def wrap360(d): return (d % 360 + 360) % 360
def south0_to_map_deg(south0_deg: float) -> float:
    return wrap360(south0_deg + 90.0)

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
    if radius <= 0:
        return
    overlay = dst.copy()
    cv2.circle(overlay, center, radius, bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), dst, float(1.0 - alpha), 0.0, dst)

def main():
    in_csv = Path("analyze/tocsv.csv")
    if not in_csv.exists():
        raise FileNotFoundError(f"missing: {in_csv}")

    out_png = Path("analyze/toimage.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    # numeric coercion
    num_cols = ["gt_x","gt_y","gt_ang","seq","pred_x","pred_y","pred_ang",
                "ground_m","bearing_deg","view_sym_deg","pos_err_cm"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["status"] = df["status"].astype(str).str.lower()
    df["cam"] = df["cam"].astype(str).str.lower()

    # panel setup
    MAP_W, MAP_H = 1000, 1000
    gap = 30
    top_pad = 70
    canvas_w = MAP_W * 3 + gap * 4
    canvas_h = MAP_H + top_pad + gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)

    panel_x = [gap + i * (MAP_W + gap) for i in range(3)]
    panel_y = top_pad

    buckets = [0, 45, -45]
    title = {
        0:  "GT heading = 0 (south0)",
        45: "GT heading = +45 (south0)",
        -45:"GT heading = -45 (south0)"
    }

    # fixed objects (필요시 너 맵에 맞게 수정)
    CAR_TL = (200, 180)
    CAR_BR = (400, 540)
    CAM_REAR = (301, 540)
    CAM_LEFT = (200, 270)

    # camera colors for PRED
    COL_PRED = {
        "rear": (0, 255, 0),   # green
        "left": (255, 0, 0),   # blue (BGR)
    }

    panels = []
    for _ in range(3):
        p = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        p[:] = (18, 18, 18)
        draw_grid(p, 50)
        cv2.rectangle(p, CAR_TL, CAR_BR, (80, 80, 220), 2)

        cv2.circle(p, CAM_REAR, 7, (0, 255, 255), -1)
        cv2.putText(p, "rear", (CAM_REAR[0] + 10, CAM_REAR[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(p, CAM_LEFT, 7, (0, 255, 255), -1)
        cv2.putText(p, "left", (CAM_LEFT[0] + 10, CAM_LEFT[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(p)

    # per panel stats
    processed = {0: 0, 45: 0, -45: 0}
    ok_by_cam = {0: {"rear": 0, "left": 0}, 45: {"rear": 0, "left": 0}, -45: {"rear": 0, "left": 0}}
    miss_cnt = {0: 0, 45: 0, -45: 0}

    for _, r in df.iterrows():
        gt_ang = r.get("gt_ang")
        if pd.isna(gt_ang):
            continue
        gt_ang = int(gt_ang)
        if gt_ang not in buckets:
            continue

        idx = buckets.index(gt_ang)
        processed[gt_ang] += 1

        gt_x, gt_y = r.get("gt_x"), r.get("gt_y")
        if pd.isna(gt_x) or pd.isna(gt_y):
            continue
        gt_x = float(gt_x); gt_y = float(gt_y)

        # draw GT always
        gt_map_deg = south0_to_map_deg(float(gt_ang))
        cv2.circle(panels[idx], (int(round(gt_x)), int(round(gt_y))), 3, (200, 200, 200), -1)
        draw_arrow(panels[idx], gt_x, gt_y, gt_map_deg, (200, 200, 200), length=26, thickness=1)

        status = str(r.get("status", "")).lower()
        cam = str(r.get("cam", "")).lower()
        if cam not in ("rear", "left"):
            cam = "rear"

        if status != "ok":
            miss_cnt[gt_ang] += 1
            draw_x(panels[idx], gt_x, gt_y, (0, 0, 255), size=9, thickness=2)
            continue

        ok_by_cam[gt_ang][cam] += 1

        px, py, pa = r.get("pred_x"), r.get("pred_y"), r.get("pred_ang")
        if pd.isna(px) or pd.isna(py) or pd.isna(pa):
            miss_cnt[gt_ang] += 1
            draw_x(panels[idx], gt_x, gt_y, (0, 0, 255), size=9, thickness=2)
            continue

        px = float(px); py = float(py); pa = float(pa)

        # error line
        cv2.line(panels[idx], (int(round(gt_x)), int(round(gt_y))),
                 (int(round(px)), int(round(py))), (120, 120, 120), 1)

        # visual encodings
        pos_err = r.get("pos_err_cm")
        if pd.isna(pos_err):
            pos_err = math.hypot(px - gt_x, py - gt_y)
        pos_err = float(pos_err)

        ground_m = r.get("ground_m")
        ground_m = float(ground_m) if not pd.isna(ground_m) else None

        view_sym = r.get("view_sym_deg")
        view_sym = float(view_sym) if not pd.isna(view_sym) else None

        # radius by error (3~10)
        rad = int(max(3, min(10, 3 + pos_err / 15.0)))

        # alpha by distance: near=1.0, far=0.25 (rough)
        if ground_m is None:
            alpha = 0.75
        else:
            alpha = float(max(0.25, min(1.0, 1.2 - 0.2 * ground_m)))

        col = COL_PRED.get(cam, (0, 255, 0))

        # pred point (camera-colored) with alpha
        alpha_blend_circle(panels[idx], (int(round(px)), int(round(py))), rad, col, alpha)

        # highlight side-view regardless of cam
        if view_sym is not None and view_sym >= 60.0:
            cv2.circle(panels[idx], (int(round(px)), int(round(py))), rad + 2, (0, 255, 255), 2, cv2.LINE_AA)

        # pred arrow (camera-colored)
        pred_map_deg = south0_to_map_deg(pa)
        draw_arrow(panels[idx], px, py, pred_map_deg, col, length=26, thickness=2)

    # paste panels + title/stats/legend
    for i, ang in enumerate(buckets):
        x0 = panel_x[i]
        y0 = panel_y
        canvas[y0:y0 + MAP_H, x0:x0 + MAP_W] = panels[i]

        cv2.putText(canvas, title[ang], (x0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)

        cv2.putText(canvas,
                    f"imgs={processed[ang]}  rear_ok={ok_by_cam[ang]['rear']}  left_ok={ok_by_cam[ang]['left']}  miss={miss_cnt[ang]}",
                    (x0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

        lx, ly = x0 + 10, y0 + 10
        cv2.rectangle(canvas, (lx, ly), (lx + 520, ly + 115), (10, 10, 10), -1)
        cv2.rectangle(canvas, (lx, ly), (lx + 520, ly + 115), (70, 70, 70), 1)
        cv2.putText(canvas, "GT: gray dot+arrow", (lx + 10, ly + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "PRED rear: green  |  PRED left: blue", (lx + 10, ly + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "dot size~pos_err, alpha~distance, line=error", (lx + 10, ly + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "MISS: red X  |  side-view(view_sym>=60): yellow ring", (lx + 10, ly + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_png), canvas)
    print("[DONE] wrote:", out_png)

if __name__ == "__main__":
    main()