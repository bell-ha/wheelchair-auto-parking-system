#!/usr/bin/env python3
# analyze/analyze_only_no_calib.py
#
# INPUT : analyze/tocsv.csv
# OUTPUT: analyze/summary_*.csv + analyze/*.png
#
# 보정/캘리브레이션 절대 안 함.
# 단순히 GT와 Pred 비교해서 규칙 찾기용 분석만 함.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def angdiff_deg(a, b):
    # (a-b) in [-180,180)
    return (a - b + 180) % 360 - 180

def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    csv_path = Path("analyze/tocsv.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    out_dir = csv_path.parent  # analyze/
    df = pd.read_csv(csv_path)

    # --- normalize ---
    df["status"] = df["status"].astype(str).str.lower()

    # 숫자 컬럼 변환(문자열로 들어간 경우 대비)
    num_cols = [
        "gt_x","gt_y","gt_ang_south0",
        "pred_x_cm","pred_y_cm","pred_heading_south0_deg",
        "weight","quality","area","reproj","ground_m","bearing_deg","num_dets"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- ok만 ---
    ok = df[df["status"].eq("ok")].copy()
    ok = ok.dropna(subset=["cam","gt_x","gt_y","gt_ang_south0",
                           "pred_x_cm","pred_y_cm","pred_heading_south0_deg"])

    if ok.empty:
        raise RuntimeError("No rows with status=ok. Nothing to analyze.")

    # --- errors ---
    ok["pos_err_cm"] = np.sqrt((ok["pred_x_cm"]-ok["gt_x"])**2 + (ok["pred_y_cm"]-ok["gt_y"])**2)
    ok["heading_err_deg"] = angdiff_deg(ok["pred_heading_south0_deg"], ok["gt_ang_south0"])
    ok["abs_heading_err_deg"] = np.abs(ok["heading_err_deg"])
    ok["abs_bearing_deg"] = np.abs(ok["bearing_deg"])

    # --- bins (규칙 찾기용) ---
    ok["dist_bin_m"] = pd.cut(ok["ground_m"], bins=[0,1,2,3,4,5,10,1e9], right=True)
    ok["bearing_bin_deg"] = pd.cut(ok["abs_bearing_deg"], bins=[0,10,20,30,40,50,60,90,180], right=True)
    ok["reproj_bin_px"] = pd.cut(ok["reproj"], bins=[0,1,2,3,4,6,8,12,1e9], right=True)

    # =========================
    # 1) 요약표(보기 쉬운 것만)
    # =========================
    # status 카운트
    status_counts = df.groupby(["cam","status"]).size().reset_index(name="count")
    status_counts.to_csv(out_dir/"summary_status_counts.csv", index=False)

    # 카메라별 전체 성능
    summary_by_cam = ok.groupby("cam").agg(
        n=("file","count"),
        pos_med=("pos_err_cm","median"),
        pos_mean=("pos_err_cm","mean"),
        pos_p90=("pos_err_cm", lambda s: float(np.percentile(s,90))),
        head_abs_med=("abs_heading_err_deg","median"),
        head_abs_mean=("abs_heading_err_deg","mean"),
        qual_med=("quality","median"),
        reproj_med=("reproj","median"),
        area_med=("area","median"),
        ground_med=("ground_m","median"),
        absbear_med=("abs_bearing_deg","median"),
    ).reset_index()
    summary_by_cam.to_csv(out_dir/"summary_by_cam.csv", index=False)

    # 카메라별 + GT각도별(0,+45,-45 같은)
    summary_by_cam_heading = ok.groupby(["cam","gt_ang_south0"]).agg(
        n=("file","count"),
        pos_med=("pos_err_cm","median"),
        pos_p90=("pos_err_cm", lambda s: float(np.percentile(s,90))),
        head_abs_med=("abs_heading_err_deg","median"),
        qual_med=("quality","median"),
        reproj_med=("reproj","median"),
        ground_med=("ground_m","median"),
        absbear_med=("abs_bearing_deg","median"),
    ).reset_index()
    summary_by_cam_heading.to_csv(out_dir/"summary_by_cam_heading.csv", index=False)

    # 거리 구간별(ground_m)
    summary_by_distbin = ok.groupby(["cam","dist_bin_m"]).agg(
        n=("file","count"),
        pos_med=("pos_err_cm","median"),
        pos_p90=("pos_err_cm", lambda s: float(np.percentile(s,90))),
        head_abs_med=("abs_heading_err_deg","median"),
        reproj_med=("reproj","median"),
        qual_med=("quality","median"),
    ).reset_index()
    summary_by_distbin.to_csv(out_dir/"summary_by_cam_distbin.csv", index=False)

    # bearing 구간별
    summary_by_bearingbin = ok.groupby(["cam","bearing_bin_deg"]).agg(
        n=("file","count"),
        pos_med=("pos_err_cm","median"),
        pos_p90=("pos_err_cm", lambda s: float(np.percentile(s,90))),
        head_abs_med=("abs_heading_err_deg","median"),
        reproj_med=("reproj","median"),
        qual_med=("quality","median"),
    ).reset_index()
    summary_by_bearingbin.to_csv(out_dir/"summary_by_cam_bearingbin.csv", index=False)

    # reproj 구간별
    summary_by_reprojbin = ok.groupby(["cam","reproj_bin_px"]).agg(
        n=("file","count"),
        pos_med=("pos_err_cm","median"),
        pos_p90=("pos_err_cm", lambda s: float(np.percentile(s,90))),
        head_abs_med=("abs_heading_err_deg","median"),
        qual_med=("quality","median"),
    ).reset_index()
    summary_by_reprojbin.to_csv(out_dir/"summary_by_cam_reprojbin.csv", index=False)

    # =========================
    # 2) 그래프(규칙 찾기용 핵심 6개)
    # =========================
    # (A) pos_err vs ground_m
    for cam, sub in ok.groupby("cam"):
        plt.figure(figsize=(6,4))
        plt.scatter(sub["ground_m"], sub["pos_err_cm"], s=10, alpha=0.6)
        plt.xlabel("ground_m (m)")
        plt.ylabel("pos_err (cm)")
        plt.title(f"{cam}: pos_err vs distance")
        save_fig(out_dir/f"{cam}_poserr_vs_ground.png")

    # (B) pos_err vs |bearing|
    for cam, sub in ok.groupby("cam"):
        plt.figure(figsize=(6,4))
        plt.scatter(sub["abs_bearing_deg"], sub["pos_err_cm"], s=10, alpha=0.6)
        plt.xlabel("|bearing_deg|")
        plt.ylabel("pos_err (cm)")
        plt.title(f"{cam}: pos_err vs |bearing|")
        save_fig(out_dir/f"{cam}_poserr_vs_absbearing.png")

    # (C) pos_err vs reproj
    for cam, sub in ok.groupby("cam"):
        plt.figure(figsize=(6,4))
        plt.scatter(sub["reproj"], sub["pos_err_cm"], s=10, alpha=0.6)
        plt.xlabel("reproj (px)")
        plt.ylabel("pos_err (cm)")
        plt.title(f"{cam}: pos_err vs reproj")
        save_fig(out_dir/f"{cam}_poserr_vs_reproj.png")

    # (D) heading_err histogram
    for cam, sub in ok.groupby("cam"):
        plt.figure(figsize=(6,4))
        plt.hist(sub["heading_err_deg"].dropna(), bins=31)
        plt.xlabel("heading_err (deg) [pred-gt], wrapped")
        plt.ylabel("count")
        plt.title(f"{cam}: heading error histogram")
        save_fig(out_dir/f"{cam}_heading_err_hist.png")

    # (E) pos_err histogram
    for cam, sub in ok.groupby("cam"):
        plt.figure(figsize=(6,4))
        plt.hist(sub["pos_err_cm"].dropna(), bins=25)
        plt.xlabel("pos_err (cm)")
        plt.ylabel("count")
        plt.title(f"{cam}: position error histogram")
        save_fig(out_dir/f"{cam}_poserr_hist.png")

    # (F) map heatmap (GT 위치 bin별 median pos_err)
    def heatmap(cam):
        sub = ok[ok["cam"]==cam].dropna(subset=["gt_x","gt_y","pos_err_cm"])
        if sub.empty:
            return
        xbins = np.linspace(sub["gt_x"].min(), sub["gt_x"].max(), 21)
        ybins = np.linspace(sub["gt_y"].min(), sub["gt_y"].max(), 21)

        xi = np.digitize(sub["gt_x"], xbins) - 1
        yi = np.digitize(sub["gt_y"], ybins) - 1

        grid = [[[] for _ in range(len(xbins)-1)] for __ in range(len(ybins)-1)]
        for xk, yk, e in zip(xi, yi, sub["pos_err_cm"]):
            if 0 <= xk < len(xbins)-1 and 0 <= yk < len(ybins)-1:
                grid[yk][xk].append(float(e))

        H = np.full((len(ybins)-1, len(xbins)-1), np.nan)
        for r in range(H.shape[0]):
            for c in range(H.shape[1]):
                if grid[r][c]:
                    H[r,c] = np.median(grid[r][c])

        plt.figure(figsize=(6,5))
        plt.imshow(H, origin="lower", aspect="auto")
        plt.colorbar(label="median pos_err (cm)")
        plt.xlabel("GT x bin")
        plt.ylabel("GT y bin")
        plt.title(f"{cam}: pos_err heatmap (by GT bins)")
        save_fig(out_dir/f"{cam}_poserr_heatmap.png")

    for cam in ok["cam"].dropna().unique():
        heatmap(cam)

    print("[DONE] Analysis-only outputs saved to:", out_dir)
    print("Key outputs:")
    print(" - summary_status_counts.csv")
    print(" - summary_by_cam.csv")
    print(" - summary_by_cam_heading.csv")
    print(" - summary_by_cam_distbin.csv")
    print(" - summary_by_cam_bearingbin.csv")
    print(" - summary_by_cam_reprojbin.csv")
    print(" - plots: *_poserr_vs_*.png, *_hist.png, *_heatmap.png")

if __name__ == "__main__":
    main()