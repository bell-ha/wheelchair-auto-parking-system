"""전 구간 연속 위치추정: 보이는 부위 전부를 랜드마크로 쓰는 후방교회법 + 시간 연속성.

- 각 검출 부위의 화면 가로 중심 → bearing(시선축 기준 방향각)
- 실물 크기 포스터라 부위 실좌표(차 좌표계)를 앎 → 3개 이상이면 (x, y, θ) 유일해
- 번호판이 보이면 핀홀 거리(155mm)로 절대 스케일 보강
- 직전 포즈를 초기값 + 약한 사전확률로 → 프레임 간 흐름(연속성)
- 결과: CSV + 궤적 PNG + 미니맵 영상(전 구간 위치 점)
좌표계: 번호판 원점, X+ = 후면 마주보고 오른쪽(=차의 오른쪽), Y+ = 차 뒤쪽. (distance_video와 동일)
"""
import csv
import json
import math
import sys

import cv2
import numpy as np

SCRIPTS = "/Users/jongha/Desktop/GitHub/wheelchair-auto-parking-system/src/v2_vision/scripts"
V = "/Users/jongha/Desktop/GitHub/wheelchair-auto-parking-system/src/v2_vision"
OUT = "/Users/jongha/Desktop/GitHub/wheelchair-auto-parking-system/src/v2_vision/outputs/poster_compare"
sys.path.insert(0, SCRIPTS)
import distance_video as dv  # noqa: E402
# 포스터의 번호판은 장형 인쇄 (정면 프레임 실측 비율 4.82 ≈ 520/110) — 단형(335x155) 아님
dv.PLATE_W_M = 0.520
dv.PLATE_H_M = 0.110
dv.TRUE_AR = dv.PLATE_W_M / dv.PLATE_H_M
from ultralytics import YOLO  # noqa: E402

# ---- 레이 부위 실좌표 (m, 도면 기반 추정. 전장 3.595 / 전폭 1.595 / 축거 2.52) ----
# 왼쪽면(운전석) 부위만 — 이번 시나리오(왼쪽→후면→왼쪽)에는 오른쪽면 미등장
LANDMARKS = {
    "plate":        (0.00, 0.00),
    "emblem_rear":  (0.00, 0.00),
    "tail_left":    (-0.61, 0.00),
    "tail_right":   (0.61, 0.00),
    "fuel_cap":     (-0.80, -0.75),
    "handle_rear":  (-0.80, -1.35),
    "handle_front": (-0.80, -2.15),
    "mirror_left":  (-0.95, -2.55),
}
CLASS_CANDS = {
    "license_plate": ["plate"],
    "car_emblem": ["emblem_rear"],
    "tail_light": ["tail_left", "tail_right"],
    "fuel_cap": ["fuel_cap"],
    "door_handle": ["handle_rear", "handle_front"],
    "side_mirror": ["mirror_left"],
}
GATE_DEG = 25          # 예측 bearing과 이보다 크게 어긋나면 매칭 거부
PRIOR_W = 0.5          # 직전 포즈 사전확률 (물리 제한이 연속성을 담당하므로 약하게)
MIN_OBS = 2            # 랜드마크 2개부터 갱신 (2개일 땐 사전확률을 강화해 드리프트 억제)
POS_W = 2.0            # 번호판 yaw 기반 절대 위치 관측 (검증된 신호 — 강하게)
BAR_W = 3.0            # 가시성 반공간 제약 가중 (포스터 평면은 앞에서만 보임)
VMAX = 1.5             # 휠체어 최대 이동속도 (m/s) — 프레임당 이동량 상한
WMAX_DEG = 90.0        # 휠체어 최대 회전속도 (deg/s)


def clamp_motion(prev, new, dt):
    """물리 한계(VMAX, WMAX)로 프레임당 포즈 변화량 제한 — 순간이동 방지."""
    dx, dy = new[0] - prev[0], new[1] - prev[1]
    d = math.hypot(dx, dy)
    maxd = VMAX * dt
    if d > maxd:
        dx, dy = dx / d * maxd, dy / d * maxd
    dth = wrap(new[2] - prev[2])
    maxth = math.radians(WMAX_DEG) * dt
    dth = max(-maxth, min(maxth, dth))
    return np.array([prev[0] + dx, prev[1] + dy, prev[2] + dth])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def solve_pose(obs, dist_obs, pos_obs, rear_seen, side_seen, init, prior_w=PRIOR_W, iters=15):
    """obs: [(lx, ly, bearing_rad)] / dist_obs: (lx, ly, D) / pos_obs: 번호판 yaw 기반 (x0, y0).
    rear_seen: 뒷면 부위 관측 → 카메라 y > 0.3 강제 / side_seen: 왼쪽면 부위 관측 → x < -0.9 강제.
    비선형 최소제곱 (Gauss-Newton, 직전 포즈 사전확률 포함)."""
    s = np.array(init, dtype=float)  # [cx, cy, theta]
    prior = np.array(init, dtype=float)
    for _ in range(iters):
        J, r = [], []
        for lx, ly, b in obs:
            dx, dy = lx - s[0], ly - s[1]
            q = dx * dx + dy * dy
            if q < 1e-6:
                continue
            phi = math.atan2(dy, dx)
            r.append(wrap(phi - s[2] - b))
            J.append([dy / q, -dx / q, -1.0])
        if dist_obs is not None:
            lx, ly, D = dist_obs
            dx, dy = s[0] - lx, s[1] - ly
            d = math.hypot(dx, dy)
            if d > 1e-6:
                r.append((d - D) * 2.0)          # 거리 잔차 (가중 2: rad와 스케일 맞춤)
                J.append([dx / d * 2.0, dy / d * 2.0, 0.0])
        if pos_obs is not None:                  # 번호판 yaw 기반 절대 위치 (collinear 퇴화 해소)
            r.append((s[0] - pos_obs[0]) * POS_W)
            J.append([POS_W, 0.0, 0.0])
            r.append((s[1] - pos_obs[1]) * POS_W)
            J.append([0.0, POS_W, 0.0])
        if rear_seen and s[1] < 0.3:             # 뒷면이 보이면 반드시 차 뒤쪽에
            r.append((0.3 - s[1]) * BAR_W)
            J.append([0.0, -BAR_W, 0.0])
        if side_seen and s[0] > -0.9:            # 왼쪽면이 보이면 반드시 왼쪽 바깥에
            r.append((s[0] + 0.9) * BAR_W)
            J.append([BAR_W, 0.0, 0.0])
        # 사전확률 (직전 포즈에서 크게 벗어나지 않게)
        for k, wgt in ((0, prior_w), (1, prior_w), (2, prior_w * 0.5)):
            rr = (s[k] - prior[k]) * wgt
            if k == 2:
                rr = wrap(s[k] - prior[k]) * wgt
            r.append(rr)
            row = [0.0, 0.0, 0.0]
            row[k] = wgt
            J.append(row)
        J, r = np.array(J), np.array(r)
        try:
            step = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        s += np.clip(step, -0.5, 0.5)
        if np.linalg.norm(step) < 1e-4:
            break
    resid = float(np.sqrt(np.mean(np.square(r[: len(obs)])))) if len(obs) else None
    return s, resid


model = YOLO(f"{V}/models/best_v5_poster.pt")
names = model.names
cap = cv2.VideoCapture(f"{V}/raw_videos/포스터레이.MOV")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(3))
h = int(cap.get(4))
focal_px, _ = dv.load_focal_px(w)

out_video = f"{V}/outputs/distance/포스터레이_pose_full.mp4"
writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

pose = None          # [cx, cy, theta]
last_track_t = None
last_strong_t = None # 마지막 "강한" 관측(번호판 or 랜드마크 3개 이상) 시각 — 이동 예산 기준  # 마지막 관측 갱신 시각 (물리 제한은 이 시점 기준 경과시간으로)
ema_dist = ema_yaw = None
zone_state = {"zone": "미확인", "cand": None, "n": 0}
empty_run = 0
rows = []
frame_idx = 0
edge = 5
trail = []           # 미니맵 궤적 꼬리

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    res = model.predict(frame, conf=0.15, iou=0.5, verbose=False)[0]

    counts = dict.fromkeys(names.values(), 0)
    dets = []            # (class_name, cx_px, box_h)
    plate_best = None
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cn = names[int(b.cls)]
        counts[cn] += 1
        dets.append((cn, (x1 + x2) / 2, y2 - y1))
        if cn == "license_plate" and not (x1 < edge or y1 < edge or x2 > w - edge or y2 > h - edge):
            if plate_best is None or (y2 - y1) > plate_best[4]:
                plate_best = (x1, y1, x2 - x1, (x1 + x2) / 2, y2 - y1)

    # 번호판 절대 거리 + yaw 기반 절대 위치 (기본 방식과 동일 계산, EMA)
    dist_obs = pos_obs = None
    if plate_best is not None:
        px1, py1, pbw, _pcx, pbh = plate_best
        d, ar_yaw = dv.estimate(pbw, pbh, focal_px)
        ema_dist = d if ema_dist is None else 0.3 * d + 0.7 * ema_dist
        dist_obs = (0.0, 0.0, ema_dist)
        quad = dv.find_plate_quad(frame, px1, py1, pbw, pbh)
        s_yaw = dv.plate_yaw(quad, ar_yaw) if quad is not None else None
        if s_yaw is not None:
            ema_yaw = s_yaw if ema_yaw is None else 0.3 * s_yaw + 0.7 * ema_yaw
        if ema_yaw is not None:
            th = math.radians(ema_yaw)
            pos_obs = (ema_dist * math.sin(th), ema_dist * math.cos(th))
    # 가시성 증거는 "그 면에서만 보이는" 부위로 한정:
    # - 후미등: 측면 패널에도 옆모습 인쇄 → 후면 증거 아님
    # - 사이드미러: 차체 밖으로 돌출해 뒤에서도 보임 → 측면 증거 아님 (랜드마크로는 사용)
    rear_seen = counts["license_plate"] + counts["car_emblem"] > 0
    side_seen = counts["door_handle"] + counts["fuel_cap"] > 0

    # 관측 → 랜드마크 매칭: 클래스별 전역 최적 그리디 (오차 작은 쌍부터 배정)
    # 관측 공백이 길수록 예측이 낡으므로 게이트를 넓힘 (최대 45도)
    t_now = frame_idx / fps
    coast_sec = 0.0 if last_track_t is None else t_now - last_track_t
    gate = math.radians(min(45.0, GATE_DEG + 10.0 * coast_sec))
    obs = []
    if pose is not None:
        pairs = []
        for di, (cn, cx_px, _bh) in enumerate(dets):
            b = math.atan((cx_px - w / 2) / focal_px)
            cands = CLASS_CANDS.get(cn, [])
            for lm in cands:
                lx, ly = LANDMARKS[lm]
                pred = wrap(math.atan2(ly - pose[1], lx - pose[0]) - pose[2])
                err = abs(wrap(pred - b))
                # 후보가 1개뿐인 클래스(번호판·엠블럼·주유구·미러)는 오매칭이 불가능하므로
                # 게이트 면제 — 게이트는 동종 다수(후미등·손잡이) 구분용
                if err < gate or (len(cands) == 1 and counts[cn] == 1):
                    pairs.append((err, di, lm, b))
        used_d, used_l = set(), set()
        for err, di, lm, b in sorted(pairs):
            if di in used_d or lm in used_l:
                continue
            used_d.add(di)
            used_l.add(lm)
            lx, ly = LANDMARKS[lm]
            obs.append((lx, ly, b))

    status = "coast"
    if pose is None:
        # 초기화: 번호판 보이면 기존 방식(yaw 없이 정면 가정), 아니면 왼쪽면 기본값
        if dist_obs is not None:
            pose = np.array([0.0, dist_obs[2], math.atan2(-dist_obs[2], 0.0)])
            status = "init_rear"
            last_track_t = last_strong_t = t_now
        elif counts["door_handle"] + counts["side_mirror"] + counts["fuel_cap"] >= 2:
            pose = np.array([-2.5, -1.4, 0.0])   # 왼쪽면 2.5m, 차를 향해 (+x 방향)
            status = "init_side"
            last_track_t = last_strong_t = t_now
    elif pos_obs is not None or len(obs) >= 3:
        # 강한 근거(번호판+yaw 또는 랜드마크 3개 이상)가 있을 때만 위치를 움직임.
        # 이동 예산 = 마지막 강한 관측부터의 경과시간 × VMAX → 공백 후엔 한 번에 따라잡되
        # 연속 추적 중엔 프레임당 5cm — 눈에 보이는 모든 이동이 물리 예산 안.
        new_pose, resid = solve_pose(obs, dist_obs, pos_obs, rear_seen, side_seen, pose, prior_w=0.5)
        dt_clamp = max(1.0 / fps, (t_now - last_strong_t) if last_strong_t is not None else 1.0 / fps)
        pose = clamp_motion(pose, new_pose, dt_clamp)
        status = f"track({len(obs)}{'+p' if pos_obs is not None else ''})"
        last_track_t = last_strong_t = t_now
    elif len(obs) >= MIN_OBS:
        # 약한 관측(랜드마크 2개, 번호판 없음): 근거 부족 — 위치 동결 (드리프트 방지)
        status = f"weak({len(obs)})"

    if sum(counts.values()) == 0:
        empty_run += 1
    else:
        empty_run = 0
    dv.update_zone(zone_state, dv.classify_zone(counts))
    if empty_run > 15:                     # 0.5초 이상 아무것도 안 보이면 정직하게 미확인
        zone_state["zone"] = "미확인"
        status = "lost"

    rows.append({
        "frame": frame_idx, "t": round(frame_idx / fps, 2), "zone": zone_state["zone"],
        "n_obs": len(obs), "status": status,
        "x": None if pose is None else round(float(pose[0]), 3),
        "y": None if pose is None else round(float(pose[1]), 3),
        "theta_deg": None if pose is None else round(math.degrees(float(pose[2])), 1),
        "dist_plate": None if ema_dist is None else round(ema_dist, 2),
        **{k: counts[k] for k in ["car_emblem", "door_handle", "fuel_cap",
                                  "license_plate", "side_mirror", "tail_light"]},
    })

    # ---- 영상: YOLO 박스 + 미니맵(전 구간 포즈) ----
    ann = res.plot()
    mp = None if (pose is None or status == "lost") else (float(pose[0]), float(pose[1]), float(pose[2]))
    dv.draw_minimap(ann, zone_state["zone"],
                    (1 if pose is not None and pose[0] > 0.3 else
                     -1 if pose is not None and pose[0] < -0.3 else None), mp)
    # 궤적 꼬리 (미니맵 위에 점으로)
    if mp is not None:
        trail.append((mp[0], mp[1]))
    size, s = dv.MINIMAP_SIZE, dv.MINIMAP_SCALE
    mx, my = w - size - 20, h - size - 20
    ox, oy = mx + size // 2, my + 120
    for tx, ty in trail[-900:]:
        u = max(mx + 3, min(mx + size - 3, int(ox + tx * s)))
        vv = max(my + 3, min(my + size - 3, int(oy + ty * s)))
        cv2.circle(ann, (u, vv), 1, (120, 220, 120), -1)
    hud = f"pose [{status}] obs={len(obs)}"
    if pose is not None:
        hud += f"  x={pose[0]:+.2f}m y={pose[1]:+.2f}m th={math.degrees(pose[2]):+.0f}deg"
    cv2.putText(ann, hud, (24, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(ann, hud, (24, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 255, 120), 2)
    writer.write(ann)
    if frame_idx % 500 == 0:
        print(frame_idx, status)

cap.release()
writer.release()

with open(f"{OUT}/pose_full.csv", "w", newline="") as fp:
    wtr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)

tracked = sum(1 for r in rows if r["status"].startswith("track"))
print(f"\n완료: {len(rows)}프레임 | 관측 기반 추정 {tracked}프레임 ({tracked/len(rows)*100:.0f}%)")

# 속도 검증 (물리 한계 준수 확인)
sp = []
prev = None
for r in rows:
    if r["x"] is None:
        continue
    if prev is not None and r["t"] > prev["t"]:
        sp.append(math.hypot(r["x"] - prev["x"], r["y"] - prev["y"]) / (r["t"] - prev["t"]))
    prev = r
if sp:
    sp.sort()
    print(f"이동속도: 중앙값 {sp[len(sp)//2]:.2f} m/s | 최대 {sp[-1]:.2f} m/s "
          f"| {VMAX}m/s 초과 {sum(1 for v in sp if v > VMAX + 0.01)}프레임")
print(f"영상: {out_video}")

# ---- 궤적 그림 (미니맵과 같은 방향: 차 앞=위) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 9))
cw, cl = dv.CAR_SIZE_M
ax.add_patch(plt.Rectangle((-cw / 2, -cl), cw, cl, fill=False, ec="k", lw=2))
ax.plot([-0.17, 0.17], [0, 0], color="gold", lw=5)
for name, (lx, ly) in LANDMARKS.items():
    ax.plot(lx, ly, "ks", ms=4)
    ax.annotate(name, (lx, ly), fontsize=7, ha="right")
pts = [(r["x"], r["y"], r["t"], r["theta_deg"]) for r in rows
       if r["x"] is not None and r["status"].startswith("track")]
xs, ys, ts, ths = zip(*pts)
sc = ax.scatter(xs, ys, c=ts, cmap="viridis", s=8)
fig.colorbar(sc, ax=ax, label="시간 (s)")
for i in range(0, len(pts), 45):
    a = math.radians(ths[i])
    ax.annotate("", xy=(xs[i] + 0.35 * math.cos(a), ys[i] + 0.35 * math.sin(a)),
                xytext=(xs[i], ys[i]), arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.set_xlabel("X (m)   ← 차의 왼쪽(운전석) | 차의 오른쪽(조수석) →")
ax.set_ylabel("Y (m)  차 뒤쪽 +")
ax.set_title("전 구간 연속 위치추정 (랜드마크 후방교회법) — 점=위치, 화살표=시선")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.invert_yaxis()
fig.savefig(f"{OUT}/trajectory_full.png", dpi=110, bbox_inches="tight")
print("궤적: trajectory_full.png")
