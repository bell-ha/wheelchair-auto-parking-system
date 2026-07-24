"""pose_full 결과 자동 검증: 물리 한계·구역 일관성·교차 검증. 실패 항목은 FAIL로 표시."""
import csv
import math

S = "/Users/jongha/Desktop/GitHub/wheelchair-auto-parking-system/src/v2_vision/outputs/poster_compare"
full = [r for r in csv.DictReader(open(f"{S}/pose_full.csv"))]
base = [r for r in csv.DictReader(open(f"{S}/pose_log.csv"))]

ok = True

def check(name, cond, detail):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok = ok and cond

# 1. 물리 한계 (이동 1.5 m/s, 회전 90 deg/s)
# 추정치는 관측 갱신(track) 시점에만 움직이므로, 갱신 간 경과시간 기준으로 측정.
# dt는 반올림된 t가 아니라 프레임 번호로 계산 (30fps).
FPS = 30.0
sp, om = [], []
prev = None
for r in full:
    if not r["x"] or not ((r["status"].startswith("track") or r["status"].startswith("pnp")) or r["status"].startswith("init")):
        continue
    if prev is not None:
        dt = (int(r["frame"]) - int(prev["frame"])) / FPS
        if dt > 0:
            sp.append(math.hypot(float(r["x"]) - float(prev["x"]),
                                 float(r["y"]) - float(prev["y"])) / dt)
            om.append(abs((float(r["theta_deg"]) - float(prev["theta_deg"]) + 180) % 360 - 180) / dt)
    prev = r
# CSV 좌표가 1mm 단위 반올림이라 프레임당 최대 ±0.04m/s 오차 허용 (5%)
nv = sum(1 for v in sp if v > 1.5 * 1.05)
nw = sum(1 for w in om if w > 90 * 1.05)
check("이동속도 한계", nv == 0, f"최대 {max(sp):.2f} m/s, 1.5 초과 {nv}프레임")
check("회전속도 한계", nw == 0, f"최대 {max(om):.0f} deg/s, 90 초과 {nw}프레임")

# 2. 구역-위치 일관성 (track 프레임만, 구역 전환 직후 2초는 과도기로 제외)
zone_t0 = {}
last_zone = None
for r in full:
    if r["zone"] != last_zone:
        last_zone = r["zone"]
        zone_t0[id(r)] = True
        t0 = float(r["t"])
    r["_settled"] = (float(r["t"]) - t0) > 2.0

rear_bad = side_bad = rear_n = side_n = 0
for r in full:
    if not r["x"] or not (r["status"].startswith("track") or r["status"].startswith("pnp")) or not r["_settled"]:
        continue
    x, y = float(r["x"]), float(r["y"])
    if r["zone"] == "후면":
        rear_n += 1
        if y < 0.2:
            rear_bad += 1
    elif r["zone"] in ("측면", "후측면"):
        side_n += 1
        if x > -0.5:
            side_bad += 1
check("후면 구역 ⇒ 차 뒤(y>0.2)", rear_bad <= rear_n * 0.02,
      f"{rear_n}프레임 중 위반 {rear_bad}")
check("측면 구역 ⇒ 왼쪽 바깥(x<-0.5)", side_bad <= side_n * 0.02,
      f"{side_n}프레임 중 위반 {side_bad}")

# 3. 교차 검증 (물리 제약을 오차로 처벌하지 않는 정의):
#   기준 = 순수 번호판 물리모델 (D·sinψ, D·cosψ) — 실측으로 검증된 방식
#   3a. 정상상태: 긴 coast(≥0.5s) 종료 후 1.5초(따라잡기 시간)는 과도기로 제외하고 RMS
#   3b. 재획득 수렴: 각 과도기에서 2.5초 내 0.3m 이내로 수렴하는지
bmap = {r["frame"]: r for r in base if r["yaw"] and r["dist"]}

# coast 런 찾기 (≥15프레임) → 과도기 = 종료 후 45프레임
coast_runs, run = [], None
for r in full:
    if r["status"] in ("coast", "lost") or r["status"].startswith("weak"):
        run = [int(r["frame"]), int(r["frame"])] if run is None else [run[0], int(r["frame"])]
    else:
        if run and run[1] - run[0] + 1 >= 15:
            coast_runs.append(run)
        run = None
# 과도기 = coast 종료 후 "수렴(0.3m 이내) 시점"까지 — 3b(2.5s 허용)와 일관된 정의.
# 수렴 시점 계산은 gaps 수집 후에 하므로, 우선 최대 창(75프레임=2.5s)으로 표시해 두고
# 아래에서 첫 수렴 프레임 이후는 과도기에서 제외한다.
transient = set()
for a, b in coast_runs:
    transient.update(range(b + 1, b + 76))

gaps = {}   # frame -> (gap, in_transient)
for r in full:
    if not r["x"] or not (r["status"].startswith("track") or r["status"].startswith("pnp")) or int(r["license_plate"])==0:
        continue
    b = bmap.get(r["frame"])
    if not b:
        continue
    th = math.radians(float(b["yaw"]))
    D = float(b["dist"])
    g = math.hypot(float(r["x"]) - D * math.sin(th), float(r["y"]) - D * math.cos(th))
    gaps[int(r["frame"])] = (g, int(r["frame"]) in transient)

# 각 coast 후 첫 수렴(<0.3m) 이후 프레임은 정상상태로 복귀시킴
for a, b in coast_runs:
    seq = [(f, gaps[f][0]) for f in sorted(gaps) if b < f <= b + 75]
    conv = next((f for f, g in seq if g < 0.3), None)
    if conv is not None:
        for f, _g in seq:
            if f > conv:
                gaps[f] = (gaps[f][0], False)
steady = sorted(g for f, (g, tr) in gaps.items() if not tr)
rms = math.sqrt(sum(g * g for g in steady) / len(steady))
check("정상상태 정합 (vs 번호판 물리모델)", rms < 0.3,
      f"{len(steady)}프레임, RMS {rms:.2f}m, 중앙값 {steady[len(steady)//2]:.2f}m")

conv_fail = []
for a, b in coast_runs:
    seq = [(f, g) for f, (g, _) in sorted(gaps.items()) if b < f <= b + 75]
    if not seq:
        continue                      # 재획득 후 번호판 미노출 → 판정 불가
    ok_f = next((f for f, g in seq if g < 0.3), None)
    if ok_f is None or (ok_f - b) / FPS > 2.5:
        conv_fail.append(b)
check("재획득 수렴 (2.5s 내 0.3m)", not conv_fail,
      f"긴 coast {len(coast_runs)}회 중 수렴 실패 {len(conv_fail)}회 {conv_fail[:3]}")

# 3.5 실측 정답 검증: 추정 카메라 높이 vs 실측 1.10m
zs = sorted(float(r["cam_z"]) for r in full
            if r.get("cam_z") and (r["status"].startswith("track") or r["status"].startswith("pnp")))
if zs:
    med_z = zs[len(zs) // 2]
    check("카메라 높이 (실측 1.10m)", abs(med_z - 1.10) < 0.15,
          f"추정 중앙값 {med_z:.2f}m ({len(zs)}프레임)")

# 4. 커버리지 — 차가 실제로 보이는 프레임 기준 (영상의 ~32%는 차가 화면에 없음)
CLS = ["car_emblem", "door_handle", "fuel_cap", "license_plate", "side_mirror", "tail_light"]
visible = [r for r in full if sum(int(r[c]) for c in CLS) > 0]
tracked = sum(1 for r in visible if (r["status"].startswith("track") or r["status"].startswith("pnp")))
check("추정 커버리지 (차가 보이는 프레임 중)", tracked / len(visible) > 0.65,
      f"{tracked}/{len(visible)} ({tracked/len(visible)*100:.0f}%) | 전체 대비 {tracked/len(full)*100:.0f}%")

print("\n종합:", "전부 통과" if ok else "실패 항목 있음 — 수정 필요")
