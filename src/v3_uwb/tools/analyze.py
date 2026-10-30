"""거리 측정 분석 — live_view.py로 기록한 CSV를 (조건, 실제 거리)별로 묶어 정확도 표를 만든다.

  python3 tools/analyze.py                        # 진행 중인 실험 폴더의 data/*.csv 전부
  python3 tools/analyze.py experiments/01_거리정확도/data/2026*_los_*.csv

보고서 항목:
  성공률    시도 대비 측정 성공 (차체에 가리면 떨어질 것)
  편향      평균 − 실제. 거리와 무관하게 일정하면 안테나 지연 문제 → 보정 가능
  표준편차  측정 흩어짐. 정지 상태에서 재야 의미 있음
  느린변화  50개(5초) 구간 평균들의 최대−최소. 크면 흩어짐이 무작위가 아니라 천천히 떠도는 것이라
            평균을 내도 잘 안 줄어든다 (주변 사람 움직임·반사 변화). 2026-10-29 0.41 m 기록에서 6 cm
  |오차|95  95%의 측정이 이 오차 안에 든다 (주행 제어에서 믿을 수 있는 폭)
  RX−FP     직진파 대비 전체 수신 세기 차이. LOS와 NLOS를 가르는 기준을 여기서 정한다

안테나 지연 추정 (LOS, 실제 거리 2개 이상일 때):
  측정 = a·실제 + b 로 맞춘다. b가 거리와 무관한 고정 오차다.
  양쪽 보드의 TX·RX 지연을 모두 δ틱씩 늘리면 거리는 2δ × 4.69 mm 줄어든다
  (1틱 = 15.65 ps, 빛이 4.69 mm 가는 시간. 지연 4개가 ToF에 절반씩 들어감)
  → δ ≈ b / 9.38 mm. 보드 2개로는 어느 보드 탓인지 못 가르므로 둘 다 같은 값을 더한다.
  보드별 값은 보드 3개로 쌍마다(A–B, B–C, A–C) 재서 푼다 (docs/원리.txt 4절, 실험노트).
  a가 1에서 1% 넘게 벗어나면 지연이 아니라 시계 오차 쪽을 의심한다.
  측정할 때 안테나 높이가 같아야 한다 — 높이차가 있으면 가까운 거리일수록 길게 나와 기울기·절편이 함께 틀어진다.
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict

from lib.uwb_proto import RECORD_DIR, levels

DATA_GLOB = str(RECORD_DIR / "*.csv")
MM_PER_TICK_ALL4 = 9.384   # 지연 4개를 모두 1틱 늘릴 때 줄어드는 거리 (mm)
BLOCK = 50                 # 느린 변화를 볼 구간 (10 Hz면 5초)


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cond"], r["true_m"])].append(r)

    out = []
    for (cond, true_m), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], float(kv[0][1] or 0))):
        ok = [r for r in rs if r["ok"] == "1"]
        d = [float(r["dist_m"]) for r in ok]
        dfp = [x for x in (levels(r)[2] for r in ok) if x is not None]
        rx = [x for x in (levels(r)[0] for r in ok) if x is not None]
        t = float(true_m) if true_m else None
        blocks = [statistics.fmean(d[i:i + BLOCK]) for i in range(0, len(d) - BLOCK + 1, BLOCK)]
        out.append({
            "cond": cond, "true": t, "n": len(rs), "ok": len(ok),
            "mean": statistics.fmean(d) if d else None,
            "sd": statistics.pstdev(d) if len(d) > 1 else None,
            "drift": max(blocks) - min(blocks) if len(blocks) >= 2 else None,
            "bias": statistics.fmean(d) - t if d and t is not None else None,
            "e95": p95([abs(x - t) for x in d]) if d and t is not None else None,
            "rx": statistics.fmean(rx) if rx else None,
            "dfp": statistics.fmean(dfp) if dfp else None,
        })
    return out


def fmt(x, spec):
    return "—" if x is None else format(x, spec)


def print_report(stats):
    """(조건, 실제 거리)마다 세 줄. 터미널에서 그대로 읽히고 실험노트(txt)에 붙여 넣을 수 있게 보통 글로."""
    cm = lambda v: None if v is None else v * 100
    for s in stats:
        rate = f"{s['ok']}/{s['n']} ({100 * s['ok'] / s['n']:.0f}%)" if s["n"] else "—"
        truth = f"실제 {s['true']:.2f} m" if s["true"] is not None else "실제 거리 없음"
        print(f"[{s['cond']}, {truth}]  성공 {rate}")
        print(f"  평균 {fmt(s['mean'], '.3f')} m   편향 {fmt(cm(s['bias']), '+.1f')} cm   "
              f"표준편차 {fmt(cm(s['sd']), '.1f')} cm   느린변화 {fmt(cm(s['drift']), '.1f')} cm   "
              f"|오차|95 {fmt(cm(s['e95']), '.1f')} cm")
        print(f"  RX {fmt(s['rx'], '.1f')} dBm   RX−FP {fmt(s['dfp'], '.1f')} dB\n")


def fit_antenna_delay(stats):
    pts = [(s["true"], s["mean"]) for s in stats
           if s["cond"] == "los" and s["true"] is not None and s["mean"] is not None]
    if len({t for t, _ in pts}) < 2:
        print("안테나 지연 추정: LOS에서 실제 거리 2개 이상 필요 (지금 %d개)" % len({t for t, _ in pts}))
        return
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    a = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    b = my - a * mx
    delta = b * 1000 / MM_PER_TICK_ALL4
    print(f"안테나 지연 추정 (LOS {len(pts)}점): 측정 = {a:.4f}·실제 {b * 100:+.1f} cm")
    if abs(a - 1) > 0.01:
        print(f"  기울기가 1에서 {abs(a - 1) * 100:.1f}% 벗어남 — 고정 오차만의 문제가 아님. 지연 보정 전에 원인부터 볼 것")
    print(f"  → 양쪽 보드 TX_ANT_DLY·RX_ANT_DLY에 각각 {delta:+.0f} 더하기 (현재 16385 → {16385 + round(delta)})")


def main():
    paths = sys.argv[1:] or sorted(glob.glob(DATA_GLOB))
    if not paths:
        sys.exit(f"CSV 없음: {DATA_GLOB}")
    stats = summarize(load(paths))
    print(f"파일 {len(paths)}개\n")
    print_report(stats)
    fit_antenna_delay(stats)


if __name__ == "__main__":
    main()
