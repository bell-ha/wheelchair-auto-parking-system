"""거리 측정 기록 — 태그(uwb_tag) 출력을 CSV로 저장하고, 1초마다 현재 통계를 보여준다.

실험 1(거리 정확도)용. 줄자로 잰 실제 거리와 조건을 라벨로 같이 남긴다.

  python3 tools/log_range.py --true 2.00 --cond los
  python3 tools/log_range.py --true 3.00 --cond nlos_car --n 300 --port /dev/cu.usbserial-030ADFBB

  --true  줄자로 잰 실제 거리 (m). 안테나 중심(흰색 칩 안테나)끼리 잰다
  --cond  조건 라벨. 예) los(가림 없음), nlos_car(차체에 가림), nlos_body(사람 몸에 가림)
  --n     성공 측정이 이 개수가 되면 종료 (기본 300 = 10 Hz로 약 30초)

포트를 열 때 태그가 재부팅된다 (부팅 메시지로 펌웨어 확인). Ctrl+C로 언제든 멈춰도 그때까지 저장된다.
저장: data/range/<시각>_<cond>_<true>m.csv (성공·실패 모두 한 줄씩)
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from uwb_proto import CSV_FIELDS, find_ports, levels, open_and_reset, parse_line, range_csv_path

HERE = Path(__file__).resolve().parent
WINDOW = 50   # 실시간 표시에 쓰는 최근 측정 개수


def pick_port(arg):
    if arg:
        return arg
    ports = find_ports()
    if len(ports) == 1:
        return ports[0][0]
    names = ", ".join(d for d, _ in ports) or "없음"
    sys.exit(f"--port로 태그를 지정하세요 (연결된 보드: {names})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", type=float, dest="true_m", help="실제 거리 (m)")
    ap.add_argument("--cond", default="los", help="조건 라벨 (기본 los)")
    ap.add_argument("--n", type=int, default=300, help="성공 측정 목표 개수")
    ap.add_argument("--secs", type=float, default=120, help="최대 기록 시간 (초)")
    ap.add_argument("--port")
    args = ap.parse_args()

    port = pick_port(args.port)
    path = range_csv_path(args.cond, args.true_m)

    s = open_and_reset(port)
    print(f"태그 {port} → {path.relative_to(HERE.parent)}")

    n_ok = n_all = 0
    recent, recent_dfp = [], []
    banner_seen = False
    pending = b""   # readline이 타임아웃으로 끊어 돌려준 줄 앞부분
    t_start = last_draw = time.time()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        try:
            while n_ok < args.n and time.time() - t_start < args.secs:
                chunk = s.readline()
                if chunk.endswith(b"\n"):
                    line, pending = (pending + chunk).decode("utf8", "replace").strip(), b""
                else:
                    pending += chunk
                    line = ""
                if line.startswith("# uwb_tag"):
                    banner_seen = True
                    print(line)
                elif line.startswith("# FATAL"):
                    sys.exit(f"태그 초기화 실패: {line}")

                row = parse_line(line) if line else None
                if row:
                    n_all += 1
                    row.update(t_pc=f"{time.time():.3f}", true_m=args.true_m if args.true_m is not None else "",
                               cond=args.cond)
                    w.writerow(row)
                    if row["ok"]:
                        n_ok += 1
                        recent = (recent + [row["dist_m"]])[-WINDOW:]
                        dfp = levels(row)[2]
                        if dfp is not None:
                            recent_dfp = (recent_dfp + [dfp])[-WINDOW:]

                now = time.time()
                if now - last_draw >= 1.0:
                    last_draw = now
                    f.flush()
                    if not banner_seen and now - t_start > 3:
                        print("! '# uwb_tag' 부팅 메시지가 안 보임 — 태그 펌웨어가 올라가 있는지 확인")
                        banner_seen = True   # 한 번만 경고
                    if recent:
                        sd = statistics.pstdev(recent) if len(recent) > 1 else 0.0
                        mean = statistics.fmean(recent)
                        err = f"  오차 {mean - args.true_m:+.3f}" if args.true_m is not None else ""
                        dfp = f"  RX−FP {statistics.fmean(recent_dfp):.1f} dB" if recent_dfp else ""
                        print(f"  {n_ok:4d}/{args.n}  성공률 {100 * n_ok / n_all:5.1f}%  "
                              f"최근{len(recent)} 평균 {mean:.3f} m ± {sd:.3f}{err}{dfp}")
                    elif n_all:
                        print(f"  성공 0 / 시도 {n_all} — 앵커 전원·거리·펌웨어 확인")
        except KeyboardInterrupt:
            print("\n중단 — 지금까지 저장함")
    s.close()
    print(f"저장: {path}  (성공 {n_ok} / 시도 {n_all})")


if __name__ == "__main__":
    main()
