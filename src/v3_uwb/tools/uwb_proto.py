"""공용 UWB 계층 — 보드 연결, firmware/uwb_tag 출력 형식, 진단값 환산 "한 벌".

check_board.py(점검)·log_range.py(기록)·analyze_range.py(분석)가 이 모듈을 같이 쓴다.
펌웨어 출력 형식이 바뀌면 uwb_tag.ino와 여기 두 곳만 맞추면 됨.

보드 연결:
  USB-UART 칩은 CP2104 (VID 0x10C4, PID 0xEA60). 포트 이름 끝이 칩 고유번호라 보드 구분에 쓴다
  (/dev/cu.usbserial-030ADFCA → "DFCA").
  RTS가 ESP32 EN(리셋)에 물려 있어서, 포트를 열 때 RTS를 한 번 올렸다 내리면 보드가 재부팅된다.
  → 부팅 메시지(# uwb_tag ...)부터 받을 수 있다.

태그 출력 (USB 시리얼, 115200bps, 한 줄씩):
  # ...                                     부팅 정보·주석 (무시)
  R,ms,seq,dist_m,clk_ppm,C,F1,F2,F3,N,D    측정 성공
  E,ms,seq,reason                           측정 실패 (timeout | rx_err | bad_frame)
  그 외 줄 (라이브러리 printf 등)             무시

진단값 → dBm (DW3000 User Manual "Estimating the receive signal power", PRF 64 MHz):
  RX  = 10·log10(C · 2^21 / N²)          + 6·D − A     전체 수신 세기
  FP  = 10·log10((F1² + F2² + F3²) / N²) + 6·D − A     첫 경로(직진파) 세기
  A = 121.7 (PRF 64 MHz, 프리앰블 코드 9)

  F1~F3는 소수부 2비트가 있어 4로 나눈다 (Qorvo 포럼 보고. 매뉴얼에는 언급 없음).
    → 2026-10-29 두 보드를 가까이(UWB로 약 0.53 m) 두고 잰 RX−FP 평균 0.9 dB. 4로 안 나누면 FP가
      12 dB 커져 −11 dB가 되는데, 첫 경로가 전체 수신보다 셀 수는 없으므로 4로 나누는 쪽이 맞다.
  D(DGC 단계)는 펌웨어가 DGC_DBG 레지스터에서 직접 읽는다 (아직 실측으로 검증 안 함).
  그래서 펌웨어는 원시값만 보내고 환산은 여기서 한다. 틀렸으면 이 파일만 고친다.

RX − FP 가 작으면 직진파가 대부분 → LOS, 크면 반사파 위주 → NLOS(차체에 가림) 가능성.
흔히 쓰는 기준은 6 dB / 10 dB지만, 위 스케일이 불확실하므로 기준은 우리 LOS·NLOS 데이터로 정한다.
"""
import math
import time
from datetime import datetime
from pathlib import Path

CP2104_VID, CP2104_PID = 0x10C4, 0xEA60
BAUD = 115200

A_PRF64 = 121.7
F_SCALE = 4.0   # F1~F3 소수부 2비트

# 측정 기록 CSV 열 순서 (log_range.py·live_view.py가 쓰고 analyze_range.py가 이 이름으로 읽는다)
CSV_FIELDS = ["t_pc", "ms", "seq", "ok", "dist_m", "clk_ppm",
              "C", "F1", "F2", "F3", "N", "D", "reason", "true_m", "cond"]
RANGE_DIR = Path(__file__).resolve().parent.parent / "data" / "range"


def range_csv_path(cond, true_m):
    """data/range/<시각>_<cond>_<true>m.csv — 폴더가 없으면 만든다."""
    RANGE_DIR.mkdir(parents=True, exist_ok=True)
    true_tag = f"{true_m:.2f}m" if true_m is not None else "free"
    return RANGE_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{cond}_{true_tag}.csv"


def find_ports():
    """연결된 UWB 보드(CP2104) 포트 목록 → [(device, 고유번호)]."""
    from serial.tools import list_ports
    return sorted((p.device, p.serial_number or "?") for p in list_ports.comports()
                  if p.vid == CP2104_VID and p.pid == CP2104_PID)


def open_and_reset(port):
    """포트를 열고 보드를 재부팅시킨다. 부팅 메시지부터 읽을 수 있는 상태로 반환."""
    import serial
    s = serial.Serial(port, BAUD, timeout=0.2)
    s.dtr = False          # IO0을 건드리지 않음 (부트로더 진입 방지)
    s.rts = True           # EN low → 리셋
    time.sleep(0.1)
    s.rts = False          # EN high → 부팅
    s.reset_input_buffer()
    return s


def parse_line(line):
    """태그 한 줄 → dict. 측정 줄이 아니면 None."""
    parts = line.strip().split(",")
    if parts[0] == "R" and len(parts) == 11:
        try:
            return {
                "ok": 1,
                "ms": int(parts[1]), "seq": int(parts[2]),
                "dist_m": float(parts[3]), "clk_ppm": float(parts[4]),
                "C": int(parts[5]), "F1": int(parts[6]), "F2": int(parts[7]),
                "F3": int(parts[8]), "N": int(parts[9]), "D": int(parts[10]),
                "reason": "",
            }
        except ValueError:
            return None   # 시리얼 노이즈로 깨진 줄
    if parts[0] == "E" and len(parts) == 4:
        try:
            return {"ok": 0, "ms": int(parts[1]), "seq": int(parts[2]), "reason": parts[3]}
        except ValueError:
            return None
    return None


def rx_dbm(C, N, D):
    if not C or not N:
        return None
    return 10 * math.log10(C * 2**21 / N**2) + 6 * D - A_PRF64


def fp_dbm(F1, F2, F3, N, D):
    f2 = sum((f / F_SCALE) ** 2 for f in (F1, F2, F3))
    if not f2 or not N:
        return None
    return 10 * math.log10(f2 / N**2) + 6 * D - A_PRF64


def levels(row):
    """R 줄 dict(또는 CSV 행) → (rx, fp, rx−fp). 값이 없으면 None."""
    try:
        C, N, D = int(row["C"]), int(row["N"]), int(row["D"])
        F1, F2, F3 = int(row["F1"]), int(row["F2"]), int(row["F3"])
    except (KeyError, ValueError, TypeError):
        return None, None, None
    rx, fp = rx_dbm(C, N, D), fp_dbm(F1, F2, F3, N, D)
    if rx is None or fp is None:
        return rx, fp, None
    return rx, fp, rx - fp
