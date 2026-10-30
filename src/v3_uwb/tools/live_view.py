"""실시간 보기 — 연결된 UWB 보드를 전부 읽어 브라우저에 보드별 카드로 띄운다.

  python3 tools/live_view.py            # http://localhost:8765 가 자동으로 열린다
  python3 tools/live_view.py --no-browser --http 8800

보이는 것 (보드마다):
  태그   현재 거리 · 최근 평균±표준편차 · 성공률 · 측정 속도 · RX−FP · 최근 30초 그래프
  앵커   초당 응답 수
  그 외  공장 테스트 펌웨어 등 — 원시 로그만

새로 꽂은 보드는 2초 안에 자동으로 카드가 생긴다. 포트를 열 때 보드가 재부팅된다(부팅 메시지로 펌웨어 판별).

기록: 카드의 '기록' → experiments/<진행 중인 실험>/data/ 에 CSV (위치는 lib/uwb_proto.py의 RECORD_DIR)

포트는 한 프로그램만 잡을 수 있다:
  - Arduino 시리얼 모니터와 동시에 열면 충돌한다
  - 펌웨어 업로드 전에는 카드의 '포트 놓기'를 누르고, 끝나면 '다시 잡기'

127.0.0.1에만 열린다 (같은 와이파이의 다른 기기에서는 안 보임).
"""
import argparse
import csv
import json
import queue
import termios
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import serial

from lib.uwb_proto import CSV_FIELDS, V3_DIR, find_ports, levels, open_and_reset, parse_line, record_csv_path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "lib" / "live_view.html"
SCAN_INTERVAL = 2.0
STALL_S = 3.0   # 태그는 10 Hz로 쉬지 않고 출력하므로 이만큼 조용하면 USB 쪽이 멈춘 것


class Stalled(Exception):
    """포트는 열려 있는데 데이터가 안 옴. 2026-10-29 맥 기본 CP2104 드라이버(AppleUSBSLCOM)가
    오래 쓰면 이렇게 멈췄다 — 그 뒤 다시 열면 'Invalid argument'. 보드를 다시 꽂으면 풀린다."""


PORT_ERRORS = (serial.SerialException, termios.error, OSError, Stalled)

# 부팅 메시지 → 펌웨어 종류
FIRMWARE_MARKS = [("# uwb_tag", "tag"), ("# uwb_anchor", "anchor"),
                  ("SIMPLE TX", "factory"), ("SIMPLE RX", "factory"),
                  ("Range TX", "factory"), ("Range RX", "factory")]


class Hub:
    """브라우저 연결(SSE)마다 큐 하나. 보드 스레드가 이벤트를 뿌린다."""

    def __init__(self):
        self.boards = {}            # port → Board
        self.subs = []
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def publish(self, ev):
        with self.lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass        # 느린 탭은 버린다 (새로고침하면 다시 받음)

    def snapshot(self):
        with self.lock:
            boards = list(self.boards.values())
        return {"type": "snapshot", "boards": [b.state() for b in boards]}


class Recorder:
    """카드의 '기록' — 진행 중인 실험 폴더의 data/에 CSV를 남긴다 (analyze.py가 읽는 형식)."""

    def __init__(self, true_m, cond, target):
        self.true_m, self.cond, self.target = true_m, cond, target
        self.path = record_csv_path(cond, true_m)
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self.w.writeheader()
        self.n_ok = self.n_all = 0

    def write(self, row):
        self.w.writerow(dict(row, t_pc=f"{time.time():.3f}",
                             true_m="" if self.true_m is None else self.true_m, cond=self.cond))
        self.f.flush()
        self.n_all += 1
        self.n_ok += row["ok"]

    @property
    def done(self):
        return self.n_ok >= self.target

    def close(self):
        self.f.close()

    def state(self):
        return {"path": str(self.path.relative_to(V3_DIR)), "n_ok": self.n_ok,
                "n_all": self.n_all, "target": self.target,
                "true_m": self.true_m, "cond": self.cond}


class Board(threading.Thread):
    """보드 하나 = 스레드 하나. 포트를 열고 줄 단위로 읽어 이벤트로 바꾼다. 끊기면 다시 시도."""

    def __init__(self, hub, port, sn):
        super().__init__(daemon=True)
        self.hub, self.port, self.sn = hub, port, sn
        self.label = sn[-4:] if sn and sn != "?" else port.rsplit("-", 1)[-1][-4:]
        self.firmware = "unknown"
        self.status = "connecting"   # connecting | live | error | released | unplugged
        self.error = ""
        self.held = True             # False면 포트를 놓는다 (업로드용)
        self.log = deque(maxlen=40)  # 측정 줄(R/E)이 아닌 것만
        self.recorder = None
        self.lock = threading.Lock()

    def state(self):
        return {"type": "board", "port": self.port, "label": self.label, "sn": self.sn,
                "firmware": self.firmware, "status": self.status, "error": self.error,
                "log": list(self.log),
                "record": self.recorder.state() if self.recorder else None}

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.hub.publish(self.state())

    # ---- 기록 ----
    def start_record(self, true_m, cond, target):
        with self.lock:
            if self.recorder:
                self.recorder.close()
            self.recorder = Recorder(true_m, cond, target)
        self.hub.publish(self.state())

    def stop_record(self, reason="stopped"):
        with self.lock:
            rec, self.recorder = self.recorder, None
        if rec:
            rec.close()
            self.hub.publish({"type": "record_done", "port": self.port, "reason": reason, **rec.state()})
            self.hub.publish(self.state())

    # ---- 읽기 ----
    def run(self):
        while True:
            if not self.held:
                if self.status != "released":
                    self._set(status="released", error="")
                time.sleep(0.2)
                continue
            if self.port not in {d for d, _ in find_ports()}:
                if self.status != "unplugged":
                    self.stop_record("unplugged")
                    self._set(status="unplugged", error="")
                time.sleep(SCAN_INTERVAL)
                continue
            try:
                if self.status != "error":   # 재시도마다 '연결 중'↔'오류'로 깜빡이지 않게
                    self._set(status="connecting", error="")
                s = open_and_reset(self.port)
            except PORT_ERRORS as e:
                self._set(status="error", error=str(e))
                time.sleep(3)
                continue
            self._set(status="live", firmware="unknown")
            try:
                self._read_loop(s)
            except PORT_ERRORS as e:
                self.stop_record("error")
                self._set(status="error", error=str(e))
                time.sleep(1)
            finally:
                try:
                    s.close()
                except PORT_ERRORS:
                    pass

    def _read_loop(self, s):
        pending = b""
        last_data = time.time()
        while self.held:
            chunk = s.readline()
            now = time.time()
            if chunk:
                last_data = now
            elif self.firmware == "tag" and now - last_data > STALL_S:
                # 앵커는 태그가 없으면 원래 조용하므로 태그만 본다
                raise Stalled(f"{STALL_S:.0f}초 넘게 데이터가 안 옴")
            if not chunk.endswith(b"\n"):
                pending += chunk
                continue
            line = (pending + chunk).decode("utf8", "replace").strip()
            pending = b""
            if line:
                self._handle(line)
        self.stop_record("released")

    def _handle(self, line):
        row = parse_line(line)
        if row is None:
            if line.startswith("# replies="):
                try:
                    n = int(line.split("=")[1].split()[0])
                    self.hub.publish({"type": "anchor", "port": self.port, "replies": n, "t": time.time()})
                except (IndexError, ValueError):
                    pass
            self.log.append(line)
            fw = next((fw for mark, fw in FIRMWARE_MARKS if line.startswith(mark)), None)
            if fw and fw != self.firmware:
                self._set(firmware=fw)          # 카드 모양이 바뀌므로 상태 전체를 보냄
            else:
                self.hub.publish({"type": "log", "port": self.port, "line": line})
            return

        if self.firmware != "tag":   # 부팅 메시지를 놓쳤어도 측정 줄이 오면 태그다
            self.firmware = "tag"
            self.hub.publish(self.state())
        rx, fp, dfp = levels(row) if row["ok"] else (None, None, None)
        self.hub.publish({"type": "sample", "port": self.port, "t": time.time(),
                          "ok": row["ok"], "seq": row["seq"], "dist_m": row.get("dist_m"),
                          "clk_ppm": row.get("clk_ppm"), "reason": row.get("reason", ""),
                          "rx": rx, "fp": fp, "dfp": dfp})
        with self.lock:
            rec = self.recorder
            if rec:
                rec.write(row)
        if rec:
            if rec.n_all % 10 == 0:
                self.hub.publish({"type": "record", "port": self.port, **rec.state()})
            if rec.done:
                self.stop_record("done")


def scanner(hub):
    """새로 꽂힌 보드를 찾아 스레드를 붙인다. 뽑힌 보드는 그 보드 스레드가 알아서 'unplugged'로 바꾼다."""
    while True:
        for dev, sn in find_ports():
            with hub.lock:
                known = dev in hub.boards
            if not known:
                b = Board(hub, dev, sn)
                with hub.lock:
                    hub.boards[dev] = b
                b.start()
                hub.publish(b.state())
        time.sleep(SCAN_INTERVAL)


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # 요청마다 찍히는 로그 끔
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.read_bytes()   # 매번 읽음 → html 고치고 새로고침만 하면 됨
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self._sse()
            else:
                self._json(404, {"error": "not found"})

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = hub.subscribe()
            try:
                self._send(hub.snapshot())
                while True:
                    try:
                        self._send(q.get(timeout=10))
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")   # 연결 유지
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

        def _send(self, ev):
            self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        def do_POST(self):
            # 다른 사이트가 이 로컬 서버로 요청을 못 보내게: 커스텀 헤더는 교차 출처면 사전 확인(preflight)이
            # 필요한데 여기서는 응답하지 않으므로 브라우저가 막는다. 우리 페이지(같은 출처)만 통과
            if self.headers.get("X-UWB") != "1":
                return self._json(403, {"error": "forbidden"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            with hub.lock:
                b = hub.boards.get(req.get("port"))
            if b is None:
                return self._json(404, {"error": "unknown port"})

            if self.path == "/api/release":
                b.held = False
            elif self.path == "/api/attach":
                b.held = True
            elif self.path == "/api/record/start":
                if b.firmware != "tag" or b.status != "live":
                    return self._json(409, {"error": "태그 펌웨어가 돌고 있는 보드만 기록할 수 있음"})
                try:
                    true_m = None if req.get("true_m") in (None, "") else float(req["true_m"])
                    target = max(1, int(req.get("target", 300)))
                except (TypeError, ValueError):
                    return self._json(400, {"error": "실제 거리·목표 개수는 숫자"})
                cond = "".join(c for c in str(req.get("cond") or "los") if c.isalnum() or c in "_-") or "los"
                b.start_record(true_m, cond, target)
            elif self.path == "/api/record/stop":
                b.stop_record("stopped")
            else:
                return self._json(404, {"error": "not found"})
            self._json(200, b.state())

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=8765, help="웹 서버 포트 (기본 8765)")
    ap.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 끔")
    args = ap.parse_args()

    hub = Hub()
    threading.Thread(target=scanner, args=(hub,), daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.http), make_handler(hub))
    server.daemon_threads = True
    url = f"http://localhost:{args.http}"
    print(f"실시간 보기: {url}   (끝내려면 Ctrl+C)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
        with hub.lock:
            boards = list(hub.boards.values())
        for b in boards:
            b.stop_record("server stopped")


if __name__ == "__main__":
    main()
