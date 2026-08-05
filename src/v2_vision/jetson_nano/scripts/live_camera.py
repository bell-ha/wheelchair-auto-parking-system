"""
카메라 실시간 YOLO 추론 + 부하(FPS) 테스트 스크립트.

목적: 젯슨 나노(또는 맥)에서 카메라를 붙여 best_v4_6cls 모델이
실시간으로 돌아가는지, FPS가 얼마나 나오는지 확인.

사용법:
    # USB 웹캠 (기본: /dev/video0)
    python3 scripts/live_camera.py models/best_v4_6cls.pt
    python3 scripts/live_camera.py models/best_v4_6cls.pt --cam 1 --conf 0.15

    # 젯슨 CSI 카메라 (라즈베리파이 카메라 모듈 등)
    python3 scripts/live_camera.py models/best_v4_6cls.pt --csi

    # SSH 등 화면 없는 환경 (창 안 띄우고 콘솔에 FPS만 출력)
    python3 scripts/live_camera.py models/best_v4_6cls.pt --no-show

    # 추론 해상도 낮춰 부하 줄이기 (기본 640)
    python3 scripts/live_camera.py models/best_v4_6cls.pt --imgsz 416

종료: 화면 모드에서는 q, 콘솔 모드에서는 Ctrl+C.
종료 시 평균 FPS / 평균 추론 시간 요약 출력.

부하 확인 팁 (젯슨): 다른 터미널에서 `sudo tegrastats` 를 같이 띄워
GPU 사용률·온도·스로틀링 여부를 함께 보면 됨.
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def csi_pipeline(width=1280, height=720, fps=30):
    """젯슨 CSI 카메라용 GStreamer 파이프라인 (nvarguscamerasrc)."""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def main():
    ap = argparse.ArgumentParser(description="카메라 실시간 YOLO 부하 테스트")
    ap.add_argument("model")
    ap.add_argument("--cam", type=int, default=0, help="USB 카메라 인덱스 (기본 0)")
    ap.add_argument("--csi", action="store_true", help="젯슨 CSI 카메라 사용")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640, help="추론 해상도 (낮출수록 빠름)")
    ap.add_argument("--no-show", action="store_true", help="창 없이 콘솔 FPS만 출력")
    args = ap.parse_args()

    model = YOLO(args.model)

    if args.csi:
        cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)
        # 일부 USB 웹캠(예: C920)은 기본 포맷 협상 시 select() 타임아웃으로
        # 무한 대기함 -> MJPG로 명시 지정.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("카메라를 열 수 없음. --cam 인덱스 또는 --csi 여부 확인 "
                 "(젯슨 CSI인데 안 열리면 gstreamer 지원 opencv인지 확인)")

    if not args.csi:
        # 스트림 시작 직후 몇 프레임은 디코딩 에러가 나는 경우가 있어 워밍업으로 소진
        for _ in range(15):
            try:
                ok, _ = cap.read()
            except cv2.error:
                ok = False
            if ok:
                break

    infer_times = []   # 순수 추론 시간(ms)
    loop_times = []    # 전체 루프 시간 → 실효 FPS
    t_prev = time.time()
    n = 0
    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                ok, frame = False, None
            if not ok:
                print("프레임 읽기 실패 — 종료")
                break

            t0 = time.time()
            res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                iou=0.5, verbose=False)[0]
            infer_ms = (time.time() - t0) * 1000
            infer_times.append(infer_ms)

            now = time.time()
            loop_times.append(now - t_prev)
            t_prev = now
            n += 1

            # 최근 30프레임 이동 평균 FPS
            recent = loop_times[-30:]
            fps = len(recent) / sum(recent) if recent else 0.0

            if args.no_show:
                if n % 30 == 0:
                    print(f"[{n:5d}] {fps:5.1f} FPS | 추론 {infer_ms:6.1f} ms | "
                          f"검출 {len(res.boxes)}개")
            else:
                out = res.plot()
                cv2.putText(out, f"{fps:.1f} FPS  infer {infer_ms:.0f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imshow("live", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if infer_times:
            total = sum(loop_times)
            print(f"\n=== 요약 ===")
            print(f"프레임 수   : {len(infer_times)}")
            print(f"평균 FPS    : {len(loop_times) / total:.1f}")
            print(f"평균 추론   : {sum(infer_times) / len(infer_times):.1f} ms "
                  f"(최대 {max(infer_times):.1f} ms)")


if __name__ == "__main__":
    main()
