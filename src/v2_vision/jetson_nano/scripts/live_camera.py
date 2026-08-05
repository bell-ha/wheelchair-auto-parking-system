"""
카메라 실시간 YOLO 추론 + 부하(FPS) 테스트 스크립트.

실행 명령어, 옵션 목록은 README.md 참고 (`--help`로도 확인 가능).
여기 아래는 젯슨 환경에서만 필요한, 코드만 봐서는 알기 어려운 이유들.

워밍업: 젯슨 첫 추론은 CUDA 커널 준비 때문에 최대 1분 정도 걸림
(model.predict를 루프 밖에서 한 번 미리 호출해 소진 — 안 하면 이 비용이
FPS 통계에 섞여 첫 30프레임 수치가 왜곡됨).

DISPLAY: SSH/VSCode 원격 터미널은 기본적으로 DISPLAY가 안 잡혀있어서
cv2.imshow가 "Can't initialize GTK backend"로 죽음 -> 젯슨 본체 HDMI
모니터(:0)가 항상 연결돼있다는 전제로 스크립트가 자동으로 채워줌
(이미 설정돼 있으면 안 건드림).

카메라: cv2.VideoCapture(index) (OpenCV 내장 V4L2 백엔드)는 C920+이
젯슨 커널 조합에서 select() 타임아웃으로 무한 대기하는 버그가 있어서
GStreamer 파이프라인으로 캡처함. 이게 동작하려면 GStreamer 지원 OpenCV가
필요한데, pip opencv-python은 미지원 빌드라 카메라 열기가 무한 대기함
-> `pip3 uninstall opencv-python`으로 지우고 JetPack 기본 OpenCV
(dist-packages)를 쓸 것. ultralytics를 pip로 재설치하면 opencv-python이
의존성으로 다시 깔리니 그때마다 다시 지울 것 (requirements.txt 참고).
appsink에 sync=false를 안 주면 첫 30프레임 정도가 파이프라인 클럭
동기화 때문에 수십 초씩 걸리는 문제도 있었음.
"""

import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO


def csi_pipeline(width=1280, height=720, fps=30):
    """젯슨 CSI 카메라용 GStreamer 파이프라인 (nvarguscamerasrc)."""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def usb_pipeline(index=0, width=1280, height=720, fps=30):
    """USB 웹캠용 GStreamer 파이프라인.

    OpenCV의 내장 V4L2 백엔드(cv2.VideoCapture(index))는 일부 카메라
    (예: C920)+젯슨 커널 조합에서 select() 타임아웃으로 무한 대기하는
    버그가 있음 -> v4l2src 기반 GStreamer 파이프라인으로 우회.
    (pip opencv-python은 GStreamer 미지원 빌드라 이 경로가 동작하려면
    JetPack 기본 제공 python3-opencv(dist-packages)가 필요함)

    appsink에 sync=false를 안 주면 GStreamer가 파이프라인 클럭에 맞춰
    프레임을 동기화하려다 첫 30프레임 정도가 수십 초씩 걸리는 문제가
    있었음 -> sync=false + max-buffers=1로 항상 최신 프레임만 즉시 반환.
    """
    return (
        f"v4l2src device=/dev/video{index} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def main():
    ap = argparse.ArgumentParser(description="카메라 실시간 YOLO 부하 테스트")
    ap.add_argument("model")
    ap.add_argument("--cam", type=int, default=0, help="USB 카메라 인덱스 (기본 0)")
    ap.add_argument("--csi", action="store_true", help="젯슨 CSI 카메라 사용")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (기본 0.25)")
    ap.add_argument("--imgsz", type=int, default=640, help="추론 해상도 (낮출수록 빠름)")
    ap.add_argument("--cam-width", type=int, default=1280,
                     help="캡처 해상도 폭. 어차피 --imgsz로 리사이즈되므로 "
                          "그보다 크게 캡처해봐야 디코딩/그리기 비용만 늘어남")
    ap.add_argument("--cam-height", type=int, default=720, help="캡처 해상도 높이")
    ap.add_argument("--no-show", action="store_true", help="창 없이 콘솔 FPS만 출력")
    args = ap.parse_args()

    if not args.no_show:
        # 젯슨 나노 본체 HDMI 모니터(X 세션 :0)가 항상 연결돼있다는 전제.
        # SSH/VSCode 원격 터미널은 DISPLAY가 기본적으로 안 잡혀있어서
        # 매번 직접 export 안 해도 되게 여기서 대신 채워줌 (이미 설정돼
        # 있으면 안 건드림).
        os.environ.setdefault("DISPLAY", ":0")
        os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

    model = YOLO(args.model)

    if args.csi:
        cap = cv2.VideoCapture(csi_pipeline(args.cam_width, args.cam_height), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(usb_pipeline(args.cam, args.cam_width, args.cam_height), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("카메라를 열 수 없음. --cam 인덱스 또는 --csi 여부 확인 "
                 "(gstreamer 지원 opencv인지 확인: cv2.getBuildInformation()에서 "
                 "GStreamer: YES 인지 볼 것 — pip opencv-python은 미지원 빌드)")

    infer_times = []   # 순수 추론 시간(ms)
    loop_times = []    # 전체 루프 시간 → 실효 FPS
    n = 0
    try:
        # 젯슨에서 첫 추론은 CUDA 커널 준비 등으로 수십 초(~50s) 걸릴 수 있음.
        # 이후 루프의 FPS 통계에 이 1회성 비용이 섞이지 않도록 미리 소진.
        print("모델 워밍업 중... (젯슨 첫 추론 특성상 최대 1분 정도 걸릴 수 있음)")
        ok, frame = cap.read()
        if ok:
            t0 = time.time()
            model.predict(frame, conf=args.conf, imgsz=args.imgsz, iou=0.5, verbose=False)
            print(f"워밍업 완료 ({time.time() - t0:.1f}s)")

        t_prev = time.time()  # 워밍업 이후로 기준점을 잡아야 루프 FPS 통계가 안 왜곡됨
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
