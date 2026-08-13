"""공용 카메라 계층 — GStreamer 파이프라인과 Webcam 래퍼 "한 벌".

main.py(통합 실행)와 guidance/(teach & repeat)가 이 모듈을 같이 쓴다.
카메라 설정을 바꿀 일이 생기면 여기 한 곳만 고치면 됨.

이 보드(C920 + Jetson Nano)에서 실측으로 확정된 규칙:
- 해상도는 반드시 16:9(1280x720 등)로 요청할 것 — 4:3(640x480 등)을
  요청하면 카메라가 센서(16:9)를 좌우 크롭해서 화각이 좁아짐.
- fps 기본 15 — 30으로 받으면 소프트웨어 JPEG 디코딩이 코어 하나를
  통째로 먹음(실측 91%). 추론이 초당 십수 장이라 15면 충분.
- cv2.VideoCapture(index) 직접 사용 금지 — C920+젯슨 커널 조합에서
  무한 대기 버그가 있어 GStreamer 파이프라인 필수.
"""
import os

import cv2

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 15


def usb_pipeline(index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
    return (
        f"v4l2src device=/dev/video{index} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def csi_pipeline(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def ensure_display():
    """SSH/VSCode 원격 터미널에서도 젯슨 본체 HDMI(:0)에 창이 뜨게 함."""
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")


class Webcam:
    """GStreamer 기반 카메라. read()는 프레임(numpy 배열) 또는 None을 반환."""

    def __init__(self, index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 fps=DEFAULT_FPS, csi=False):
        if csi:
            pipeline = csi_pipeline(width, height, fps)
        else:
            pipeline = usb_pipeline(index, width, height, fps)
        self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            raise RuntimeError(
                "cannot open Jetson camera; check camera index/--csi and "
                "GStreamer: YES in cv2.getBuildInformation()")

    def read(self):
        try:
            ok, frame = self.capture.read()
        except cv2.error:
            return None
        return frame if ok else None

    def release(self):
        self.capture.release()
