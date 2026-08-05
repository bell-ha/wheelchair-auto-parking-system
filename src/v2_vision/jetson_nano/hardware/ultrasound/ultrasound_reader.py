"""ultrasound.ino가 보내는 벽 각도/거리 데이터를 백그라운드 스레드로 계속
읽어서 최신 값만 들고 있는 모듈.

프로토콜 (ultrasound.ino 쪽과 형식을 맞춰야 함):
    US,d1,d2,d3,angle_deg,center_distance_cm,perpendicular_distance_cm,distance_error_cm
    US,ERR                                                      (측정 실패)
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

from hardware.serial_link import SerialLink


@dataclass
class WallPose:
    d1: float
    d2: float
    d3: float
    angle_deg: float
    center_distance_cm: float
    perpendicular_distance_cm: float
    distance_error_cm: float
    timestamp: float


class UltrasoundReader:
    def __init__(self, link: SerialLink):
        self._link = link
        self._latest: Optional[WallPose] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_latest(self) -> Optional[WallPose]:
        """가장 최근에 파싱된 벽 포즈. 아직 못 받았으면 None."""
        with self._lock:
            return self._latest

    def _run(self):
        while self._running:
            line = self._link.readline()
            if not line:
                continue  # 타임아웃, 계속 재시도
            pose = self._parse(line)
            if pose is not None:
                with self._lock:
                    self._latest = pose

    @staticmethod
    def _parse(line: str) -> Optional[WallPose]:
        parts = line.split(",")
        if not parts or parts[0] != "US":
            return None
        if len(parts) != 8:
            return None
        try:
            d1, d2, d3, angle, center, perp, err = (float(p) for p in parts[1:])
        except ValueError:
            return None
        return WallPose(d1, d2, d3, angle, center, perp, err, time.time())
