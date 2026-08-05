"""공용 시리얼 연결 헬퍼.

초음파 수신(UltrasoundReader)과 조이스틱 송신(JoystickController)이 각자
serial.Serial을 따로 여는 대신 이 클래스를 통해 연결을 관리한다.
같은 보드 하나가 초음파/조이스틱 역할을 다 겸하는 경우, 인스턴스 하나를
양쪽에 공유해서 넘기면 된다 (main.py 참고).
"""
import threading

import serial


class SerialLink:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self._serial = serial.Serial(port, baudrate, timeout=timeout)
        self._write_lock = threading.Lock()

    def readline(self):
        """한 줄 읽기. timeout 시 빈 문자열 반환."""
        raw = self._serial.readline()
        return raw.decode("utf-8", errors="ignore").strip()

    def write_line(self, text):
        with self._write_lock:
            self._serial.write(f"{text}\n".encode("utf-8"))

    def close(self):
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
