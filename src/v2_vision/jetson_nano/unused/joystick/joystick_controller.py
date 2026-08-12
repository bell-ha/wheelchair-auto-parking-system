"""ESP32/아두이노 쪽으로 방향 명령을 보내는 모듈.

keyboard.py(수동 조종)와 main.py(자동 주행) 양쪽에서 공용으로 쓴다.
프로토콜: 각도(0/90/180/270) 문자열 또는 "stop" — 기존 keyboard.py가
쓰던 것과 동일.
"""
from serial_link import SerialLink


class JoystickController:
    def __init__(self, link: SerialLink):
        self._link = link

    def send_raw(self, text):
        self._link.write_line(text)

    def send_direction(self, angle_deg):
        self.send_raw(str(angle_deg))

    def stop(self):
        self.send_raw("stop")

    def close(self):
        self._link.close()
