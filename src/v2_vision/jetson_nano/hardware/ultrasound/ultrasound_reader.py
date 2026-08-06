import serial
import json
import time

PORT = "/dev/ttyUSB0"   # 필요하면 /dev/ttyACM0 로 변경
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Serial connected:", PORT)

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        data = json.loads(line)

        us = data["us"]
        side_angle = data["side_angle"]
        side_dist = data["side_dist"]
        yaw = data["yaw"]

        print("US:", us)
        print("side_angle:", side_angle, "deg")
        print("side_dist:", side_dist, "cm")
        print("yaw:", yaw, "deg")
        print("-" * 40)

    except json.JSONDecodeError:
        print("JSON parse error:", line)

    except KeyboardInterrupt:
        break

ser.close()