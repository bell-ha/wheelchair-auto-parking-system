from pynput import keyboard
import paho.mqtt.client as mqtt

# MQTT 설정
mqtt_broker = "broker.emqx.io"  # 브로커 IP 주소 입력
mqtt_port = 1883
mqtt_topic = "wheelchair/command"
mqtt_username = "jiwookmqtt"
mqtt_password = "public"

# MQTT 클라이언트 설정
client = mqtt.Client()

# MQTT 연결 콜백
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT 연결 성공!")
    else:
        print(f"MQTT 연결 실패! 코드: {rc}")

# MQTT 설정 및 연결
client.on_connect = on_connect
client.connect(mqtt_broker, mqtt_port, 60)
client.loop_start()  # MQTT 이벤트 루프 시작

# 방향키에 따른 각도 매핑
direction_map = {
    'up': 0,          # 위 방향키
    'right': 90,      # 오른쪽 방향키
    'down': 180,      # 아래 방향키
    'left': 270,      # 왼쪽 방향키
}

current_direction = 0  # 기본값 (정면)
pressed_keys = set()  # 현재 눌린 키를 저장

# 키보드 입력 처리 콜백 함수
def on_press(key):
    global current_direction
    try:
        # 방향키 입력 감지
        if key == keyboard.Key.up:
            pressed_keys.add('up')
        elif key == keyboard.Key.right:
            pressed_keys.add('right')
        elif key == keyboard.Key.down:
            pressed_keys.add('down')
        elif key == keyboard.Key.left:
            pressed_keys.add('left')
        elif hasattr(key, 'char') and key.char == 's':  # 's' 키를 눌렀을 때
            # 방향키만 제거하고 나머지는 그대로 둡니다.
            pressed_keys.discard('up')
            pressed_keys.discard('right')
            pressed_keys.discard('down')
            pressed_keys.discard('left')
            client.publish(mqtt_topic, "stop")  # "stop" 명령 전송
            print("Stop 명령 전송! 방향키 초기화됨.")

        # 현재 눌린 키 조합에 따른 방향 계산
        direction = calculate_direction(pressed_keys)
        if direction is not None and direction != current_direction:
            current_direction = direction
            # 현재 방향을 MQTT로 전송
            client.publish(mqtt_topic, str(current_direction))
            print(f"Direction: {current_direction}°")

    except AttributeError:
        pass  # 특수 키 처리 (예: Shift, Ctrl 등)

def on_release(key):
    try:
        # 키에서 손을 뗄 때 키 제거
        if key == keyboard.Key.up:
            pressed_keys.discard('up')
        elif key == keyboard.Key.right:
            pressed_keys.discard('right')
        elif key == keyboard.Key.down:
            pressed_keys.discard('down')
        elif key == keyboard.Key.left:
            pressed_keys.discard('left')

        # 방향 업데이트
        direction = calculate_direction(pressed_keys)
        if direction is not None and direction != current_direction:
            client.publish(mqtt_topic, str(direction))
            print(f"Direction: {direction}°")

    except AttributeError:
        pass

# 눌린 키 조합에 따른 방향 계산
def calculate_direction(keys):
    if 'up' in keys:
        return direction_map['up']
    elif 'right' in keys:
        return direction_map['right']
    elif 'down' in keys:
        return direction_map['down']
    elif 'left' in keys:
        return direction_map['left']
    else:
        return None  # 키가 눌리지 않았을 경우

# 키보드 리스너 시작
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# MQTT 이벤트 루프 종료
client.loop_stop()
client.disconnect()
