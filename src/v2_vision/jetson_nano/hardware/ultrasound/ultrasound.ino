#include <Arduino.h>
#include <ESP32Servo.h>

// 예시용 센서값
float us[8];

float side_angle_deg = 0.0;
float side_distance_cm = 0.0;

float imu_yaw = 0.0;
float imu_pitch = 0.0;
float imu_roll = 0.0;

const float SENSOR_SPACING_TOTAL_CM = 10.5;  // D1-D3 거리
const float TARGET_DISTANCE_CM = 40.0;

// ======================================================
// Servo (Jetson의 test_sample.py 수동 조종용)
// 프로토콜: {"cmd":"servo","x":<0~180>,"y":<0~180>}\n 한 줄
// ======================================================
const int SERVO_X_PIN = 18;
const int SERVO_Y_PIN = 19;

Servo servoX;
Servo servoY;

String serialLineBuffer;

// json 문자열에서 "key":숫자 패턴을 찾아 outValue에 담는다. 못 찾으면 false.
bool extractJsonNumber(const String &json, const String &key, float &outValue) {
  String needle = "\"" + key + "\":";
  int idx = json.indexOf(needle);
  if (idx < 0) return false;

  int start = idx + needle.length();
  int end = start;
  while (end < (int)json.length() &&
         (isDigit(json[end]) || json[end] == '-' || json[end] == '.')) {
    end++;
  }
  if (end == start) return false;

  outValue = json.substring(start, end).toFloat();
  return true;
}

void handleServoCommand(const String &line) {
  if (line.indexOf("\"cmd\":\"servo\"") < 0) return;

  float x, y;

  if (extractJsonNumber(line, "x", x)) {
    servoX.write(constrain((int)(x + 0.5f), 0, 180));
  }
  if (extractJsonNumber(line, "y", y)) {
    servoY.write(constrain((int)(y + 0.5f), 0, 180));
  }
}

// Jetson -> ESP32로 들어오는 시리얼 명령을 한 줄씩 모아서 처리 (non-blocking)
void pollSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      serialLineBuffer.trim();
      if (serialLineBuffer.length() > 0) {
        handleServoCommand(serialLineBuffer);
      }
      serialLineBuffer = "";
    } else if (c != '\r') {
      serialLineBuffer += c;
    }
  }
}

void setup() {
  Serial.begin(115200);

  servoX.setPeriodHertz(50);
  servoY.setPeriodHertz(50);
  servoX.attach(SERVO_X_PIN);
  servoY.attach(SERVO_Y_PIN);

  // test_sample.py의 초기 목표각(90/90)과 맞춤
  servoX.write(90);
  servoY.write(90);
}

void loop() {
  // 0. Jetson에서 온 서보 명령 처리 (키보드 조종)
  pollSerialCommands();

  // 1. 초음파센서 8개 읽기
  // 실제로는 readUltrasonicCM() 같은 함수로 순차 측정
  for (int i = 0; i < 8; i++) {
    us[i] = readUltrasonicDummy(i);
  }

  // 2. 예: 오른쪽 측면 센서 3개가 us[0], us[1], us[2]라고 가정
  float d1 = us[0];  // 앞쪽
  float d2 = us[1];  // 중앙
  float d3 = us[2];  // 뒤쪽

  side_angle_deg = atan((d3 - d1) / SENSOR_SPACING_TOTAL_CM) * 180.0 / PI;
  side_distance_cm = d2;

  // 3. IMU 값 읽기
  // 실제로는 BNO086 라이브러리에서 yaw/pitch/roll 읽기
  imu_yaw = readYawDummy();
  imu_pitch = 0.0;
  imu_roll = 0.0;

  // 4. JSON 한 줄로 Jetson에 전송
  Serial.print("{\"us\":[");
  for (int i = 0; i < 8; i++) {
    Serial.print(us[i], 2);
    if (i < 7) Serial.print(",");
  }

  Serial.print("],\"side_angle\":");
  Serial.print(side_angle_deg, 2);

  Serial.print(",\"side_dist\":");
  Serial.print(side_distance_cm, 2);

  Serial.print(",\"yaw\":");
  Serial.print(imu_yaw, 2);

  Serial.print(",\"pitch\":");
  Serial.print(imu_pitch, 2);

  Serial.print(",\"roll\":");
  Serial.print(imu_roll, 2);

  Serial.print(",\"t\":");
  Serial.print(millis());

  Serial.println("}");

  delay(50);  // 약 20Hz
}

// 테스트용 더미 함수
float readUltrasonicDummy(int index) {
  return 30.0 + index;
}

float readYawDummy() {
  return millis() / 1000.0;
}