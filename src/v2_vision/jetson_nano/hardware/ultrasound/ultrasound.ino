#include <Arduino.h>

// 예시용 센서값
float us[8];

float side_angle_deg = 0.0;
float side_distance_cm = 0.0;

float imu_yaw = 0.0;
float imu_pitch = 0.0;
float imu_roll = 0.0;

const float SENSOR_SPACING_TOTAL_CM = 10.5;  // D1-D3 거리
const float TARGET_DISTANCE_CM = 40.0;

void setup() {
  Serial.begin(115200);
}

void loop() {
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