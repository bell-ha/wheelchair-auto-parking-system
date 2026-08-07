#include <Arduino.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <math.h>

// ======================================================
// Pin map
// ======================================================

// Ultrasonic index
// 0: Front
// 1: Left Front
// 2: Left Rear
// 3: Right Front
// 4: Right Rear

const int NUM_US = 5;

const int trigPins[NUM_US] = {
  13,  // Front
  14,  // Left Front
  25,  // Left Rear
  26,  // Right Front
  27   // Right Rear
};

const int echoPins[NUM_US] = {
  32,  // Front
  33,  // Left Front
  34,  // Left Rear
  35,  // Right Front
  23   // Right Rear
};

// Servo pins
const int SERVO_X_PIN = 18;  // Servo 1: A/D
const int SERVO_Y_PIN = 19;  // Servo 2: W/S

// ======================================================
// Constants
// ======================================================

// 좌우 각각 앞/뒤 초음파센서 간 거리
const float SIDE_SENSOR_SPACING_CM = 31.5;

// Ultrasonic valid range
const float MIN_VALID_CM = 2.0;
const float MAX_VALID_CM = 300.0;

// Servo range
const float SERVO_MIN_DEG = 0.0;
const float SERVO_MAX_DEG = 180.0;

// Initial servo angle
float servoX_deg = 90.0;
float servoY_deg = 90.0;

// Target angle received from Jetson
float targetServoX_deg = 90.0;
float targetServoY_deg = 90.0;

// Servo pulse width
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2500;

// Loop timing
unsigned long lastSensorUpdateMs = 0;
const unsigned long SENSOR_UPDATE_INTERVAL_MS = 100;

// Serial baudrate
const unsigned long SERIAL_BAUD = 115200;

// Servo objects
Servo servoX;
Servo servoY;

// Latest ultrasonic values
float us_cm[NUM_US] = {-1, -1, -1, -1, -1};

// ======================================================
// Utility
// ======================================================

float clampFloat(float value, float minVal, float maxVal) {
  if (value < minVal) return minVal;
  if (value > maxVal) return maxVal;
  return value;
}

void writeServoDeg(Servo &servo, float angleDeg) {
  angleDeg = clampFloat(angleDeg, SERVO_MIN_DEG, SERVO_MAX_DEG);

  int pulseUs = SERVO_MIN_US + (int)((angleDeg / 180.0) * (SERVO_MAX_US - SERVO_MIN_US));
  servo.writeMicroseconds(pulseUs);
}

// ======================================================
// Ultrasonic
// ======================================================

float readUltrasonicCM(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // 25 ms timeout ≈ 4 m class
  unsigned long duration = pulseIn(echoPin, HIGH, 25000UL);

  if (duration == 0) {
    return -1.0;
  }

  float distance = duration * 0.0343 / 2.0;

  if (distance < MIN_VALID_CM || distance > MAX_VALID_CM) {
    return -1.0;
  }

  return distance;
}

void readAllUltrasonicSequential() {
  for (int i = 0; i < NUM_US; i++) {
    us_cm[i] = readUltrasonicCM(trigPins[i], echoPins[i]);

    // 초음파 간섭 방지용 딜레이
    delay(25);
  }
}

// ======================================================
// Pose calculation
// ======================================================

float calcSideDistance(float frontCm, float rearCm) {
  if (frontCm <= 0 || rearCm <= 0) {
    return -1.0;
  }

  return (frontCm + rearCm) / 2.0;
}

float calcSideAngleDeg(float frontCm, float rearCm) {
  if (frontCm <= 0 || rearCm <= 0) {
    return 0.0;
  }

  // 앞/뒤 센서 거리 차이를 이용한 상대 각도 추정
  // angle > 0 / < 0 방향은 실제 장착 방향에 따라 실험적으로 해석
  return atan((rearCm - frontCm) / SIDE_SENSOR_SPACING_CM) * 180.0 / PI;
}

// ======================================================
// Serial command parsing
// ======================================================

void processSerialCommandLine(String line) {
  line.trim();

  if (line.length() == 0) {
    return;
  }

  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, line);

  if (err) {
    Serial.print("{\"type\":\"error\",\"msg\":\"json_parse_failed\",\"raw\":\"");
    Serial.print(line);
    Serial.println("\"}");
    return;
  }

  const char* cmd = doc["cmd"] | "";

  if (strcmp(cmd, "servo") == 0) {
    float x = doc["x"] | targetServoX_deg;
    float y = doc["y"] | targetServoY_deg;

    targetServoX_deg = clampFloat(x, SERVO_MIN_DEG, SERVO_MAX_DEG);
    targetServoY_deg = clampFloat(y, SERVO_MIN_DEG, SERVO_MAX_DEG);

    // 초음파 측정 주기(최대 ~250ms 블로킹)를 기다리지 않고 즉시 반영
    updateServos();
  }
}

void receiveSerialCommands() {
  static String rxLine = "";

  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      processSerialCommandLine(rxLine);
      rxLine = "";
    } else if (c != '\r') {
      rxLine += c;

      // 비정상적으로 긴 입력 방지
      if (rxLine.length() > 200) {
        rxLine = "";
      }
    }
  }
}

// ======================================================
// Servo update
// ======================================================

void updateServos() {
  servoX_deg = targetServoX_deg;
  servoY_deg = targetServoY_deg;

  writeServoDeg(servoX, servoX_deg);
  writeServoDeg(servoY, servoY_deg);
}

// ======================================================
// Telemetry send
// ======================================================

void sendTelemetryJson() {
  float front = us_cm[0];

  float left_front = us_cm[1];
  float left_rear  = us_cm[2];
  float left_dist  = calcSideDistance(left_front, left_rear);
  float left_angle = calcSideAngleDeg(left_front, left_rear);

  float right_front = us_cm[3];
  float right_rear  = us_cm[4];
  float right_dist  = calcSideDistance(right_front, right_rear);
  float right_angle = calcSideAngleDeg(right_front, right_rear);

  StaticJsonDocument<384> doc;

  doc["type"] = "telemetry";
  doc["front"] = front;

  doc["left_front"] = left_front;
  doc["left_rear"] = left_rear;
  doc["left_dist"] = left_dist;
  doc["left_angle"] = left_angle;

  doc["right_front"] = right_front;
  doc["right_rear"] = right_rear;
  doc["right_dist"] = right_dist;
  doc["right_angle"] = right_angle;

  doc["servo_x"] = servoX_deg;
  doc["servo_y"] = servoY_deg;

  doc["t"] = millis();

  serializeJson(doc, Serial);
  Serial.println();
}

// ======================================================
// Setup / Loop
// ======================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  for (int i = 0; i < NUM_US; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
    digitalWrite(trigPins[i], LOW);
  }

  servoX.setPeriodHertz(50);
  servoY.setPeriodHertz(50);

  servoX.attach(SERVO_X_PIN, SERVO_MIN_US, SERVO_MAX_US);
  servoY.attach(SERVO_Y_PIN, SERVO_MIN_US, SERVO_MAX_US);

  writeServoDeg(servoX, servoX_deg);
  writeServoDeg(servoY, servoY_deg);

  Serial.println("{\"type\":\"status\",\"msg\":\"esp32_ready\"}");
}

void loop() {
  // 1. Jetson에서 온 키보드 기반 서보 명령 수신
  receiveSerialCommands();

  unsigned long now = millis();

  if (now - lastSensorUpdateMs >= SENSOR_UPDATE_INTERVAL_MS) {
    lastSensorUpdateMs = now;

    // 2. 초음파센서 5개 순차 측정
    readAllUltrasonicSequential();

    // 3. 센서값이 갱신될 때마다 현재 목표각으로 모터 제어
    updateServos();

    // 4. Jetson으로 로그 전송
    sendTelemetryJson();
  }
}