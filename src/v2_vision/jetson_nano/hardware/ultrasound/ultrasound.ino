#include <Arduino.h>
#include <math.h>

// =====================
// 초음파 센서 핀 설정
// =====================

// D1: 앞쪽 센서
const int TRIG_D1 = 6;
const int ECHO_D1 = 7;

// D2: 중앙 센서
const int TRIG_D2 = 4;
const int ECHO_D2 = 3;

// D3: 뒤쪽 센서
const int TRIG_D3 = 11;
const int ECHO_D3 = 10;

// =====================
// 센서 배치 정보
// =====================

const float SENSOR_SPACING_CM = 5.25;   // D1-D2, D2-D3 간격
const float X1 = -SENSOR_SPACING_CM;    // D1 위치
const float X2 = 0.0;                   // D2 위치
const float X3 = SENSOR_SPACING_CM;     // D3 위치

// 목표 벽 거리
const float TARGET_DISTANCE_CM = 30.0;  // 원하는 벽과의 거리, 필요하면 수정

// 초음파 측정 범위
const float MIN_VALID_CM = 2.0;
const float MAX_VALID_CM = 200.0;

// =====================
// 초음파 거리 측정 함수
// =====================

float readUltrasonicCM(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // timeout: 25ms 정도면 약 4m까지 측정 가능
  unsigned long duration = pulseIn(echoPin, HIGH, 25000UL);

  if (duration == 0) {
    return -1.0;  // 측정 실패
  }

  // 음속 기준 거리 계산
  // 거리(cm) = 시간(us) * 0.0343 / 2
  float distance = duration * 0.0343 / 2.0;

  if (distance < MIN_VALID_CM || distance > MAX_VALID_CM) {
    return -1.0;  // 비정상값 처리
  }

  return distance;
}

// =====================
// 3개 센서값으로 벽 각도/거리 계산
// =====================

void calculateWallPose(
  float d1,
  float d2,
  float d3,
  float &angle_deg,
  float &center_distance_cm,
  float &perpendicular_distance_cm,
  float &distance_error_cm
) {
  /*
    좌표계:
    x축: 휠체어 앞뒤 방향
    y축: 센서가 바라보는 벽 방향

    D1 = (x1, d1)
    D2 = (x2, d2)
    D3 = (x3, d3)

    벽을 직선 y = m*x + b 로 근사
  */

  // 3개 점을 이용한 직선 기울기 m 계산
  // x 좌표가 -a, 0, +a로 대칭이므로 단순화 가능
  float m = (d3 - d1) / (2.0 * SENSOR_SPACING_CM);

  // 절편 b 계산
  // 세 점의 평균을 사용해서 중앙 거리 추정
  float b = (d1 + d2 + d3) / 3.0;

  // 벽과 휠체어의 상대 각도
  float angle_rad = atan(m);
  angle_deg = angle_rad * 180.0 / PI;

  // 중앙 센서 위치에서 벽까지의 거리
  center_distance_cm = b;

  // 휠체어 중심점에서 벽까지의 최단거리
  // 직선 y = mx + b 를 mx - y + b = 0 으로 보고 원점과의 거리 계산
  perpendicular_distance_cm = b / sqrt(m * m + 1.0);

  // 목표 거리와의 오차
  // 양수: 벽에서 너무 멂
  // 음수: 벽에 너무 가까움
  distance_error_cm = center_distance_cm - TARGET_DISTANCE_CM;
}

void setup() {
  Serial.begin(9600);

  pinMode(TRIG_D1, OUTPUT);
  pinMode(ECHO_D1, INPUT);

  pinMode(TRIG_D2, OUTPUT);
  pinMode(ECHO_D2, INPUT);

  pinMode(TRIG_D3, OUTPUT);
  pinMode(ECHO_D3, INPUT);

  Serial.println("3 Ultrasonic Wall Angle & Distance Test Start");
}

void loop() {
  // 초음파 센서는 동시에 쏘면 간섭이 생길 수 있으므로 순차 측정
  float d1 = readUltrasonicCM(TRIG_D1, ECHO_D1);
  delay(30);

  float d2 = readUltrasonicCM(TRIG_D2, ECHO_D2);
  delay(30);

  float d3 = readUltrasonicCM(TRIG_D3, ECHO_D3);
  delay(30);

  // 측정 실패 처리
  if (d1 < 0 || d2 < 0 || d3 < 0) {
    Serial.println("Sensor read error");
    delay(200);
    return;
  }

  float angle_deg;
  float center_distance_cm;
  float perpendicular_distance_cm;
  float distance_error_cm;

  calculateWallPose(
    d1,
    d2,
    d3,
    angle_deg,
    center_distance_cm,
    perpendicular_distance_cm,
    distance_error_cm
  );

  // =====================
  // 결과 출력
  // =====================

  Serial.print("D1: ");
  Serial.print(d1);
  Serial.print(" cm, ");

  Serial.print("D2: ");
  Serial.print(d2);
  Serial.print(" cm, ");

  Serial.print("D3: ");
  Serial.print(d3);
  Serial.print(" cm | ");

  Serial.print("Angle: ");
  Serial.print(angle_deg);
  Serial.print(" deg | ");

  Serial.print("Center Distance: ");
  Serial.print(center_distance_cm);
  Serial.print(" cm | ");

  Serial.print("Perpendicular Distance: ");
  Serial.print(perpendicular_distance_cm);
  Serial.print(" cm | ");

  Serial.print("Distance Error: ");
  Serial.print(distance_error_cm);
  Serial.println(" cm");

  delay(100);
}