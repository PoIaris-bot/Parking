#include <Servo.h>
#include <math.h>
// pin definitions
const int LEFT_MOTOR_PIN = 11;
const int RIGHT_MOTOR_PIN = 6;
const int LEFT_DIR_PIN = 5;
const int RIGHT_DIR_PIN = 3;
const int SERVO_PIN = 9;
// direction definitions
const int LEFT_FORWARD = 0;
const int LEFT_BACKWARD = 255;
const int RIGHT_FORWARD = 255;
const int RIGHT_BACKWARD = 0;
// offset definitions
const int angle_offset = 0;  // 1: 4; 4: 0; 5: 0; 6: -2; 7: -3; 9: 6
const int speed_offset = -1;  // 1: -1; 4: 0; 5: 1; 6: 3; 7: 4; 9: 4

Servo servo;
int dir = 0;
int speed = 0;
int angle = 90;
int left_dir = LEFT_FORWARD;
int right_dir = RIGHT_FORWARD;
String data = "";

int constraint(int value, int lb, int ub) {
  return (value > ub ? ub : (value < lb ? lb : value));
}

void setup() {
  Serial.begin(115200);

  servo.attach(SERVO_PIN);
  servo.write(angle);

  pinMode(LEFT_MOTOR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PIN, OUTPUT);
  pinMode(LEFT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);
  analogWrite(LEFT_DIR_PIN, left_dir);
  analogWrite(RIGHT_DIR_PIN, right_dir);
  analogWrite(LEFT_MOTOR_PIN, speed);
  analogWrite(RIGHT_MOTOR_PIN, speed);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')) data += c;
    delay(1);
  }
  Serial.flush();

  if (data.length()) {
    // format: cmd[xxx][y][zz]dmc
    // xxx: angle [75, 105]; y: direction {0: backward, 1: forward}; zz: speed [0, 30]
    int begin = data.indexOf("cmd");
    int end = data.indexOf("dmc");
    if (begin != -1 && end != -1 && end - begin == 9) {
      angle = atoi(data.substring(begin + 3, begin + 6).c_str());
      dir = atoi(data.substring(begin + 6, begin + 7).c_str());
      speed = atoi(data.substring(begin + 7, begin + 9).c_str()) + speed_offset;
    }
    data = "";
  }
  left_dir = (dir == 0 ? LEFT_BACKWARD : LEFT_FORWARD);
  right_dir = (dir == 0 ? RIGHT_BACKWARD : RIGHT_FORWARD);

  servo.write(constraint(angle, 75, 105) + angle_offset);
  analogWrite(LEFT_DIR_PIN, left_dir);
  analogWrite(RIGHT_DIR_PIN, right_dir);
  // differential drive
  analogWrite(LEFT_MOTOR_PIN, constraint(speed - 15 * sin((angle - 90) / 180 * 3.14), 0, 30));
  analogWrite(RIGHT_MOTOR_PIN, constraint(speed + 15 * sin((angle - 90) / 180 * 3.14), 0, 30));
  delay(2);
}
