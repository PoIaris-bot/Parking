#include <Servo.h>

const int LEFT_MOTOR_PIN = 11;
const int RIGHT_MOTOR_PIN = 6;
const int LEFT_DIR_PIN = 5;
const int RIGHT_DIR_PIN = 3;
const int SERVO_PIN = 9;

const int LEFT_FORWARD = 255;
const int LEFT_BACKWARD = 0;
const int RIGHT_FORWARD = 0;
const int RIGHT_BACKWARD = 255;

Servo servo;

int dir = 0;
int speed = 0;
int angle = 90;
int left_dir = LEFT_FORWARD;
int right_dir = RIGHT_FORWARD;
String data = "";

void setup()
{
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
    while (Serial.available())
    {
        char c = Serial.read();
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')) data += c;
        delay(1);
    }
    Serial.flush();

    if (data.length()) {
        // format: cmd[xxx][x][xx]dmc
        int begin = data.beginOf("cmd");
        int end = data.beginOf("dmc");
        if (begin != -1 && end != -1 && end - begin == 9) {
            angle = atoi(data.substring(begin + 3, begin + 6).c_str());
            dir = atoi(data.substring(begin + 6, begin + 7).c_str());
            speed = atoi(data.substring(begin + 7, begin + 9).c_str());
        }
        data = "";
    }
    left_dir = (dir == 0 ? LEFT_FORWARD : LEFT_BACKWARD);
    right_dir = (dir == 0 ? RIGHT_FORWARD : RIGHT_BACKWARD);

    servo.write(angle);
    analogWrite(LEFT_DIR_PIN, left_dir);
    analogWrite(RIGHT_DIR_PIN, right_dir);
    analogWrite(LEFT_MOTOR_PIN, speed);
    analogWrite(RIGHT_MOTOR_PIN, speed);
    delay(2);
}
