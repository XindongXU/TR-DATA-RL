#include <Servo.h>
#include <stdio.h>

Servo myservo;

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo gripper;
Servo wrist_ver;

long target;

// Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
#define SOFT_START_CONTROL_PIN  12
#define CONTROL_DELAY 10
#define STEP 0.2

//#define MAXRIGHT 95
//#define MAXLEFT 91
//#define STOP 93

#define STOP_PULSE_WIDTH 1504
#define LEFT_PULSE_WIDTH 544
#define RIGHT_PULSE_WIDTH 2400

void setup() {
  // Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
  Serial.begin(9600);
  pinMode(SOFT_START_CONTROL_PIN, OUTPUT);
  digitalWrite(SOFT_START_CONTROL_PIN,HIGH);

  base.attach(11);

  base.writeMicroseconds(STOP_PULSE_WIDTH);
  delay(1000);
}

void serialEvent() {
  target = Serial.parseInt();
  Serial.read();  // get rid of the trailing newline
}

void loop() {
    if (target == 1)
        base.writeMicroseconds(RIGHT_PULSE_WIDTH);
    else if (target == -1)
        base.writeMicroseconds(LEFT_PULSE_WIDTH);
    else
        base.writeMicroseconds(STOP_PULSE_WIDTH);
        
    delay(STEP * 1000);
    
    base.writeMicroseconds(STOP_PULSE_WIDTH);
    delay(CONTROL_DELAY);
    target = 0;
}
