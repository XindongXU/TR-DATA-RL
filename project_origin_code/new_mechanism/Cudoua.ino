#include <Servo.h>
#include <stdio.h>

Servo myservo;

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo gripper;
Servo wrist_ver;


long target1, target2, cur1, cur2, tr;
int servo_cnt; // 1 means base, 2 means

// Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
#define SOFT_START_CONTROL_PIN  12
#define CONTROL_DELAY 10


void setup() {
  // Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
  Serial.begin(9600);
  pinMode(SOFT_START_CONTROL_PIN,OUTPUT);
  digitalWrite(SOFT_START_CONTROL_PIN,HIGH);

  base.attach(11);
  shoulder.attach(10);

  base.write(90);
  shoulder.write(90);
  delay(1000);
  cur1 = 90;
  cur2 = 90;
}

void serialEvent() {
  servo_cnt = Serial.parseInt();
  Serial.read();  // get rid of the trailing newline
  tr = Serial.parseInt();
  Serial.read();  // get rid of the trailing newline
  tr = constrain(tr, -75, 75);
  tr += 90;

  if (servo_cnt == 1)
    target1 = tr;
  if (servo_cnt == 2)
    target2 = tr;
  Serial.print("Servo nr: ");
  Serial.println(servo_cnt);
  Serial.print("received ");
  Serial.println(tr);
}

void loop() {
      
    if (cur1 < target1)
        cur1++;
    else if (cur1 > target1)
        cur1--;
      
    if (cur2 < target2)
        cur2++;
    else if (cur2 > target2)
        cur2--;
        
    base.write(cur1);
    shoulder.write(cur2);
    delay(CONTROL_DELAY);
  
}
