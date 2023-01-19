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
#define STEP 0.1
#define MAXRIGHT 90
#define MAXLEFT -90
#define STOP 3


void setup() {
  // Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
  Serial.begin(9600);
  pinMode(SOFT_START_CONTROL_PIN,OUTPUT);
  digitalWrite(SOFT_START_CONTROL_PIN,HIGH);

  base.attach(11);

  base.write(STOP);
  delay(1000);
}

void serialEvent() {
  target = Serial.parseInt();
  Serial.read();  // get rid of the trailing newline
//
//  Serial.print("received ");
//  Serial.println(target);
}

void loop() {
  
//    if (cur != target) {
//      Serial.print(target);
//      Serial.print(" ");
//      Serial.println(cur);
//    }
      
    if (target == 1)    
        base.write(MAXRIGHT);
    else
        if (target == -1)
            base.write(MAXLEFT);
        else
            base.write(STOP);
    delay(STEP * 1000);
    
    base.write(STOP)
    
    delay(CONTROL_DELAY);
  
}
