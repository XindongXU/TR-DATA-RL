#include <Servo.h>
#include <stdio.h>

Servo full; // full length servo control -> 1
Servo half; // half length servo control -> 0

long target0, target1, cur0, cur1, target;

// Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
#define SOFT_START_CONTROL_PIN  12
#define CONTROL_DELAY 10

void setup() {
  
  // Braccio shield garbage, see Braccio GitHub for what Braccio::begin does.
  Serial.begin(9600);
  pinMode(SOFT_START_CONTROL_PIN,OUTPUT);
  digitalWrite(SOFT_START_CONTROL_PIN,HIGH);

  full.attach(11);
  half.attach(10);

  full.write(0);
  half.write(0);
  
  delay(1000);
  
  cur0 = target0 = 0;
  cur1 = target1 = 0;
}

void serialEvent() {
  target = Serial.parseInt();
  Serial.read();  // get rid of the trailing newline

  if (target & 1)
    target1 = constrain((target >> 1), 0, 180);
  else
    target0 = constrain((target >> 1), 0, 180);

//  Serial.print("Servo nr: ");
//  Serial.println(target & 1);
//  Serial.print("received ");
//  Serial.println(target >> 1);
}

void loop() {
      
    if (cur1 < target1)
        cur1++;
    else if (cur1 > target1)
        cur1--;
      
    if (cur0 < target0)
        cur0++;
    else if (cur0 > target0)
        cur0--;
        
    full.write(cur1);
    half.write(cur0);
    delay(CONTROL_DELAY);
}
