#!/usr/bin/env python3
import serial
import time

def reset_pos():
    servo_0_target = 0
    servo_1_target = 0
    ser = serial.Serial('/dev/ttyACM0')
    print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    time.sleep(2)


if __name__ == '__main__':
    reset_pos()