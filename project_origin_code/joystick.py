#!/usr/bin/env python3
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from pprint import pprint
from threading import Thread
from time import sleep, time
import serial

servo_0_value = 0
servo_1_value = 0
servo_0_target = 0
servo_1_target = 0


def print_add(joy):
    print('added', joy)


def print_remove(joy):
    print('removed', joy)


def key_received(key):
    global servo_0_value, servo_1_value
    global servo_0_target, servo_1_target

    # key.value varies from -1 to 1

    if key.keyname == 'Button 0':
        servo_0_target = 0
        servo_1_target = 0

    if key.keyname.endswith('Axis 3'):
        servo_0_value = key.value if abs(key.value) > 0.05 else 0

    if key.keyname.endswith('Axis 4'):
        servo_1_value = -key.value if abs(key.value) > 0.05 else 0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))
    

def thread_fn():
    # a theme or a characterisitc, typically forming one of several,
    # running throughout a situation or piece of writing

    global servo_0_target, servo_1_target
    speed = 60

    ser = serial.Serial('/dev/ttyACM0')

    old_time = time()

    while True:
        new_time = time()
        delta_time = new_time - old_time
        old_time = new_time

        servo_0_target = servo_0_target + speed * servo_0_value * delta_time
        servo_1_target = servo_1_target + speed * servo_1_value * delta_time

        servo_0_target = clamp(servo_0_target, 0, 180)
        servo_1_target = clamp(servo_1_target, 0, 180)

        print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
        ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
        sleep(0.1)


def main():
    communication_thread = Thread(target=thread_fn)
    communication_thread.start()

    run_event_loop(print_add, print_remove, key_received)



if __name__ == '__main__':
    main()
