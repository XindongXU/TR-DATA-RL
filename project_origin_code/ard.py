#!/usr/bin/env python3
import time
import random

PHI = 0.61803398875
DIM = 32
REP = 3


def lerp(a, b, t):
    return a * t + b * (1 - t)


def write_position(ser, seed=0.5):
    pos01 = seed

    while True:
        pos01 = (pos01 + PHI) % 1
        angle = int(lerp(-75, 75, pos01))
        print(angle)
        ser.write(f'{angle}\n'.encode())

        time.sleep(2)


def generate_positions(seed=0.5):

    pos = [-75, 75]

    pos01 = seed
    for i in range(0, DIM - 2):
        pos01 = (pos01 + PHI) % 1
        angle = round(lerp(-75, 75, pos01))
        pos.append(angle)

    return pos


if __name__ == '__main__':
    # ser = serial.Serial('/dev/ttyACM0', 9600)
    # time.sleep(2) # wait for the serial connection to initialize
    # write_position(ser)

    v = generate_positions()
    for i in range(0, REP):
        aux = v.copy()
        random.shuffle(aux)
        v.extend(aux)

    print(v)
