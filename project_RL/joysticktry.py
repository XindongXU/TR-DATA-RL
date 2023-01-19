from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from pygame import joystick
import numpy as np

key3values = []
x = 0

def print_add(joy):
    print('Added', joy)

def print_remove(joy):
    print('Removed', joy)

def key_received(key):

    global key3values
    global x

    print('Key:', key)
    if key.keyname == 'Button 0':
        print(key.value)
        # print(min(key3values), max(key3values), min(np.abs(key3values)))
        print(x+1)
        x = x + 1

    if key.keyname.endswith('Axis 3'):
        print(key.value)
        key3values.append(key.value)

    if key.keyname.endswith('Axis 4'):
        print(key.value)

run_event_loop(print_add, print_remove, key_received)