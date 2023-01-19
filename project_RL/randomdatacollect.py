#!/usr/bin/env python3
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from pprint import pprint
from threading import Thread
import time
import serial
import numpy as np
from webcam import webcamera
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt

servo_0_value = 0
servo_1_value = 0
servo_0_target = 0
servo_1_target = 0

servo_0_pos = []
servo_1_pos = []
action_0 = []
action_1 = []
s_0 = []
s_1 = []


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


def random_target():
    value = np.random.uniform(-1, 1)
    return value if abs(value) > 0.05 else 0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def top_detection(mask):
    global s_0, s_1

    greenpos0 = []
    greenpos1 = []
    for (index0,liste) in enumerate(mask):
        for (index1,value) in enumerate(liste):
            if value == 255:
                greenpos0.append(index0)
                greenpos1.append(index1)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(np.array(greenpos0).reshape(-1, 1), np.array(greenpos1).reshape(-1, 1))
    inlier_mask = ransac.inlier_mask_

    top1 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]))
    top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]))

    if top1 > top2 :
        s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0])
        s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0])
    else:
        s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0])
        s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0])


def thread_fn():
    # a theme or a characterisitc, typically forming one of several,
    # running throughout a situation or piece of writing
    
    global servo_0_target, servo_1_target
    global servo_0_value, servo_1_value
    global servo_0_pos, servo_1_pos, action_0, action_1, s_0, s_1

    ser = serial.Serial('/dev/ttyACM0')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    time.sleep(2)

    ## detection of top point at s_t
    mask = mask_detect()
    top_detection(mask)

    ## running the arm
    speed = 60
    # it turns out to be the speed of order giving
    old_time = time.time()
    start_t  = old_time

    while (old_time - start_t < 1500):
        print(old_time - start_t)
        new_time = time.time()
        # delta_time = new_time - old_time
        old_time = new_time

        servo_0_value = random_target()
        servo_1_value = random_target()

        servo_0_target = servo_0_target + speed * servo_0_value * 1
        servo_1_target = servo_1_target + speed * servo_1_value * 1

        servo_0_target = clamp(servo_0_target, 0, 180)
        servo_1_target = clamp(servo_1_target, 0, 180)

        servo_0_pos.append(servo_0_target)
        servo_1_pos.append(servo_1_target)
        action_0.append(servo_0_value)
        action_1.append(servo_1_value)

        print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}')
        ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
        time.sleep(1)

        ## detection of top point at s_t
        mask = mask_detect()
        top_detection(mask)



    
    return np.array([servo_0_pos, servo_1_pos]), np.array([action_0, action_1]), np.array([s_0, s_1])
    # print(servo_0, servo_1)

def mask_detect():
    # set green thresh
    lower_green = np.array([35,70,60])
    upper_green = np.array([120,255,255])

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret :
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
        # cv2.imshow('frame', frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # edges = cv2.Canny(mask, 100, 200)
        # cv2.imshow('green edges0', edges)

    cap.release()
    # cv2.destroyAllWindows()
    # print('mask returned')

    return mask


def thread_run():
    communication_thread = Thread(target=thread_fn)
    communication_thread.start()

    run_event_loop(print_add, print_remove, key_received)

def thread_camera():
    communication_thread = Thread(target = webcamera)
    communication_thread.start()



if __name__ == '__main__':
    
    # thread_camera()
    
    Servo_t, A_t, s_t = thread_fn()
    print(Servo_t)
    print(A_t)
    print(s_t)

    plt.figure(figsize = (6, 8))
    plt.xlim((0, 480))
    plt.ylim((0, 640))
    # plt.plot(s_t[0], s_t[1])
    plt.scatter(s_t[0], s_t[1], s = 20, marker = 'x')

    plt.figure(figsize = (8, 8))
    plt.xlim((-0.1, 180.1))
    plt.ylim((-0.1, 180.1))
    # plt.plot(Servo_t[0], Servo_t[1])
    plt.scatter(Servo_t[0], Servo_t[1], s = 20, marker = 'x')
    plt.show()

    with open('/home/mig5/Desktop/project_RL/data.npy', 'wb') as f:
        np.save(f, Servo_t)
        np.save(f, A_t)
        np.save(f, s_t)