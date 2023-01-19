#!/usr/bin/env python3
import requests
import cv2
import numpy as np
from image_processing import frame_computation, get2points
from ard import *
import time
from random import randint
import serial

CNT = 1000
START = 0
STOP = 180
POS_TIME = 0.5
CONNECTION_TIME = 3
VIB_TIME = 0.5


def visualise(img, row):
    m, c = row[2], row[3]
    [xlin, ylin] = get2points(m, c)

    xlin = xlin.round().astype(int)
    ylin = ylin.round().astype(int)

    img = cv2.line(img, (ylin[0], xlin[0]), (ylin[-1], xlin[-1]), (0, 0, 255), 2)
    cv2.imshow("Static frame", img)


def position(angle, servo_nr):
    return ((angle << 1) | servo_nr)


def main():

    # url = "http://10.202.247.24:8080/shot.jpg"
    # rows = np.zeros([CNT, 5]) # angle1, angle2, x, y, theta
    # count = 0

    ser = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(CONNECTION_TIME)  # wait for the serial connection to initialize

    prevangle0 = 0
    prevangle1 = 0
    ser.write(f'{position(prevangle0, 0)}\n'.encode())
    ser.write(f'{position(prevangle1, 1)}\n'.encode())

    time.sleep(POS_TIME)

    while True:
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
        
        angle0 = randint(START, STOP + 1)
        angle1 = randint(START, STOP + 1)

        # print(f"{count}: {angle0}, {angle1}")

        ser.write(f'{position(angle0, 0)}\n'.encode())
        ser.write(f'{position(angle1, 1)}\n'.encode())

        time.sleep(VIB_TIME + POS_TIME * max(abs(angle0 - prevangle0), abs(angle1 - prevangle1)) / 180)

        # img_resp = requests.get(url)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # img = cv2.imdecode(img_arr, -1)
        # # cv2.imshow("Web camera", img)

        # try:
        #     row = frame_computation(img)
        # except (IndexError, ZeroDivisionError, ValueError):
        #     continue

        # if len(row) != 5:
        #     continue

        # # visualise(img, row)

        # rows[count, 2:] = row[[0, 1, 4]]
        # rows[count, 0] = angle0
        # rows[count, 1] = angle1

        # count += 1

        # if count >= CNT:
        #     break

    # fields = "angle0, angle1, x, y, theta"
    # np.savetxt("2servos2.csv", rows[:count], delimiter=",", header=fields, fmt='%.7f', comments='')
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
