#!/usr/bin/env python3
import requests
import cv2
import numpy as np
from image import get_points
from image_processing import frame_computation, get2points
import time
import serial

CNT = 5000
MAXTIME = 300
DIR = 1
STOP = 3

def manual_calibration(url, ser):

    TIME = 150
    while True:
        key = cv2.waitKey(1)

        # Press Esc key to exit
        if key == 27:
            break

        modify = 0

        if key == 100:
            modify = DIR
        elif key == 97:
            modify = -DIR

        if modify != 0:
            ser.write(f'{modify}\n'.encode())
            TIME = TIME + modify

        time.sleep(0.25)  # wait for the arm to reach the desired position

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        cv2.imshow("Static frame", img)

def main():

    url = "http://10.202.247.126:8080/shot.jpg"
    rows = np.zeros([CNT, 4])
    count = 0

    ser = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(3)  # wait for the serial connection to initialize

    manual_calibration(url, ser)

    TIME = 150
    while True:
        key = cv2.waitKey(1)

        # Press Esc key to exit
        if key == 27:
            break

        modify = 0

        if key == 100:
            modify = DIR
        elif key == 97:
            modify = -DIR

        print(TIME)
        print(count)

        if TIME + modify > MAXTIME:
            modify = 0
        elif TIME + modify < 0:
            modify = 0

        if modify != 0:
            ser.write(f'{modify}\n'.encode())
            TIME = TIME + modify

        time.sleep(0.25)  # wait for the arm to reach the desired position

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        try:
            row = frame_computation(img)
        except (IndexError, ZeroDivisionError, ValueError):
            continue

        if len(row) != 5:
            continue

        m, c = row[2], row[3]
        [xlin, ylin] = get2points(m, c)

        img = cv2.line(img, (ylin[0], xlin[0]), (ylin[-1], xlin[-1]), (0, 0, 255), 2)
        cv2.imshow("Static frame", img)

        rows[count, 1:4] = row[[0, 1, 4]]
        rows[count, 0] = TIME

        count += 1

        if count >= CNT:
            break

    fields = "time, base_x, base_y, theta"
    np.savetxt("test2.csv", rows[:count], delimiter=",", header=fields, fmt='%.7f', comments='')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
