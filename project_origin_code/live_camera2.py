#!/usr/bin/env python3
import requests
import cv2
import numpy as np
from image_processing import frame_computation, get_points
import time
import serial
from coordinates_processing2 import get_splines, inverse_kinematics

CNT = 1000
XMAX = 1258.9
XMIN = 765.34


def main():

    url = "http://10.202.247.106:8080/shot.jpg"
    time.sleep(3)  # wait for the ip camera to connect

    aux = get_splines()
    # theta_spl = aux[0:2]
    x_spl = aux[2:4]
    # y_spl = aux[4:6]

    ser = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(3)  # wait for the serial connection to initialize

    curr_x = 0.5  # daca statea puta cum trebuie
    curr_angle = 0
    err = 0
    n = 0
    ser.write(f'{curr_angle}\n'.encode())

    while True:
        key = cv2.waitKey(1)

        # Press Esc key to exit
        if key == 27:
            break

        modified = False

        if key == 97:
            curr_x += 0.01
            dir = 0
            modified = True
        elif key == 100:
            curr_x -= 0.01
            dir = 1
            modified = True

        print(curr_x)

        if curr_x > XMAX:
            curr_x = XMAX

        if curr_x < XMIN:
            curr_x = XMIN

        if modified:
            curr_angle = inverse_kinematics(x_spl[dir], 0.5)
            ser.write(f'{curr_angle}\n'.encode())

        time.sleep(0.001)  # wait for the arm to reach the desired position

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        try:
            row = frame_computation(img)
        except (IndexError, ZeroDivisionError, ValueError):
            continue

        if len(row) != 5:
            continue

        x = row[0]
        x_01 = (x - XMIN) / (XMAX - XMIN)

        print(np.abs(x_01 - curr_x))
        err += np.abs(x_01 - curr_x)
        n += 1

        m, c = row[2], row[3]
        [xlin, ylin] = get_points(m, c)

        xlin = xlin.round().astype(int)
        ylin = ylin.round().astype(int)

        img = cv2.line(img, (ylin[0], xlin[0]), (ylin[-1], xlin[-1]), (0, 0, 255), 2)
        cv2.imshow("Static frame", img)

    cv2.destroyAllWindows()
    print(err / n)


if __name__ == "__main__":
    main()
