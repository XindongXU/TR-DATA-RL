#!/usr/bin/env python3
import requests
import cv2
import numpy as np
from image_processing import frame_computation, get_points
from ard import *
import time
import random
import serial

CNT = 1000


def main():

    url = "http://10.202.247.106:8080/shot.jpg"
    time.sleep(3) # wait for the ip camera to connect
    rows = np.zeros([CNT, 5])
    count = 0

    ser1 = serial.Serial('/dev/ttyACM0', 9600)
    ser2 = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(3)  # wait for the serial connection to initialize
    angle1 = 0
    angle2 = 0
    ser1.write(f'{angle1}\n'.encode())
    ser1.write(f'{angle2}\n'.encode())
    while True:
        key = cv2.waitKey(1)

        # Press Esc key to exit
        if key == 27:
            break

        modified = False

        if key == 97:
            angle1 += 1
            modified = True
        elif key == 100:
            angle1 -= 1
            modified = True
            
        if key == 97: 
            angle2 += 1
            modified = True
        elif key == 100:
            angle2 -= 1
            modified = True
        
        print(angle1)
        print(angle2)
        print(count)

        if angle1 > 75:
            angle1 = 75
        
        if angle1 < -75:
            angle1 = -75

        if angle2 > 75:
                angle2 = 75
        
        if angle2 < -75:
            angle2 = -75

        print('inaite de seriala')
        if modified:
            ser1.write(f'{angle1}\n'.encode())
            ser2.write(f'{angle2}\n'.encode())

        print('inainte de timeout')
        time.sleep(0.001)  # wait for the arm to reach the desired position

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        # cv2.imshow("Web camera", img)

        try:
            row = frame_computation(img)
        except (IndexError, ZeroDivisionError, ValueError):
            continue

        if len(row) != 5:
            continue

        m, c = row[2], row[3]
        [xlin, ylin] = get_points(m, c)

        xlin = xlin.round().astype(int)
        ylin = ylin.round().astype(int)

        img = cv2.line(img, (ylin[0], xlin[0]), (ylin[-1], xlin[-1]), (0, 0, 255), 2)
        cv2.imshow("Static frame", img)

        rows[count, 2:] = row[[0, 1, 4]]
        rows[count, 0] = angle1
        rows[count, 1] = angle2
        # save angle2 somehow
        count += 1

        if count >= CNT:
            break

    fields = "angle, base_x, base_y, theta"
    np.savetxt("test.csv", rows[:count], delimiter=",", header=fields, fmt='%.7f', comments='')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
