from operator import ge
import serial
import numpy as np
from sklearn import linear_model
import time
import cv2

def mask_detect():
    lower_green = np.array([35,70,60])
    upper_green = np.array([120,255,255])

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret :
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
    cap.release()
    return mask

def top_position(mask):
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
        return np.array([np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]])
    else:
        return np.array([np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]])


def get_low():
    servo_0_target = 0
    servo_1_target = 0
    ser = serial.Serial('/dev/ttyACM0')
    print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    time.sleep(2)

    mask_low = mask_detect()
    print(top_position(mask_low))
    return top_position(mask_low)



def get_high():
    servo_0_target = 180
    servo_1_target = 180
    ser = serial.Serial('/dev/ttyACM0')
    print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    time.sleep(2)

    mask_high = mask_detect()
    print(top_position(mask_high))
    return top_position(mask_high)

def get_min_max():
    
    low_pos  = get_low()
    high_pos = get_high()
    print(low_pos, high_pos)
    return np.array([min(low_pos[0], high_pos[0]), min(low_pos[1], high_pos[1])]), np.array([max(low_pos[0], high_pos[0]), max(low_pos[1], high_pos[1])])


if __name__ == '__main__':
    
    print(get_min_max())
    # get_high()