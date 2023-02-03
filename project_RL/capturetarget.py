import cv2
import numpy as np
from sklearn import linear_model
from threading import Thread
from time import sleep, time
import serial

target_pos_list = []
lower_green = np.array([45,70,60])
upper_green = np.array([90,255,255])

def top_detection(mask):
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

    # linemask = np.zeros((480, 640))
    # id0 = (np.array(greenpos0).reshape(-1, 1)[inlier_mask]).reshape(1, -1)[0]
    # id1 = (np.array(greenpos1).reshape(-1, 1)[inlier_mask]).reshape(1, -1)[0]
    
    # for i in range(len(id0)):
    #     linemask[id0[i], id1[i]] = 255
    top1 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]))
    top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0, 0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0, 0]))

    if top1 > top2 :
        # s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0])
        # s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0])
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]
    else:
        # s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0])
        # s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0])
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]
    return x, y


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

actions_liste = [[0, 0], [0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60], [0, 70], [0, 80], [0, 90], 
            [0, 100], [0, 110], [0, 120], [0, 130], [0, 140], [0, 150], [0, 160], [0, 170], [0, 180], 
            [0, 180], [10, 180], [20, 180], [30, 180], [40, 180], [50, 180], [60, 180], [70, 180], [80, 180], [90, 180], 
            [100, 180], [110, 180], [120, 180], [130, 180], [140, 180], [150, 180], [160, 180], [170, 180], [180, 180],
            [180, 180], [180, 170], [180, 160], [180, 150], [180, 140], [180, 130], [180, 120], [180, 110], [180, 100], [180, 90], 
            [180, 80], [180, 70], [180, 60], [180, 50], [180, 40], [180, 30], [180, 20], [180, 10], [180, 0],
            [180, 0], [170, 0], [160, 0], [150, 0], [140, 0], [130, 0], [120, 0], [110, 0], [100, 0], [90, 0], 
            [80, 0], [70, 0], [60, 0], [50, 0], [40, 0], [30, 0], [20, 0], [10, 0], [0, 0]]


speed = 60
ser = serial.Serial('/dev/ttyACM0')

cap = cv2.VideoCapture(0)
for actions in actions_liste:

    servo_0_target = actions[0]
    servo_1_target = actions[1]

    print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    sleep(0.5)
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    # cv2.imshow('frame', frame)

    frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # cv2.imshow('green mask', mask)

    pointmask = np.zeros((480, 640, 3))
    pointmask[:, :, 1] = mask
    top0, top1 = top_detection(mask)
    print(top0, top1)
    target_pos_list.append([top0, top1])
    # mask = mask/2
    # # print(np.shape(frame))
    # # print(top0, top1)
    pointmask[top0, top1, 0] = 255
    pointmask[top0, top1, 2] = 255
    # cv2.imshow('green point', pointmask)

    with open('./target_pos_list.npy', 'wb') as f:
        np.save(f, np.array(target_pos_list))

#     # press escape to exit
#     if (cv2.waitKey(30) == 27):
#        break
# cap.release()
# cv2.destroyAllWindows()