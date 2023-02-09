import cv2
import numpy as np
from sklearn import linear_model
from threading import Thread
from time import sleep, time
import serial
import matplotlib.pyplot as plt

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
            [80, 0], [70, 0], [60, 0], [50, 0], [40, 0], [30, 0], [20, 0], [10, 0], [10, 10], 
            [10, 20], [10, 30], [20, 20], [30, 30], [10, 40], [20, 50], [40, 60], [50, 70], [40, 80], [20, 90], 
            [30, 100], [60, 110], [40, 120], [80, 130], [90, 140], [100, 150], [110, 160], [20, 170], [50, 170], 
            [80, 160], [10, 120], [20, 130], [30, 140], [40, 150], [50, 150], [60, 150], [70, 130], [80, 100], [90, 100], 
            [100, 10], [110, 170], [120, 150], [130, 80], [140, 60], [150, 50], [160, 80], [170, 20], [170, 120],
            [170, 10], [170, 50], [150, 160], [130, 150], [170, 140], [110, 130], [140, 120], [150, 110], [130, 100], [160, 90], 
            [150, 80], [150, 70], [40, 60], [30, 50], [160, 40], [170, 30], [140, 20], [120, 10], [110, 0],
            [130, 20], [170, 20], [160, 30], [150, 40], [140,80], [130, 40], [120, 60], [110, 70], [100, 50], [90, 20], 
            [80, 10], [70, 80], [60, 40], [50, 60], [40, 80], [30, 40], [20, 70], [10, 40], [0, 100]]


def random_target():
    value = np.random.randint(0, 19)
    return value


speed = 60
ser = serial.Serial('/dev/ttyACM0')

cap = cv2.VideoCapture(0)
for actions in actions_liste:
# for i in range(100):

    # servo_0_target = random_target()*10
    # servo_1_target = random_target()*10
    servo_0_target = actions[0]
    servo_1_target = actions[1]

    print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
    ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
    sleep(1)
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

    # with open('./target_pos_list.npy', 'wb') as f:
    #     np.save(f, np.array(target_pos_list))

#     # press escape to exit
#     if (cv2.waitKey(30) == 27):
#        break
# cap.release()
# cv2.destroyAllWindows()

with open('./target_pos_list.npy', 'rb') as f:
    target_pos_list = np.load(f)
    
print(np.shape(target_pos_list))
plt.figure(figsize = (8, 6))
plt.xlim((0, 640))
plt.ylim((-480, 50))
plt.xlabel('x')
plt.ylabel('y')
plt.title('green point position')
plt.scatter(target_pos_list[:, 1], -target_pos_list[:, 0], s = 20, marker = 'x')
plt.show()
