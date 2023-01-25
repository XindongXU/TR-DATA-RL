import cv2
import numpy as np
from sklearn import linear_model

cap = cv2.VideoCapture(0)

lower_green = np.array([45,70,60])
upper_green = np.array([90,255,200])

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
        
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('green mask', mask)

    pointmask = np.zeros((480, 640, 3))
    pointmask[:, :, 1] = mask
    top0, top1 = top_detection(mask)
    # mask = mask/2
    # # print(np.shape(frame))
    # # print(top0, top1)
    pointmask[top0, top1, 0] = 255
    pointmask[top0, top1, 2] = 255
    cv2.imshow('green point', pointmask)

    
    # press escape to exit
    if (cv2.waitKey(30) == 27):
       break

cap.release()
cv2.destroyAllWindows()