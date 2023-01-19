import cv2
import numpy as np

def webcamera():
    cap = cv2.VideoCapture(0)
    # set blue thresh
    lower_green = np.array([35,70,60])
    upper_green = np.array([99,255,200])

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
        cv2.imshow('frame', frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        edges = cv2.Canny(mask, 100, 200)
        cv2.imshow('green edges', edges)

        # detection of the green part in the video
        # pointdetected = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2RGB)
        # cv2.imshow('pointdetected', pointdetected)

        # change to hsv model and get mask
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.imshow('mask', mask)
        
        # # detect blue
        # res = cv2.bitwise_and(frame, frame, mask = mask)
        # cv2.imshow('Result', res)
        
        
        # press escape to exit
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

# webcamera()