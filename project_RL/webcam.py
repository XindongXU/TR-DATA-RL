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
        # edges = cv2.Canny(mask, 100, 200)
        cv2.imshow('green mask', mask)

        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcamera()