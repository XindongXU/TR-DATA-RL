import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    pointdetected = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
    cv2.imshow('pointdetected', pointdetected)

    
    # press escape to exit
    if (cv2.waitKey(30) == 27):
       break

cap.release()
cv2.destroyAllWindows()