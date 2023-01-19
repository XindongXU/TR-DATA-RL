import cv2
import matplotlib.pyplot as plt
import time

video = cv2.VideoCapture("tentacle.mp4")

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()

for i in range(0, 250):

    frame_nr = i

    print(frame_nr)
    video.set(1, frame_nr)

    succes, frame = video.read()
    frame = cv2.flip(frame, 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    viewer.clear()
    viewer.imshow(frame)
    plt.pause(0.01)
    fig.canvas.draw()
