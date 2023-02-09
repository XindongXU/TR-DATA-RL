from re import I
import tarfile
from DQ_Learning import DQNet, environment, mask_detect, top_detect
from reset_position import reset_pos
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import sleep, time
import serial
import cv2
from reset_position import reset_pos


ser = serial.Serial('/dev/ttyACM0')
actions = [130, 160]
cap = cv2.VideoCapture(0)

servo_0_target = actions[0]
servo_1_target = actions[1]

print(f'\r {servo_0_target:.2f} {servo_1_target:.2f}', end='')
ser.write(f'{int(servo_0_target) << 1}\n{(int(servo_1_target) << 1) + 1}\n'.encode())
sleep(5)

mask = mask_detect()
top0, top1 = top_detect(mask)
print(top0, top1)

reset_pos()

model = DQNet()
envir = environment()
model.load_weights("/home/mig5/Desktop/TR_DATA_RL/project_RL/predict_model_jeudi")

mask = mask_detect()
s_c = top_detect(mask)

a_t = 5
s_n, r = envir.run_one_step(state = s_c, 
                            target_pos = [0, 0], 
                            action = a_t)
print("current state =", s_n)
s_c = s_n
sleep(2)

target_pos = [top0, top1]
reward_list = []
try_times = 50
r = -300
while (r <= -5 and try_times >= 0):

    a_t = model.get_best(   state_current = s_c,
                            target_pos = target_pos,
                            get_action = True,
                            is_training = False)

    s_n, r = envir.run_one_step(state = s_c, 
                                target_pos = target_pos, 
                                action = a_t)

    reward_list.append(r)
    print("current state  =", s_n)
    print("current target =", target_pos)
    print("current reward =", r)
    s_c = s_n
    try_times = try_times - 1

plt.plot(reward_list)
plt.xlabel('time')
plt.ylabel('reward')
plt.title('Running progress')
plt.show()