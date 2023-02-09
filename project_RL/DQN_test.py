from re import I
import tarfile
from DQ_Learning import DQNet, environment, mask_detect, top_detect
from reset_position import reset_pos
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random

actions_liste = [[0, 0], [0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60], [0, 70], [0, 80], [0, 90], 
            [0, 100], [0, 110], [0, 120], [0, 130], [0, 140], [0, 150], [0, 160], [0, 170], [0, 180], 
            [0, 180], [10, 180], [20, 180], [30, 180], [40, 180], [50, 180], [60, 180], [70, 180], [80, 180], [90, 180], 
            [100, 180], [110, 180], [120, 180], [130, 180], [140, 180], [150, 180], [160, 180], [170, 180], [180, 180],
            [180, 180], [180, 170], [180, 160], [180, 150], [180, 140], [180, 130], [180, 120], [180, 110], [180, 100], [180, 90], 
            [180, 80], [180, 70], [180, 60], [180, 50], [180, 40], [180, 30], [180, 20], [180, 10], [180, 0],
            [180, 0], [170, 0], [160, 0], [150, 0], [140, 0], [130, 0], [120, 0], [110, 0], [100, 0], [90, 0], 
            [80, 0], [70, 0], [60, 0], [50, 0], [40, 0], [30, 0], [20, 0], [10, 0], [0, 0]]

with open('./target_pos_list.npy', 'rb') as f:
    target_pos_list = np.load(f)

# target_idx = np.random.randint(0, len(target_pos_list))
# target_pos = [target_pos_list[target_idx][0], target_pos_list[target_idx][1]]
# print(target_pos, actions_liste[target_idx])

target_pos_list = np.ndarray.tolist(target_pos_list)
target_pos_list_random = random.sample(target_pos_list, len(target_pos_list))

model = DQNet()
envir = environment()
model.load_weights("/home/mig5/Desktop/TR_DATA_RL/project_RL/predict_model_friday")

mask = mask_detect()
s_c = top_detect(mask)

a_t = 5
s_n, r = envir.run_one_step(state = s_c, 
                            target_pos = [0, 0], 
                            action = a_t)
print("current state =", s_n)
s_c = s_n
time.sleep(2)

acc = 0
# len(target_pos_list_random)
for i in range(20):
    target_pos = target_pos_list_random[i]
    reward_list = []
    try_times = 50
    r = -300
    while (r <= -10 and try_times >= 0):

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
    print('index', i)
    plt.plot(reward_list)
    if (r >= -10):
        acc = acc + 1

print(acc/(i+1))
# 0.8289473684210527
plt.show()