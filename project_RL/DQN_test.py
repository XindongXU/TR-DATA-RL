import tarfile
from DQ_Learning import DQNet, environment, mask_detect, top_detection
import os
import tensorflow as tf
import numpy as np

with open('./target_pos_list.npy', 'rb') as f:
    target_pos_list = np.load(f)
target_idx = np.random.randint(0, len(target_pos_list))
target_pos = [target_pos_list[target_idx][0], target_pos_list[target_idx][1]]
print(target_pos)

checkpoint_path = "/home/mig5/Desktop/TR_DATA_RL/project_RL/deepqlearning_model"
checkpoint_dir  = os.path.dirname(checkpoint_path)
model = DQNet()
envir = environment()
model.load_weights(checkpoint_path)
model.compile(  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1),
                loss = tf.keras.losses.MeanSquaredError(),
                metrics = 'mae')

# reward_list = []

mask = mask_detect()
s_c = top_detection(mask)
r = -300

a_t = 9
s_n, r = envir.run_one_step(state = s_c, 
                            target_pos = target_pos, 
                            action = a_t)
print("current state =", s_n)
s_c = s_n

while r <= -10:

    a_t = model.get_best(   state_current = s_c,
                            target_pos = target_pos,
                            get_action = True)
    print(a_t)

    s_n, r = envir.run_one_step(state = s_c, 
                                target_pos = target_pos, 
                                action = a_t)

    # reward_list.append(r)
    # print("current reward =", r)
    print("current state =", s_n)
    s_c = s_n

    