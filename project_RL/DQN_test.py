import tarfile
from DQ_Learning import DQNet, environment, mask_detect, top_detection
import os
import tensorflow as tf
import numpy as np

with open('../target_pos_list.npy', 'rb') as f:
    target_pos_list = np.load(f)
target_pos = target_pos_list[np.random.randint(0, len(target_pos_list))]
print(target_pos)

checkpoint_path = "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model = DQNet()
envir = environment()
model.load_weights(checkpoint_path)
model.compile(  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1),
                loss = tf.keras.losses.MeanAbsoluteError(),
                metrics = 'mae')

# reward_list = []

mask = mask_detect()
s_c = np.array(top_detection(mask))

while True:

    a_t = model.get_best(   state_current = s_c,
                            target_pos = target_pos,
                            get_action = True)

    s_n, r = envir.run_one_step(state = s_c, 
                                target_pos = target_pos, 
                                action = a_t)

    # reward_list.append(r)
    print("current reward =", r)
    print("current state =", s_n)
    s_c = s_n

    