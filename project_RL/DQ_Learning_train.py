from DQ_Learning import environment, DQNet, LearningRateReducerCb, GetMinibatch
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

replay_memory = np.load('./replay_memory.npy', allow_pickle=True)
replay_memory = np.ndarray.tolist(replay_memory)

envir = environment()
model = DQNet()
model.save_weights("./predict_model_0")
model_ = DQNet()
model_.load_weights("./predict_model_0")
model_.save_weights("./target_model_0")

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = tf.keras.losses.MeanSquaredError(), metrics = 'mae')
model_.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = tf.keras.losses.MeanSquaredError(), metrics = 'mae')

mae_list = []

for step in range(33):
    
    x_train, y_train = [], []
    minibatch = GetMinibatch(1000, replay_memory)

    for (i, mini) in enumerate(minibatch):
        if mini[5] >= -10:
            y_train.append(mini[5])
        else:
            value_ = model_.get_best([mini[6], mini[7]], [mini[2], mini[3]], get_action = False)
            y_train.append(mini[5] + 0.99*value_)
            
        action_0 = 0.1 * (-1 + (mini[4] - 1)//3)
        action_1 = 0.1 * (-1 + (mini[4] - 1) %3)
        x_train.append([mini[0], mini[1], mini[2], mini[3], action_0, action_1])
        
    history = model.fit(np.array(x_train), np.array(y_train), batch_size = 64, epochs = 100)
    # callbacks = [LearningRateReducerCb(step)]
    model.save_weights("./predict_model_0")
    mae_list.append(history.history['mae'])
    
    print(step)
    if (step%3 == 0):
        print("___________target network update___________")
        model_.load_weights("./predict_model_0")
        model_.save_weights("./target_model_0")


plt.plot(np.array(mae_list).reshape((-1)))
plt.show()