import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import MappingView
# from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from pprint import pprint
from threading import Thread
import time
import serial
import random
import numpy as np
from webcam import webcamera
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt
from reset_position import reset_pos
import tensorflow as tf


def mask_detect():
    # set green thresh
    lower_green = np.array([45,70,60])
    upper_green = np.array([90,255,255])

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret :
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # edges = cv2.Canny(mask, 100, 200)
    cap.release()
    return mask
        

def top_detect(mask):
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
    
    top1 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]))
    top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0, 0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0, 0]))

    if top1 > top2 :
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]
    else:
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]
    return x, y

def eval_reward(s_pos, target_pos):
    """
    Get the current state reward, 
    by calculating the distance betweem the current toothstick top and the given target.

    Args:
        s_pos (list): [top_x, top_y], a two elements list.
        target_pos (list): [x, y], a two elements list.
        NO np.ndarray

    Returns:
        reward (float): second norm betweem s_pos and t_pos.
    """
    reward = np.linalg.norm(np.array(s_pos) - np.array(target_pos))
    reward = reward * -1
    return reward

class DQNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(4, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense3(x)
        output = x
        return output
    
    def get_best(self, state_current, target_pos, get_action = True):
        action = 1
        inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], 
                               action]])
        value = self.call(inputs = inputs).numpy()[0][0]
        
        for i in range(8):
            inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], 
                               i + 2]])

            value_new = self.call(inputs = inputs).numpy()[0][0]
            if value <= value_new:
                value = value_new
                action = i + 2

        return action if get_action else value

class environment:
    
    def __init__(self):
        # representing the min and max of each state possible values
        self.servo_0_value = 0
        self.servo_1_value = 0
        self.servo_0_target = 0
        self.servo_1_target = 0

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
        
    def run_one_step(self, state, target_pos, action):
        # perform action chosen, return new state info, reward and other info
        # action at = 1, 2, ..., 9
        speed = 100
        ser = serial.Serial('/dev/ttyACM0')

        self.servo_0_value = 0.1 * (-1 + (action - 1)//3)
        self.servo_1_value = 0.1 * (-1 + (action - 1) %3)
        # servo values vary from -0.1 to 0.1, and need to be timed by speed
        self.servo_0_target = self.servo_0_target + speed * self.servo_0_value
        self.servo_1_target = self.servo_1_target + speed * self.servo_1_value

        # self.servo_0_target = action[0]
        # self.servo_1_target = action[1]

        self.servo_0_target = max(0, min(180, self.servo_0_target))
        self.servo_1_target = max(0, min(180, self.servo_1_target))

        print(f'\r {self.servo_0_target:.2f} {self.servo_1_target:.2f}')
        ser.write(f'{int(self.servo_0_target) << 1}\n{(int(self.servo_1_target) << 1) + 1}\n'.encode())
        # time.sleep(max(self.servo_0_target, self.servo_1_target)/90)
        time.sleep(0.2)

        ## detection of top point at next time step
        mask = mask_detect()
        top_x, top_y = top_detect(mask)

        state_next = [top_x, top_y]
        # print("target position =", self.__decode(self.target_pos), self.target_pos)
        # print("ourarm position =", self.__decode(s_t), s_t)

        reward_current = eval_reward(state_next, target_pos)
        print("current reward  =", reward_current)

        return state_next, reward_current

def save_memory(replay_memory, memory_size, state_current, target_pos, action_current, reward_current, state_next):
    """
    Save all the enivornment parameters into an experience storage, which
    will be used during replay learning, and the trainning of network.
    And ensure that the size of repaly memory is smaller than the given size.

    Args:
        replay_memory   (list): a list of list of 8 elements.
        memory_size     (int)
        state_current   (list of iwo elements)
        target_pos      (list of iwo elements)
        action_current  (int): varies from 1 to 9.
        reward_current  (float)
        state_next      (list of iwo elements)

    Returns:
        None.
    """
    replay_memory.append([state_current[0], state_current[1], target_pos[0], target_pos[1], 
                        action_current, reward_current, state_next[0], state_next[1]])

    if len(replay_memory) > memory_size:
        replay_memory.popleft()

def GetMinibatch(minibatch_size, replay_memory):
    """
    Randomly sample some components from replay memory storage, 
    with a length of given value.

    Args:
        minibatch_size  (int) : maximum of volume of one mini batch.
        replay_memory   (list): a list of list of 8 elements.

    Returns:
        minibatch (list): a list of list of 8 elements, with length of batch size.
    """
    batch_size = minibatch_size if len(replay_memory) > minibatch_size else len(replay_memory)
    minibatch = random.sample(replay_memory, batch_size)

    return minibatch
    


def train(episode, target_pos_list):
    
    replay_memory = []
    memory_size = 10000
    minibatch_size = 500
    gamma = 0.9

    target_idx = 30
    target_pos = [target_pos_list[target_idx][0], target_pos_list[target_idx][1]]
    print(target_pos)
    
    reward_list = []
    envir = environment()
    DQL = DQNet()
    DQL.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
                loss = tf.keras.losses.MeanSquaredError(), metrics = 'mae')
    DQL.save_weights("./predict_model")
    
    DQL_ = DQNet()
    DQL_.load_weights("./predict_model")
    DQL_.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
                loss = tf.keras.losses.MeanSquaredError(), metrics = 'mae')
    DQL_.save_weights("./target_model")


    for e in range(episode):
        start = time.time()
        epsilon = 10 / (e + 1)
        step_num = 100

        # detection of initial state, randomize first action, memorize first sequence
        mask = mask_detect()
        action = np.random.randint(1, 10)
        s_c = top_detect(mask)
        s_n, reward = envir.run_one_step(s_c, target_pos, action)
        save_memory(replay_memory, memory_size, s_c, target_pos, action, reward, s_n)

        while step_num:
            s_c = s_n
            if np.random.random() <= epsilon:
                action = np.random.randint(1, 10)
            else:
                action = DQL.get_best(s_c, target_pos, get_action = True)
            
            s_n, reward = envir.run_one_step(s_c, target_pos, action)
            save_memory(replay_memory, memory_size, s_c, target_pos, action, reward, s_n)

            # update parameters in deep q learning network
            if (len(replay_memory) > 100 and step_num%50 == 0):
                x_train, y_train = [], []
                minibatch = GetMinibatch(minibatch_size, replay_memory)
                for (i, mini) in enumerate(minibatch):
                    # one example of mini with 8 elements: 
                    # [state_current[0], state_current[1], target_pos[0], target_pos[1], 
                    # action_current, reward_current, state_next[0], state_next[1]]
                    if mini[5] >= -10:
                        y_train.append(mini[5])
                    else:
                        value_ = DQL_.get_best([mini[6], mini[7]], [mini[2], mini[3]], get_action = False)
                        y_train.append(mini[5] + gamma*value_)
                    x_train.append([mini[0], mini[1], mini[2], mini[3], mini[4]])

                DQL.fit(np.array(x_train), np.array(y_train),
                        # batch_size = 100, #每一批batch的大小为32，
                        epochs = 100,)
                        # validation_split = 0.2, #从数据集中划分20%给测试集
                        # validation_freq = 20)
                DQL.save_weights("./predict_model")

            reward_list.append(reward)
            step_num -= 1
            print(step_num)
        
        end = time.time()
        print('Episode:{0:d}'.format(e),
              '    time:{0:.4f}'.format(end-start),
              '    reward:{0:4f}'.format(reward),
              )

        # reset_pos()
        # print(reward_list)
        plt.plot(reward_list)
        file_name = './deep_img' + str(e) + '.png'
        plt.savefig(file_name)

        if (e%5 == 0):
            DQL_.load_weights("./predict_model")
            DQL_.save_weights("./target_model")

    with open('./reward_data.npy', 'wb') as f:
        np.save(f, reward_list)
        # np.save(f, Q_table)


if __name__ == '__main__':

    with open('./target_pos_list.npy', 'rb') as f:
        target_pos_list = np.load(f)

    train(episode = 1000, target_pos_list = target_pos_list)