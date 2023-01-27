import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import MappingView
from pyjoystick.sdl2 import Key, Joystick, run_event_loop
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
        

def top_detection(mask):
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

    # linemask = np.zeros((480, 640))
    # id0 = (np.array(greenpos0).reshape(-1, 1)[inlier_mask]).reshape(1, -1)[0]
    # id1 = (np.array(greenpos1).reshape(-1, 1)[inlier_mask]).reshape(1, -1)[0]
    
    # for i in range(len(id0)):
    #     linemask[id0[i], id1[i]] = 255
    top1 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]))
    top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0, 0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0, 0]))

    if top1 > top2 :
        # s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0])
        # s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0])
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]
    else:
        # s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0])
        # s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0])
        x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0]
        y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]
    return x, y

def eval_reward(s_pos, target_pos):
    reward = np.linalg.norm(s_pos - target_pos)
    return -reward

class DQNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.conv1 = tf.keras.layers.Conv2D(
        #     filters=16,
        #     kernel_size=[8, 8],
        #     padding='same',
        #     activation=tf.nn.relu)
        # self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # self.conv2 = tf.keras.layers.Conv2D(
        #     filters=32,
        #     kernel_size=[4, 4],
        #     padding='same',
        #     activation=tf.nn.relu)
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # self.flatten = tf.keras.layers.Flatten(input_shape=(5))
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # x = self.flatten(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        output = x
        return output
    
    def get_best(self, state_current, target_pos, get_action = True):
        action = 1
        inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], 
                               action]])
        value = self.call(inputs = inputs)
        
        for i in range(8):
            inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], 
                               i + 2]])

            value_new = self.call(inputs = inputs)
            if value < value_new:
                value = value_new
                action = i + 2
        return action if get_action else value

class environment:
    
    def __init__(self):
        # low and high are lists
        # representing the min and max of each state possible values
        self.servo_0_value = 0
        self.servo_1_value = 0
        self.servo_0_target = 0
        self.servo_1_target = 0
        self.servo_0_pos = []
        self.servo_1_pos = []
        self.action_list = []

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
        
    def run_one_step(self, state, target_pos, action):
        # perform action chosen, return new state info, reward and other info
        # action at = 0, 1, 2...
        speed = 100
        ser = serial.Serial('/dev/ttyACM0')

        self.servo_0_value = 0.1 * (-1 + (action - 1)//3)
        self.servo_1_value = 0.1 * (-1 + (action - 1) %3)

        self.servo_0_target = self.servo_0_target + speed * self.servo_0_value * 1
        self.servo_1_target = self.servo_1_target + speed * self.servo_1_value * 1

        # self.servo_0_target = action[0]
        # self.servo_1_target = action[1]

        self.servo_0_target = max(0, min(180, self.servo_0_target))
        self.servo_1_target = max(0, min(180, self.servo_1_target))

        self.servo_0_pos.append(self.servo_0_target)
        self.servo_1_pos.append(self.servo_1_target)
        self.action_list.append(action)

        print(f'\r {self.servo_0_target:.2f} {self.servo_1_target:.2f}')
        ser.write(f'{int(self.servo_0_target) << 1}\n{(int(self.servo_1_target) << 1) + 1}\n'.encode())
        # time.sleep(max(self.servo_0_target, self.servo_1_target)/90)
        time.sleep(0.2)

        ## detection of top point at next time step
        mask = mask_detect()
        top_x, top_y = top_detection(mask)

        state_next = np.array([top_x, top_y])
        # print("target position =", self.__decode(self.target_pos), self.target_pos)
        # print("ourarm position =", self.__decode(s_t), s_t)

        reward_current = eval_reward(state_next, target_pos)
        print("current reward  =", reward_current)

        return state_next, reward_current

def save_memory(replay_memory, memory_size, state_current, target_pos, action_current, reward_current, state_next):
    replay_memory.append([state_current, target_pos, action_current, reward_current, state_next])

    if len(replay_memory) > memory_size:
        replay_memory.popleft()

def GetMinibatch(minibatch_size, replay_memory):
    batch_size = minibatch_size if len(replay_memory) > minibatch_size else len(replay_memory)
    minibatch = random.sample(replay_memory, batch_size)

    return minibatch
    


def train(episode, target_pos_list):
    
    replay_memory = []
    memory_size = 10000
    minibatch_size = 100
    gamma = 0.99
    
    reward_list = []
    envir = environment()
    DQL = DQNet()
    
    for e in range(episode):

        target_pos = target_pos_list[np.random.randint(0, len(target_pos_list))]
        start = time.time()
        epsilon = 5 / (e + 1)
        step_num = 100

        # detection of initial state, randomize first action, memorize first sequence
        mask = mask_detect()
        action = np.random.randint(1, 10)
        s_c = np.array(top_detection(mask))
        s_n, reward = envir.run_one_step(s_c, target_pos, action)
        save_memory(replay_memory, memory_size, s_c, target_pos, action, reward, s_n)

        while step_num:
            target_pos = target_pos_list[np.random.randint(0, len(target_pos_list))]

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
                    value_ = DQL.get_best(mini[4], mini[1], get_action = False)
                    y_train.append(mini[3] + gamma*value_)
                    x_train.append([mini[0][0], mini[0][1], mini[1][0], mini[1][1], mini[2]])

                DQL.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
                            loss = tf.keras.losses.MeanAbsoluteError(),
                            metrics = 'mae')

                DQL.fit(np.array(x_train), np.array(y_train),
                        batch_size = 64, #每一批batch的大小为32，
                        epochs = 600,)
                        # validation_split = 0.2, #从数据集中划分20%给测试集
                        # validation_freq = 20)

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

        with open('./reward_data.npy', 'wb') as f:
            np.save(f, reward_list)
            # np.save(f, Q_table)


if __name__ == '__main__':

    with open('./target_pos_list.npy', 'rb') as f:
        target_pos_list = np.load(f)

    train(episode = 1000, target_pos_list = target_pos_list)