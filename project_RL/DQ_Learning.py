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
        lower_green = np.array([35,70,60])
        upper_green = np.array([120,255,255])

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

def eval_reward(s_pos):
    global target_pos
    # reward(s_t[:, 0], s_t[:, 1])
    reward = np.linalg.norm(s_pos - target_pos)
    return -reward

class DQNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[8, 8],
            padding='same',
            activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[4, 4],
            padding='same',
            activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        # x = self.flatten(inputs)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(inputs)                      # [batch_size, 1024]
        x = self.dense2(x)
        x = self.dense3(x)
        output = x
        return output

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
        
    def run_one_step(self, state, action):
        # perform action chosen, return new state info, reward and other info
        # action at = 0, 1, 2...
        speed = 60
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

        ## detection of top point at s_t
        mask = mask_detect()
        top_x, top_y = top_detection(mask)

        state_next = np.array([top_x, top_y])
        # print("target position =", self.__decode(self.target_pos), self.target_pos)
        # print("ourarm position =", self.__decode(s_t), s_t)

        reward_current = eval_reward(state_next)
        print("current reward  =", reward_current)

        return state_next, reward_current

def save_memory(replay_memory, memory_size, state_current, action_current, reward_current, state_next):
    replay_memory.append([state_current, action_current, reward_current, state_next])

    if len(replay_memory) > memory_size:
        replay_memory.popleft()

def GetMinibatch(minibatch_size, replay_memory):
    batch_size = minibatch_size if len(replay_memory) > minibatch_size else len(replay_memory)
    minibatch = random.sample(replay_memory, batch_size)

    return minibatch
    


def train(episode):
    global target_pos

    replay_memory = []
    memory_size = 1000
    minibatch_size = 100
    gamma = 0.9
    
    reward_list = []
    envir = environment()
    DQL = DQNet()
    


    for e in range(episode):

        start = time.time()
        epsilon = 10 / (e + 1)

        mask = mask_detect()
        top_x, top_y = top_detection(mask)
        s_c = np.array([top_x, top_y])
        # ac0 = np.array([np.random.random()*180, np.random.random()*180])
        action = np.random.randint(1, 10)
        s_t, reward = envir.run_one_step(s_c, action)

        save_memory(replay_memory, memory_size, s_c, action, reward, s_t)
        step_num = 100

        while step_num:
            s_c = s_t
            if np.random.random() <= epsilon:
                action = np.random.randint(1, 10)
            else:
                action = 1
                value = DQL(tf.constant([[s_t[0], s_t[1], target_pos[0], target_pos[1], action]]))
                
                for i in range(8):
                    value_new = DQL(tf.constant([[s_t[0], s_t[1], target_pos[0], target_pos[1], i + 2]]))
                    if value < value_new:
                        value = value_new
                        action = i + 2
            
            s_t, reward = envir.run_one_step(s_c, action)
            save_memory(replay_memory, memory_size, s_c, action, reward, s_t)
            minibatch = GetMinibatch(minibatch_size, replay_memory)

            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            for (i, mini) in enumerate(minibatch):
                with tf.GradientTape() as tape:
                    action_ = 1
                    value = DQL(tf.constant([[mini[3][0], mini[3][1], target_pos[0], target_pos[1], action_]]))
                    print(value)
                    for i in range(8):
                        value_new = DQL(tf.constant([[mini[3][0], mini[3][1], target_pos[0], target_pos[1], i + 2]]))
                        if value < value_new:
                            value = value_new
                            action_ = i + 2

                    y = mini[2] + gamma*value
                    y_pred = DQL(tf.constant([[mini[0][0], mini[0][1], target_pos[0], target_pos[1], mini[1]]]))

                    loss = tf.square(y_pred - y)
                grads = tape.gradient(loss, DQL.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
                optimizer.apply_gradients(grads_and_vars=zip(grads, DQL.variables))

            reward_list.append(reward)
            step_num -= 1
            print(step_num)
        
        end = time.time()
        print('Episode:{0:d}'.format(e),
              '    time:{0:.4f}'.format(end-start),
              '    reward:{0:4f}'.format(reward),
              )

        # reset_pos()
        print(reward_list)
        plt.plot(reward_list)
        file_name = './img' + str(e) + '.png'
        plt.savefig(file_name)

    with open('./reward_data.npy', 'wb') as f:
        np.save(f, reward_list)
        # np.save(f, Q_table)


if __name__ == '__main__':
    target_pos = np.array([400, 500])
    train(episode = 1000)