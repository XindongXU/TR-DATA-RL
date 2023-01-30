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
    """
    Fonction that detects the green part on the current image captured by webcam.
    cv2.VideoCapture(0)

    Args:
        None.

    Returns:
        mask (np.ndarray): A mask in form of (480, 640) with values of 0 or 255.
    """
    # set green thresh
    lower_green = np.array([45,70,60])
    upper_green = np.array([90,255,255])

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret :
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
    cap.release()
    print("debug: shape of mask is :", np.shape(mask))
    return mask
        

def top_detect(mask):
    """
    Fonction that detects the top of the green toothstick based on the mask for the green part
    cv2.VideoCapture(0)

    Args:
        mask (np.ndarray): A mask in form of (480, 640) with values of 0 or 255.

    Returns:
        top_x, top_y (float, float): Position of the current top of the green toothstick.
    """
    greenpos0 = []
    greenpos1 = []
    for (index0, liste) in enumerate(mask):
        for (index1, value) in enumerate(liste):
            if value == 255:
                greenpos0.append(index0)
                greenpos1.append(index1)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(np.array(greenpos0).reshape(-1, 1), np.array(greenpos1).reshape(-1, 1))

    # get a mask of inlier points in which includes the tops of the line
    inlier_mask = ransac.inlier_mask_

    # calculate the distance of the two tops to (0, 0)
    dis_top1 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]))
    dis_top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0, 0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0, 0]))

    if dis_top1 > dis_top2 :
        top_x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0]
        top_y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0]
    else:
        top_x = np.array(greenpos0).reshape(-1, 1)[inlier_mask][0, 0]
        top_y = np.array(greenpos1).reshape(-1, 1)[inlier_mask][0, 0]
    return top_x, top_y


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
    """
    A tensorflow keras model class, which gets input of current state information, 
    given target position and the action chosen by the algorithm, and tries to predict 
    the Q state action value, which presents the quality of such an action under 
    this circumstance.
    
    Attributes:
        convo: tf.keras.layers.Conv2D(filters, kernel_size, padding)
        dense: tf.keras.layers.Dense(units = output shape, activation = tf.nn.tanh or none)
    """
    def __init__(self):
        super().__init__()
        # self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=[8, 8], 
        #               padding='same', activation=tf.nn.relu)
        # self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[4, 4],
        #               padding='same', activation=tf.nn.relu)
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # self.flatten = tf.keras.layers.Flatten(input_shape=(5))
        self.dense1 = tf.keras.layers.Dense(64, activation = tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(16, activation = tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(4, activation = tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense2(inputs)
        x = self.dense3(x)
        x = self.dense4(x)
        output = x
        return output
    
    def get_best(self, state_current, target_pos, get_action = True):
        """
        Fonction that gives the best action given the current state situation
        and target position, which maximize the Q state-action value.

        Args:
            state_current (list): [top_x, top_y], a two elements list.
            target_pos (list): [x, y], a two elements list.
            get_action (bool): return the action index, if true, 
                               else the best Q value could obtain.

        Returns:
            action (int): varies from 1 to 9, if get_action == True
            value (float): best Q value could obtain.
        """
        action = 1
        inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], action]])
        value = self.call(inputs = inputs).numpy()[0][0]
        print("debug: best q value is: ", value)
        
        for i in range(8):
            # i varies from 0 to 7, action of i+2 varies from 2 to 9
            inputs = tf.constant([[state_current[0], state_current[1], 
                               target_pos[0], target_pos[1], i + 2]])
            value_new = self.call(inputs = inputs).numpy()[0][0]
            if value <= value_new:
                value = value_new
                action = i + 2
        
        return action if get_action else value


class environment:
    """
    An environment which updates at every time step after the algorithm of robotic arm
    taking action to the current situation.
    
    Attributes:
        servo_value:  the action decided by the agent.
        servo_target: the true servo command value that is going to pass to the adrino board.
    """
    def __init__(self):
        # low and high are lists
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
    gamma = 0.95
    
    envir = environment()
    DQL = DQNet()
    DQL.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001),
                loss = tf.keras.losses.MeanSquaredError(), metrics = 'mae')
    
    for e in range(episode):
        target_idx = np.random.randint(0, len(target_pos_list))
        target_pos = [target_pos_list[target_idx][0], target_pos_list[target_idx][1]]
        start = time.time()
        epsilon = 6 / (e + 1)
        step_num = 100

        # detection of initial state, randomize first action, memorize first sequence
        mask = mask_detect()
        action = np.random.randint(1, 10)
        s_c = top_detect(mask)
        s_n, reward = envir.run_one_step(s_c, target_pos, action)
        save_memory(replay_memory, memory_size, s_c, target_pos, action, reward, s_n)

        while step_num:
            target_idx = np.random.randint(0, len(target_pos_list))
            target_pos = [target_pos_list[target_idx][0], target_pos_list[target_idx][1]]
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
                        value_ = DQL.get_best([mini[6], mini[7]], [mini[2], mini[3]], get_action = False)
                        y_train.append(mini[5] + gamma*value_)
                    x_train.append([mini[0], mini[1], mini[2], mini[3], mini[4]])

                DQL.fit(np.array(x_train), np.array(y_train),
                        # batch_size = 100, #每一批batch的大小为32，
                        epochs = 300,)
                        # validation_split = 0.2, #从数据集中划分20%给测试集
                        # validation_freq = 20)

                DQL.save_weights('/home/mig5/Desktop/TR_DATA_RL/project_RL/deepqlearning_model')
                # np.save('minibatch', minibatch)
                # print(minibatch)
            
            step_num -= 1
            print(step_num)
        
        end = time.time()
        print('Episode:{0:d}'.format(e),
              '    time:{0:.4f}'.format(end-start),
              '    reward:{0:4f}'.format(reward),
              )

        # reset_pos()
        # print(reward_list)
        plt.plot()
        plt.savefig('./deep_img' + str(e) + '.png')


if __name__ == '__main__':

    with open('./target_pos_list.npy', 'rb') as f:
        target_pos_list = np.load(f)

    train(episode = 1000, target_pos_list = target_pos_list)