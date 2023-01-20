from pyjoystick.sdl2 import Key, Joystick, run_event_loop
from pprint import pprint
from threading import Thread
import time
import serial
import numpy as np
from webcam import webcamera
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt
from reset_position import reset_pos


class Qtable:
    
    def __init__(self, action_num = 9, state_num = 2, s_split_num = 50):
        self.action_num = action_num
        self.s_split_num = s_split_num
        self.table = np.random.uniform(low=0, high=1, size=(s_split_num**state_num, action_num))
    
    def __cal_idx(self, state_decode):
        # state_decode = (x, y), x=[0, 9], y=[0, 9], split num=10
        # state_decode = (1, 2) -> idx = 2*1 + 1*10
        idx = 0
        length = len(list(state_decode))
        for i in range(length):
            idx += state_decode[length-i-1] * (self.s_split_num**i)
        return idx
    
    def update(self, state_decode, action, reward, lr, GAMMA, max_q_next):
        # update of q table, according to reward at time step t
        idx = self.__cal_idx(state_decode)
        self.table[idx, action] += lr * (reward + GAMMA * max_q_next - self.table[idx, action])
    
    def get_Q(self, state_decode, action):
        # state_decode is the state information after being decoded 
        # decoding by being split into split_num**state_num possibilities, and starting from zeros
        # return the Q value under certain state(state_decode) and action(action)
        idx = self.__cal_idx(state_decode)
        return self.table[idx, action]
    
    def get_best_action(self, state_decode):
        # return the action of the optimal Q value can get under certain state(state_decode)
        idx = self.__cal_idx(state_decode)
        return np.argmax(self.table[idx])
    
    def get_max_q(self, state_decode):
        # return the optimal Q value can get under certain state(state_decode)
        idx = self.__cal_idx(state_decode)
        return float(np.max(self.table[idx]))

class environment:
    
    def __init__(self, low, high, s_split_num, target_pos):
        # low and high are lists
        # representing the min and max of each state possible values
        self.low = low
        self.high = high
        self.s_split_num = s_split_num
        self.target_pos = target_pos
        self.servo_0_value = 0
        self.servo_1_value = 0
        self.servo_0_target = 0
        self.servo_1_target = 0
        self.servo_0_pos = []
        self.servo_1_pos = []
        self.action_0 = []
        self.action_1 = []
        self.s_0 = []
        self.s_1 = []
    
    def __decode(self, state):
        # decoding the continuous state values into several bins
        # return an one dimension array
        # state = (130, 207) -> array([1, 2])
        st_decode = []
        for i in range(len(state)):
            a  = np.linspace(self.low[i], self.high[i], self.s_split_num-1)
            id = np.digitize(state[i], a)
            # numpy.digitize(x, bins, right = False)
            st_decode.append(id)
        return np.array(st_decode)

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
        # cv2.destroyAllWindows()
        # print('mask returned')
        return mask

    def top_detection(mask):
        global s_0, s_1

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
        top2 = np.linalg.norm((np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0], np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0]))

        if top1 > top2 :
            s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][-1,0])
            s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][-1,0])
        else:
            s_0.append(np.array(greenpos0).reshape(-1, 1)[inlier_mask][0,0])
            s_1.append(np.array(greenpos1).reshape(-1, 1)[inlier_mask][0,0])

    def eval_reward(s_pos, target_pos):
        # reward(s_t[:, 0], s_t[:, 1])
        reward = np.linalg.norm(s_pos - target_pos)
        return reward

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
        
    def run_one_step(self, action):
        # perform action chosen, return new state info, reward and other info

        speed = 20
        ser = serial.Serial('/dev/ttyACM0')

        self.servo_0_value = 0.1 * (-1 + action//3)
        self.servo_1_value = 0.1 * (-1 + action %3)

        self.servo_0_target = self.servo_0_target + speed * self.servo_0_value * 1
        self.servo_1_target = self.servo_1_target + speed * self.servo_1_value * 1

        self.servo_0_target = self.clamp(self.servo_0_target, 0, 180)
        self.servo_1_target = self.clamp(self.servo_1_target, 0, 180)

        self.servo_0_pos.append(self.servo_0_target)
        self.servo_1_pos.append(self.servo_1_target)
        self.action_0.append(self.servo_0_value)
        self.action_1.append(self.servo_1_value)

        print(f'\r {self.servo_0_target:.2f} {self.servo_1_target:.2f}')
        ser.write(f'{int(self.servo_0_target) << 1}\n{(int(self.servo_1_target) << 1) + 1}\n'.encode())
        time.sleep(1)

        ## detection of top point at s_t
        mask = self.mask_detect()
        self.top_detection(mask)

        s_t = np.array([s_0[-1], s_1[-1]])
        s_t = self.__decode(s_t)
        reward = self.eval_reward(s_t, self.__decode(self.target_pos))

        return s_t, reward

def train(episode):

    s_split_num = 50
    action_num = 9
    state_num  = 2
    Q_table = Qtable(action_num, state_num, s_split_num)

    low  = [80, 191]
    high = [411, 568]
    envir = environment(low=low, high=high, s_split_num=s_split_num)

    for e in range(episode):
        start = time.time()
        epsilon = 10 / (e + 1)

        action = 9
        s_t, reward = envir.run_one_step(action)
        step_num = 10000

        while step_num:
            if np.random.random() <= epsilon:
                action = np.random.randint(0, 10)
            else:
                action = Q_table.get_best_action(s_t)
            
            s_t1, reward = envir.run_one_step(action)

            max_q_next = Q_table.get_max_q(s_t1)
            Q_table.update(state_decode=s_t, action=action, reward=reward, 
                           lr=0.5, GAMMA=0.99, max_q_next=max_q_next)
            s_t = s_t1
            step_num -= 1
        
        end = time.time()
        print('Episode:{0:d}'.format(e),
              '    time:{0:.4f}'.format(end-start),
              '    reward:{0:4f}'.format(reward),
              )
        reset_pos()


if __name__ == '__main__':
    train(episode = 100)