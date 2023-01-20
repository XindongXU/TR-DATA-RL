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

def reward(s_pos, target_pos):
    # reward(s_t[:, 0], s_t[:, 1])
    reward = np.linalg.norm(s_pos - target_pos)
    return reward

