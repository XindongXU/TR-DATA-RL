import tensorflow as tf
from sklearn import datasets
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(120)
np.random.shuffle(x_train)
np.random.seed(120)
np.random.shuffle(y_train)

tf.random.set_seed(120)

class irisModel(tf.keras.Model):
    def __init__(self):
        super(irisModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2()) #搭建网络块，这一层命名为d1
 
    def call(self, x):
        y = self.d1(x)
        return  y

model = irisModel()

#第四步，model.compile()
model.compile(  #使用model.compile()方法来配置训练方法
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1), #使用SGD优化器，学习率为0.1
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), #配置损失函数
    metrics = ['sparse_categorical_accuracy'] #标注网络评价指标
)

#第五步，model.fit()
model.fit(  #使用model.fit()方法来执行训练过程，
    x_train, y_train, #告知训练集的输入以及标签，
    batch_size = 32, #每一批batch的大小为32，
    epochs = 500, #迭代次数epochs为500
    validation_split = 0.2, #从数据集中划分20%给测试集
    validation_freq = 20 #测试的间隔次数为20,每迭代20次测试一次准确率
)

#第六步，model.summary()
model.summary() #打印神经网络结构，统计参数数目