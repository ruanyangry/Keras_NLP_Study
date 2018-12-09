# _*_ coding:gbk _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
'''

from keras.models import Sequential
from keras.layers import Dense, Activation

# First build layer
# 1. 可以通过向Sequential模型传递一个layer的list来构造该模型
# 2. 也可以通过.add()方法一个个的将layer加入模型中

# method 1
model=Sequential([Dense(32,units=784),Activation("relu"),Dense(10),Activation("softmax")])

# method 2
# Dense == fully connected
model=Sequential()
model.add(Dense(32,input_shape=(784,)))
model.add(Activation="relu")

# 指定输入数据的 shape
# 从前往后，一层的输出作为另一层的输入，直到最后一个输出层

# 基于 input_shape 关键字指定输入数据 shape，tuple 类型
# None == 此位置可能是任何正整数
# 对于2D层（Dense)，可以指定其输入维度 input_dim 来隐含的
# 指定输入数据的 shape

model=Sequential()
model.add(Dense(32,input_dim=784))

model=Sequential()
model.add(Dense(32,input_shape=(784,)))

# compile
# 使用 compile 是对模型进行配置操作
# compile 接收参数：优化器optimizer,损失函数loss,指标列表metrics

# For a multi-class classification problem

model.compile(optimizer="rmsprop",loass="categorical_crossentropy",\
metrics=["accuracy"])

# For a binary classification problem

model.compile(optimizer="rmsprop",loass="categorical_crossentropy",\
metrics=["accuracy"])

# For a mean squared error regression problem

model.compile(optimizer="rmsprop",loss="mse")

# For custom metrics

import keras.backend as K

def mean_pred(y_true,y_pred):
	return K.mean(y_pred)
	
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy",\
mean_pred]

# 训练
# Keras以Numpy数组作为输入数据和标签的数据类型

# For a single-input model with 2 classes (binary classification):

model=Sequential()
model.add(Dense(32,activation="relu",input_dim=100))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

# Generate dummy data

import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(2,size=(1000,1))

# Train the model,iterating on the data in batches of 32 samples

model.fit(data,labels,epochs=10,batch_size=32)

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))

# 10 classes
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
