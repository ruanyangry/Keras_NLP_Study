# _*_ coding:gbk _*_

# MLP的二分类

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Reference: https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
'''

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',\
metrics=['accuracy'])

model.fit(x_train, y_train,epochs=20,batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)

print("#----------------------------------#")
print(score)
print("#----------------------------------#")
print("\n")
              
             
