# _*_ coding:utf-8 _*_

'''
@Author: Ruan Yang
@Date: 2018.12.9
@Purpose: 熟悉 keras 中的双向 LSTM 网络结构
@Reference: https://github.com/ruanyangry/keras/blob/master/examples/imdb_bidirectional_lstm.py
'''

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

# 定义参数

max_features=20000
maxlen=100
batch_size=32

print("#-----------------------------------------------#")
print("加载数据集.................")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train),"train sequences")
print(len(x_test),"test sequences")
print("#---------------------------------------------------#")
print("\n")

print("#---------------------------------------------------#")
print("使用pad_sequences将输入句子固定成相同长度 (samples x time")
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
y_train=np.array(y_train)
y_test=np.array(y_test)
print("#---------------------------------------------------#")
print("\n")

print("#---------------------------------------------------#")
print("Build model .................")
print("NN structure .......")
print("Embedding layer --- Bidirectional layer --- Dense layer")
print("#---------------------------------------------------#")

model=Sequential()
model.add(Embedding(max_features,128,input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

print("#---------------------------------------------------#")
print("模型编译..............")
print("#---------------------------------------------------#")
print("\n")

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("#---------------------------------------------------#")
print("开始训练.......")
print("#---------------------------------------------------#")
print("\n")

model.fit(x_train,y_train,batch_size=batch_size,epochs=2,validation_data=(x_test,y_test))

# 训练得分和准确度

score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测

predictions=model.predict(x_test)

print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
print(predictions)
print("#---------------------------------------------------#")
print("\n")

# 模型预测类别

predict_class=model.predict_classes(x_test)

print("#---------------------------------------------------#")
print("测试集的预测类别")
print(predict_class)
print("#---------------------------------------------------#")
print("\n")

# 模型保存

#model.save("imdb_lstm.h5")

print("#---------------------------------------------------#")
print("保存模型")
print("#---------------------------------------------------#")
print("\n")

# 模型总结

summary=model.summary()

print("#---------------------------------------------------#")
print("输出模型总结")
print(summary)
print("#---------------------------------------------------#")
print("\n")

# 模型的配置文件

config=model.get_config()

print("#---------------------------------------------------#")
print("输出模型配置信息")
print(config)
print("#---------------------------------------------------#")
print("\n")

