# _*_ coding:utf-8 _*_

'''
@Author: Ruan Yang
@Date: 2018.12.9
@Purpose: 熟悉 在keras 中使用預训练的词向量模型
@Reference: https://github.com/ruanyangry/keras/blob/master/examples/pretrained_word_embeddings.py
@Attention: 本例使用 20 Newsgroup dataset,存在20个类别
@           預训练模型使用 GloVe embeddings
'''

import os
import codecs
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

BASE_DIR = r'C:\Users\RY\Desktop\Keras_tutorial\data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print("#-----------------------------------------------#")
print("index mapping word and index mapping word vector")
print("#-----------------------------------------------#")
print("\n")

embeddings_index={}

with codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),"r","utf-8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype="float32")
        embeddings_index[word]=coefs

print("#-----------------------------------------------#")
print("預训练词向量中的单词总数：{}".format(len(embeddings_index)))
print("#-----------------------------------------------#")
print("\n")

print("#-----------------------------------------------#")
print("处理新闻数据集和label信息")
print("#-----------------------------------------------#")
print("\n")

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print("#-----------------------------------------------#")
print("输入文本总数：{}".format(len(texts)))
print("#-----------------------------------------------#")
print("\n")

print("#-----------------------------------------------#")
print("输入文本向量化")
print("注意使用 keras.preprocessing.text 中的 Tokenizer 进行句子处理时")
print("默认是使用 空格区分单词的，所以在处理中文的时候，传递进行的文本需要")
print("预先使用空格进行分隔")
print("#-----------------------------------------------#")
print("\n")

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# word_index 是 word-index 的字典
word_index=tokenizer.word_index

#for key,value in word_index:
#    print("{} --- {}".format(key,value))

print("#-----------------------------------------------#")
print("单独 tokens 个数：{}".format(len(word_index)))
print("#-----------------------------------------------#")
print("\n")


print("#-----------------------------------------------#")
print("padding sequences")
print("#-----------------------------------------------#")
print("\n")

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("#-----------------------------------------------#")
print("将label从字符串转换成整数")


labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print("#-----------------------------------------------#")
print("\n")

print("#-----------------------------------------------#")
print("拆分数据为训练和测试集")

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print("#-----------------------------------------------#")
print("\n")

print("#-----------------------------------------------#")
print("构建词汇表index-word vector矩阵 embedding_matrix")
print("#-----------------------------------------------#")
print("\n")

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
print("#-----------------------------------------------#")
print("将預训练数据加载到 Embedding layer中")
print("设定 trainable = False 保证输入的 embedding 是固定的")
print("#-----------------------------------------------#")

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print("#-----------------------------------------------#")
print("训练模型")
print("NN structure .......")
print("Conv1D layer --- Conv1D layer --- Conv1D layer --- Dense layer")
print("#-----------------------------------------------#")

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

print("#---------------------------------------------------#")
print("模型编译..............")
print("#---------------------------------------------------#")
print("\n")

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("#---------------------------------------------------#")
print("开始训练.......")
print("#---------------------------------------------------#")
print("\n")

model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          validation_data=(x_val, y_val))

# 训练得分和准确度

batch_size=128
score,acc=model.evaluate(x_val, y_val,batch_size=batch_size)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测

predictions=model.predict(x_val)

print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
print(predictions)
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



