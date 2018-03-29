import os
import re
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv',sep=',')

print(train.columns)

train_length = len(train)
print(train_length)

test = pd.read_csv('test.csv',sep=',')

test_length = len(test)
print(test_length)

x_train = np.zeros((train_length,28,28))
x_test = np.zeros((test_length,28,28))

y_train = np.zeros((train_length,1))

for i in range(train_length):
  y_train[i] = train["label"][i]

train = train.drop(["label"],axis=1)

x_train = train.as_matrix()

x_test = test.as_matrix()

x_train = x_train.reshape((train_length,28,28,1))

x_test = x_test.reshape((test_length,28,28,1))

import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras import metrics

x_train = x_train/255
x_test = x_test/255

label_train = keras.utils.to_categorical(y_train,num_classes=10)

model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=(3,3)))
model.add(Conv2D(16,kernel_size=(3,3)))
model.add(Conv2D(8,kernel_size=(3,3)))
model.add(Conv2D(4,kernel_size=(3,3)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,label_train,batch_size=512,epochs=50)

model.save('best_mnist_weights.h5')

model = load_weights('best_mnist_weights.h5')

pred = model.predict(x_test)

test_labels = np.zeros(test_length)

for i in range(test_length):
  test_labels[i] = int(np.argmax(pred[i]))

import csv

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['ImageID', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    count = 1
    writer.writeheader()
    for predict in range(test_length):
        num = count
        label = int(test_labels[predict])
        writer.writerow({'ImageID': num, 'Label': label})
        count+=1


