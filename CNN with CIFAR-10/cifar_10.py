import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

import keras

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(len(x_train))
print(len(x_test))

print(x_train.shape)
print(x_test.shape)

y_train = keras.utils.to_categorical(y_train,num_classes=10)
y_test = keras.utils.to_categorical(y_test,num_classes=10)

x_train = x_train/255
x_test = x_test/255

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras import metrics

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=50,shuffle=False)

model.save('cifar10.h5')

model = load_model('cifar10.h5')

pred = model.predict(x_test)

labels = np.zeros(10000)

for i in range(10000):
  labels[i] = np.argmax(pred[i])

count = 0

for i in range(10000):
  if(y_test[i][int(labels[i])]==1):
    count = count + 1

print(count/10000)

