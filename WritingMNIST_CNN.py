# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 07:54:21 2019

@author: Ujjawal
"""

import keras as kr
import tensorflow as tf
import matplotlib.pyplot as plt

writing_data = kr.datasets.fashion_mnist

(train_writing,train_lable_writing),(test_writing,test_lable_writing) = writing_data.load_data()
plt.imshow(train_writing[1])
print("The Size of Training images: ",train_writing)
train_writing = train_writing.reshape(60000,28,28,1)
print(train_writing[1])
train_writing = train_writing/255
test_writing = test_writing.reshape(10000,28,28,1)
test_writing  = test_writing/255
model = kr.models.Sequential([
        kr.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
                                               kr.layers.MaxPooling2D(2,2),
                                               kr.layers.Conv2D(64,(3,3),activation='relu'),
                                               kr.layers.MaxPooling2D((2,2)),
                                               kr.layers.Flatten(),
                              kr.layers.Dense(128,activation=tf.nn.relu),
                              kr.layers.Dense(10,activation = tf.nn.softmax)
                              ])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss ='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(train_writing,train_lable_writing,epochs=4)
classification = model.predict(test_writing,test_lable_writing)

