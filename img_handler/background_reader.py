import cv2 as cv
import numpy as np
import os, os.path
import time
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.constraints import maxnorm

from img_capture import find_window, capture_window
from matplotlib import pyplot as plt

from PIL import Image

def learn_model():
    data = np.load('background.npz')    
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(354, 623, 3), padding='same', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), padding='same', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation='softmax'))

    model.compile(loss ='categorical_crossentropy' ,optimizer='adam',metrics=['accuracy'])
    hist = model.fit(data['image'], data['answer'])    

if __name__=='__main__':

    # data = np.load('background.npz')
    # type(data['image'])
    # plt.imshow(data['image'][1])
    # plt.show()
                
    learn_model()
