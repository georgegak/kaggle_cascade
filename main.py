#Import
#from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
#from keras.layers.core import Dropout
#from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import SGD
#from keras import backend as K
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from sklearn.utils import shuffle
#import sklearn.preprocessing as pre
#import os
#import sys
#import cv2
import numpy as np
#import h5py
#import time
#import math
#import copy
from extract import extract_to_cnn

#Globals
#seed = 50
#np.random.seed(seed)
#opt = SGD(lr=0.1)
#training=False
#resume_training=False
#skip_training_first=False

def initialize_model(weight):
    #Cascade
    early_stopping = EarlyStopping(monitor='loss', patience=15)
    save_best=ModelCheckpoint('weights'+weight, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    np.random.seed(seed)
    #Model1
    model1 = Sequential()
    model1.add(Convolution2D(64, 5, 5, border_mode="same",input_shape=(3, 768, 768),bias=True,init='he_normal'))
    model1.add(Activation("relu"))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Flatten())
    model1.add(Dense(21,bias=True,init='glorot_normal'))
    return model1

nol=extract_to_cnn(10,10,True)
