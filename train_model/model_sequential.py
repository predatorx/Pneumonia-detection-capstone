import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def model_fn(input_shape):
    model = Sequential()

    #Block 1
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation="relu", name="inputs",
                     padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2), strides = 2 , padding = 'same'))
    
    #Block 2
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.22))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    
    #Block 3
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    #Block 4
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    #Head
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    #Final Outputs
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer = keras.optimizers.Adam(lr=0.00003), loss='binary_crossentropy', metrics=["binary_accuracy"])

    return model