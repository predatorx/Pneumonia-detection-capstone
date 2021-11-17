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
    
    base_model = tf.keras.applications.InceptionResNetV2(
    weights='imagenet',
    input_shape=input_shape,
    include_top=False)

    base_model.trainable = True
    
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    x = base_model(inputs)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    #Final Layer (Output)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[inputs], outputs=output)
    model.compile(loss='binary_crossentropy'
              , optimizer = keras.optimizers.Adam(lr=0.00002), metrics=['binary_accuracy'])
    
    model.summary()
        
    return model