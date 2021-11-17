import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.keras import backend as K
from tensorflow.python import logging as tf_logging
from model_InceptionResNetV2 import model_fn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import glob
import cv2
import matplotlib as mat
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from botocore.exceptions import ClientError 
from library import *

from tensorflow.keras import callbacks

bucket_name = 'aws-ml-sagemaker-pneumonia-detection-v2'        # Pneumonia images data bucket
s3 = boto3.client('s3')
plt.figure(dpi=600)


import warnings
warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.INFO)
tf_logging.set_verbosity(tf_logging.INFO)

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

os.environ['PYTHONHASHSEED']=str(1)
IMG_SIZE = 224
HEIGHT = IMG_SIZE
WIDTH = IMG_SIZE
BATCH_SIZE = 32
BATCH = 32
SEED = 42
DEPTH = 3
hyperparameters = {}

#INPUT_TENSOR_NAME = "inputs_input" # Watch out, it needs to match the name of the first layer + "_input"      

if __name__ == "__main__":
    
    args, _ = parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.train
    input_shape = (224, 224, 3)
    model_name = 'InceptionResNetV2'
    
    reset_random_seeds()
    
    model = model_fn(input_shape)
    model.summary()

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, factor=0.3, min_lr=0.000001)
    
    print("batch_size = {}, epochs = {}, learning rate = {}, training_dir = {}".format(batch_size, epochs, lr, training_dir))
    training_dir = '/opt/ml/input/data/training'
    evaluation_dir = '/opt/ml/input/data/eval'
    test_dir = '/opt/ml/input/data/test'
    ds_train, ds_val, train_df, val_df = train_input_fn(training_dir, hyperparameters)
    ds_eval, df_eval = eval_input_fn(evaluation_dir, hyperparameters)
    ds_test, df_test = test_input_fn(test_dir, hyperparameters)
    
    history = model.fit(ds_train,
          epochs = epochs,
          validation_data=ds_val,
          callbacks=[learning_rate_reduction],
          steps_per_epoch=(len(train_df)/batch_size),
          validation_steps=(len(val_df)/batch_size));
    
    generate_learning_curve_loss_plot(history, model_name)
    history_dict = history.history
    print('history keys: {}'.format(history_dict.keys()))
    generate_learning_curve_accuracy_plot(history, model_name)
    
    # Evaluate the validation data set
    score = model.evaluate(
        ds_eval,
        steps=len(df_eval)/batch_size,
        verbose=0,
    )
    
    print("Evaluation loss:{}".format(score[0]))
    print("Evaluation accuracy:{}".format(score[1]))
    
    # Evaluate the test data set
    score = model.evaluate(
        ds_test,
        steps=len(df_test)/batch_size,
        verbose=0,
    )

    print("Test loss:{}".format(score[0]))
    print("Test accuracy:{}".format(score[1]))
    
    num_label = {'Normal': 0, 'Pneumonia' : 1}
    Y_test = df_test['class'].copy().map(num_label).astype('int')
    
    ds_test.reset()
    predictions = model.predict(ds_test)
    pred_labels = np.where(predictions > 0.5, 1, 0)
    
    generate_confusion_matrix_plot(Y_test, pred_labels, ds_test, df_test, model_name)
    
    print(metrics.classification_report(Y_test, pred_labels, target_names = ['Normal (Class 0)', 'Pneumonia (Class 1)']))
    
    roc_auc = metrics.roc_auc_score(Y_test, predictions)
    print('ROC_AUC: ', roc_auc)
    
    
    save_model(model, args.model_output_dir)
    
