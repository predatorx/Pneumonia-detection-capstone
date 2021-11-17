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

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)


INPUT_TENSOR_NAME = "inputs_input" # Watch out, it needs to match the name of the first layer + "_input"

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=5
                       )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--gpu-count", type=int, 
                                      default=os.environ['SM_NUM_GPUS'])

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model_output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))

    return parser.parse_known_args()


def serving_input_fn(hyperparameters):
    # Here we need a placeholder to store the inference case
    # the incoming images ...
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    print('Training Directory in train_input_fn: {}'.format(training_dir))
    train_path = training_dir

    train_normal = glob.glob(train_path+"/NORMAL/*.jpeg")
    train_pneumonia = glob.glob(train_path+"/PNEUMONIA/*.jpeg")

    train_list = [ x for x in train_normal]
    train_list.extend([x for x in train_pneumonia])

    df_train = pd.DataFrame(np.concatenate([['Normal']*len(train_normal), 
                                            ['Pneumonia']*len(train_pneumonia)]), columns = ['class'])
    df_train['image'] = [x for x in train_list]

    train_df, val_df = train_test_split(df_train, test_size = 0.20, random_state = SEED, stratify = df_train['class'])
    
    ds_train = _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_df=train_df)
    ds_val = _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_df=val_df, shuffle = True)
    
    return ds_train, ds_val, train_df, val_df


def eval_input_fn(evaluation_dir, hyperparameters):
    print('Evaluation Directory in eval_input_fn: {}'.format(evaluation_dir))

    eval_path = evaluation_dir

    eval_normal = glob.glob(eval_path+"/NORMAL/*.jpeg")
    eval_pneumonia = glob.glob(eval_path+"/PNEUMONIA/*.jpeg")

    eval_list = [ x for x in eval_normal]
    eval_list.extend([x for x in eval_pneumonia])

    df_eval = pd.DataFrame(np.concatenate([['Normal']*len(eval_normal), 
                                            ['Pneumonia']*len(eval_pneumonia)]), columns = ['class'])
    
    df_eval['image'] = [x for x in eval_list]
    ds_eval = _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_df=df_eval, shuffle = True)
    
    return ds_eval, df_eval


def test_input_fn(training_dir, hyperparameters):
    print('Evaluation Directory in test_input_fn: {}'.format(training_dir))

    test_path = training_dir
    test_normal = glob.glob(test_path+"/NORMAL/*.jpeg")
    test_pneumonia = glob.glob(test_path+"/PNEUMONIA/*.jpeg")

    test_list = [ x for x in test_normal]
    test_list.extend([x for x in test_pneumonia])

    df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal), 
                                            ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])
    
    df_test['image'] = [x for x in test_list]
    
    ds_test = _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_df=df_test, shuffle = False)
    
    return ds_test, df_test


def _input(mode, batch_size, data_df, shuffle = False):
    print('data_df in _input_fn: {} - {} - {}'.format(data_df, batch_size, mode))
    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)
    
    generator = datagen.flow_from_dataframe(data_df,
                                             x_col = 'image',
                                             y_col = 'class',
                                             target_size = (IMG_SIZE, IMG_SIZE),
                                             class_mode = 'binary',
                                             batch_size = batch_size,
                                             seed = SEED,
                                             shuffle = shuffle)
    
    return generator


def save_model(model, output):
    print("model: {}".format(model))
    print("Output: {}".format(output))
    signature = predict_signature_def(
        inputs={"inputs_input": model.input}, outputs={"scores": model.output}
    )

    builder = SavedModelBuilder(output + "/1/")
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tag_constants.SERVING],
        signature_def_map={
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        },
    )

    builder.save()
    logging.info("Model successfully saved at: {}".format(output))
    

def upload_image_to_s3(image_path, image_name):
    img_data = open(image_path + image_name, "rb")
    boto3.resource('s3').Bucket(bucket_name).put_object(Key=image_name, Body=img_data, 
                                 ContentType="image/png", ACL="public-read")


def generate_confusion_matrix_plot(num_label, Y_test, pred_labels, ds_test, df_test, model):
    labels = ['NORMAL', 'PNEUMONIA']
    confusion_matrix = metrics.confusion_matrix(Y_test, pred_labels)
    cm = pd.DataFrame(confusion_matrix , index = ['0','1'] , columns = ['0','1'])

    plt.figure(figsize = (10,10))
    plt.xlabel("Predicted Label", fontsize= 12)
    plt.ylabel("True Label", fontsize= 12)
    sns.heatmap(cm, cmap = "Blues", linecolor = 'black' , 
                linewidth = 1 , annot = True, fmt='',
                xticklabels = labels, yticklabels = labels)

    image_name = 'confusion_matrix_InceptionResNetV2.png'
    image_path = "/opt/ml/output/"
    plt.savefig(image_path + image_name)
    upload_image_to_s3(image_path, image_name)


def generate_learning_curve_loss_plot(history, hyperparameters):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x = history.epoch, y = history.history['loss'])
    sns.lineplot(x = history.epoch, y = history.history['val_loss'])
    ax.set_title('Learning Curve (Loss)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 0.5)
    ax.legend(['train', 'val'], loc='best')
    image_name = 'learning_curve_loss_InceptionResNetV2.png'
    image_path = "/opt/ml/output/"
    plt.savefig(image_path + image_name)
    upload_image_to_s3(image_path, image_name)

    
def generate_learning_curve_accuracy_plot(history, hyperparameters):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x = history.epoch, y = history.history['binary_accuracy'])
    sns.lineplot(x = history.epoch, y = history.history['val_binary_accuracy'])
    ax.set_title('Learning Curve (Accuracy)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0.80, 1.0)
    ax.legend(['train', 'val'], loc='best')
    image_name = 'learning_curve_accuracy_InceptionResNetV2.png'
    image_path = "/opt/ml/output/"
    plt.savefig(image_path + image_name)
    upload_image_to_s3(image_path, image_name)
    

if __name__ == "__main__":
    
    args, _ = parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.train
    input_shape = (224, 224, 3)
    
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
    
    generate_learning_curve_loss_plot(history, hyperparameters)
    history_dict = history.history
    print('history keys: {}'.format(history_dict.keys()))
    generate_learning_curve_accuracy_plot(history, hyperparameters)
    
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
    
    generate_confusion_matrix_plot(num_label, Y_test, pred_labels, ds_test, df_test, model)
    
    print(metrics.classification_report(Y_test, pred_labels, target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
    
    roc_auc = metrics.roc_auc_score(Y_test, predictions)
    print('ROC_AUC: ', roc_auc)
    
    
    save_model(model, args.model_output_dir)
    
