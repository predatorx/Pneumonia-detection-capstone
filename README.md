# Pneumonia-detection-capstone
Pneumonia Detection Udacity Capstone Project 

Welcome to the Udacity Capstone Project - Pneumonia Detection. In this project I have leveraged AWS Sagemaker to run Tensorflow v2 model with x-ray images input uploaded to S3.

The project requires some of the common pip libraries used in the class. Since I am running in Sagemaker, Tensorflow graphs cannot be plotted in the Juypter notebook directly. So I have created requirements.txt to install few packages on the Sagemaker instances that spin up.

Pip packages required on sagemaker instance:
matplotlib
seaborn
boto3

All the x-ray image data for pneumonia detection is provided by Kaggle. 

Kaggle project: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
Dataset to download to notebook: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download

I have commands enabled to unzip and do the next steps as part of the notebook.

Model is run on conda_tensorflow2_36 (Conda with Tensorflow v2 & python 36).

If bucket name needs to be changed, then it needs to be updated in few places. 

Notebook initiates a Tensorflow object which has entry point of train.py located under train_model directory. Model is written in model.py which is referred by train.py.
