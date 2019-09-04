# Basic Data preprocesing and analysis to feed NN

This notebook contains basic steps for data preprocessing and analysis 
in order to prepare it for Neural Net training.

### Prerequisites

Python 3.x   
from os import sys   
import os   
import pandas as pd   
import numpy as np   
import keras   
from keras import optimizers   
from keras.models import Sequential   
from keras.callbacks import History   
from keras.layers import Dense, Dropout, Activation   
from keras.layers.normalization import BatchNormalization   
from keras.callbacks import ModelCheckpoint   
from keras.utils.np_utils import to_categorical   
from sklearn.model_selection import train_test_split   
from sklearn.datasets import load_digits   
from sklearn.preprocessing import MinMaxScaler   
from sklearn.metrics import f1_score   
from numpy import argmax   
from tqdm import tqdm   
import matplotlib.pyplot as plt   
from seaborn import heatmap   

### Contributing and Collaborating

The code contains model pre-built on Keras that requires fine-tuning.   
Any contributions, comments, enhancements are welcome.   

### Developers Team

* **Viacheslav Nesterov**


# Data_processing_for_ML
