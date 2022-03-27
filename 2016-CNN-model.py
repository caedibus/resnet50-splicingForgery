import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.metrics import Precision, Recall

EPOCHS = 5 #args["epochs"]
BATCH_SIZE = 16 #args["batchsize"]
TESTING_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMG_SIZE = 128


inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

model = tf.keras.Sequential()
model.add
