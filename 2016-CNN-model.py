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
from tensorflow.keras.layers import InputLayer, Conv2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.metrics import Precision, Recall

EPOCHS = 5 #args["epochs"]
BATCH_SIZE = 16 #args["batchsize"]
TESTING_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMG_SIZE = 128


inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

model = tf.keras.Sequential()
#convolutional Layer 1
model.add(Conv2D(kernel_size = 30, (5,5), strides = (2,2), activation='relu', input_shape=(inputTensor)))
#Pooling layer 2
model.add(MaxPooling2D(kernel_size = 30, pool_size=(2,2)))
#convolutional Layer 3
model.add(Conv2D(kernel_size = 16, (3,3), activation='relu'))
#convolutional Layer 4
model.add(Conv2D(kernel_size = 16, (3,3), activation='relu'))
#convolutional Layer 5
model.add(Conv2D(kernel_size = 16, (3,3),  activation='relu'))
#pooling layer 6
model.add(MaxPooling2D(kernel_size = 30, pool_size=(2,2)))
#convolutional Layer 7
model.add(Conv2D(kernel_size = 16, (3,3),  activation='relu'))
#convolutional Layer 8
model.add(Conv2D(kernel_size = 16, (3,3),  activation='relu'))
#convolutional Layer 9
model.add(Conv2D(kernel_size = 16, (3,3),  activation='relu'))
#convolutional Layer 10
model.add(Conv2D(kernel_size = 16, (3,3),  activation='relu'))
#Output layer
model.add(Dense(1,activation='softmax'))


ap = argparse.ArgumentParser()
ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
ap.add_argument("-sm", "--saveModel", default='/save_model', help ="saved model output")

# ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
args = vars(ap.parse_args())


EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
TESTING_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMG_SIZE = 224
SEED_VALUE = 30

#Decalre dataset for training split
train_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SIZE,
    rotation_range=30,
    height_shift_range=0.2,
    vertical_flip = True,
    horizontal_flip=True
)

#Don't know if I need further preprocessing here:
train_img_generator = train_datagen.flow_from_directory(
    directory = args["training"],
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    color_mode = 'rgb',
    subset = 'training'
    )

img_validation_generator = ImageDataGenerator(
    # preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SIZE,
    rotation_range=30,
    height_shift_range=0.2,
    vertical_flip = True,
    horizontal_flip=True
)

#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = args["training"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = 10,
    # batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    color_mode = 'rgb',
    subset = 'validation'
    )
