import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet  import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras.models import load_model

VALIDATION_SIZE = 0.2
LOADED_MODEL = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\save_model'


img_validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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
    # color_mode = 'rgb',
    subset = 'validation'
    )

model = keras.models.load_model(LOADED_MODEL)

model.fit(
    validation_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    callbacks = [csv_logger],
)
