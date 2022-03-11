import numpy as np
import matplotlib.pyplot as plt
# import os
# import cv2
import argparse
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer


ap = argparse.ArgumentParser()
ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
# ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
args = vars(ap.parse_args())


EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
TESTING_SIZE = 0.85
VALIDATION_SIZE = 0.15
IMG_SIZE = 224
SEED_VALUE = 30

#Decalre dataset for training split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SIZE)

#Don't know if I need further preprocessing here:
train_img_generator = train_datagen.flow_from_directory(
    directory = args["training"],
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    color_mode = 'rgb',
    subset = 'training'
    )

img_validation_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=VALIDATION_SIZE)
#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = args["training"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = 10,
    # batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    color_mode = 'rgb',
    subset = 'validation'
    )

inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# inputTensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

pretrained_resnet101 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputTensor)

for layer in pretrained_resnet101.layers:
    layer.trainable = False
#Define layers that will be put on top of the freezed Resnet1010 model
output = pretrained_resnet101.layers[-1].output
#output = keras.layers.Flatten()(output)

# output = pretrained_resnet101.layers[-2].output
output = pretrained_resnet101.output
pretrained_resnet101 = Model(inputs=pretrained_resnet101.input, outputs = output)

output = keras.layers.GlobalAveragePooling2D()(output)
output = keras.layers.Flatten()(output)
output = keras.layers.Dense(512, activation='relu')(output)
output = keras.layers.Dropout(0.25)(output)
output = keras.layers.Dense(1, activation='sigmoid')(output)


pretrained_resnet101 = Model(inputs=pretrained_resnet101.input, output = output)

#for layer in pretrained_resnet101.layers:
#    layer.trainable = False

pretrained_resnet101.summary()

#Define optimizer function
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
sgd = tf.keras.optimizers.SGD(learning_rate = 0.0001)

print("Compile model:")
pretrained_resnet101.compile(
# model.compile(
    optimizer = adam,
    loss="binary_crossentropy",
    # loss="categorical_crossentropy",
    # loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

#Save predictions to csv file
# tf.keras.callbacks.CSVLogger('output history.csv', separator=",", append=False)
# csv_logger = CSVLogger(args["csvName"], "training.csv")
csv_logger = CSVLogger(args["csvName"])

#Inspired by: https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
#fit_generator is used for big datasets that does not fit in to memory
print("Training model with Fit function")
history = pretrained_resnet101.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    callbacks = [csv_logger],
    validation_data=(validation_img_generator),
    # validation_steps=2000
)

# pretrained_resnet101.summary()



#Plot model accuracy in graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend(['Train','Val'])
plt.savefig("acc.pdf")
plt.figure() #Train accuracy

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.legend(['Train','Val']) #Train Loss
# plt.show()

plt.savefig("loss.pdf")
