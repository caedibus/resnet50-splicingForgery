import numpy as np
import matplotlib.pyplot as plt
# import os
# import cv2
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.metrics import Precision, Recall

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet  import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer


ap = argparse.ArgumentParser()
ap.add_argument("-t","--train", required=True, help="Path to training directory")
ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
ap.add_argument("-sm", "--saveModel", default="2016-model", help = "saved 2016 model")
# ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
args = vars(ap.parse_args())


EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
TESTING_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMG_SIZE = 256
SEED_VALUE = 30

#Decalre dataset for training split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SIZE)

#Don't know if I need further preprocessing here:
train_img_generator = train_datagen.flow_from_directory(
    directory = args["train"],
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    # color_mode = 'rgb',
    # subset = 'training'
)

img_validation_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=VALIDATION_SIZE)
#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = args["validation"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = 16,
    class_mode = 'binary',
    seed = SEED_VALUE,
)

inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

pretrained_resnet50 = keras.applications.ResNet101(include_top=False, weights='imagenet', input_tensor=inputTensor)

for layer in pretrained_resnet50.layers:
    layer.trainable = False
output = pretrained_resnet50.output
output = keras.layers.GlobalAveragePooling2D()(output)
output = keras.layers.Flatten()(output)
output = keras.layers.Dense(256, activation='relu')(output)
output = keras.layers.Dropout(0.25)(output)
# output = keras.layers.Dense(256, activation='relu')(output)
# output = keras.layers.Dropout(0.25)(output)
output = keras.layers.Dense(1, activation='sigmoid')(output)

pretrained_resnet50 = Model(inputs=pretrained_resnet50.input, outputs = output)

# pretrained_resnet50.summary()

#Define optimizer function
#Decay is used for smaller learning steps duing the last epochs
#see: https://pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/

adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
# sgd = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum=0.9)

print("Compile model:")
pretrained_resnet50.compile(
    optimizer = adam,
    loss="binary_crossentropy",
    metrics=['accuracy', Precision(), Recall(), tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)]
)


#Save predictions to csv file
csv_logger = CSVLogger(args["csvName"])

#Reduces LR when val_loss metric does not improve
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

early_stop = EarlyStopping(monitor="val_loss", patience=10)

#Inspired by: https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
print("Training model with Fit function")
history = pretrained_resnet50.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    callbacks = [csv_logger, reduce_lr, early_stop],
    validation_data=validation_img_generator,
    # validation_steps=2000
)

#Save model to folder
pretrained_resnet50.save(args["saveModel"])

#Plot model accuracy in graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']

epochs = range(len(acc))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend(['Train_acc','Val_acc'])
# plt.savefig("acc.pdf")
plt.figure() #Train accuracy

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
# plt.legend(['Train','Val']) #Train Loss
plt.legend(['loss','Val_loss']) #Train Loss

# plt.show()
# plt.savefig("loss.pdf")

plt.plot(epochs, precision)
plt.plot(epochs, val_precision)
plt.title('Training and validation precision')
plt.legend(['Precision','Val_precision']) #Train Loss
# plt.show()
# plt.savefig("precision.pdf")

plt.plot(epochs, recall)
plt.plot(epochs, val_recall)
plt.title('Training and validation recall')
plt.legend(['Train_acc','Val_acc', 'loss','Val_loss', 'Precision','Val_precision', 'Recall','Val_recall']) #Train Loss
# plt.show()
# plt.savefig("recall.pdf")
