import numpy as np
import matplotlib.pyplot as plt
# import os
# import cv2
import argparse
import tensorflow as tf

from tensorflow import keras
from keras.callbacks import CSVLogger
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

ap = argparse.ArgumentParser()
ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
# ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
args = vars(ap.parse_args())

#Defined path for training new weights for the top layer
resnet50_pretrained_weights_notop = 'C:\\Users\\Malene\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_pretrained_weights = 'C:\\Users\\Malene\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels.h5'

# training_path= 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\master_test\\image_path\\train'
# validation_path= 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\master_test\\image_path\\validation'

EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
TESTING_SIZE = 0.9
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
    subset = 'validation'
    )

#Future TODO when tweaking model:
callback_args = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3),
    # tf.keras.callbacks.ModelCheckpoint(filepath=)
]

#---------------------------------- MODEL --------------------------------

#Add resnet50 model to the sequential model
# inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
model = Sequential()
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    # weights = resnet50_pretrained_weights_notop
    )
)
#Adding a Dense layer with the two classes forged and valid images
# model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))   #Sigmoid is used for binary classification
model.layers[0].trainable=False
#The first layer is not trained with new weights from this project
model.summary()

#Define optimizer function
adam = tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6)
model.compile(
    optimizer = adam,
    loss="binary_crossentropy",
    # loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

#Save predictions to csv file
# tf.keras.callbacks.CSVLogger('output history.csv', separator=",", append=False)
csv_logger = CSVLogger('training.csv')

history = model.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    callbacks = [csv_logger],
    validation_data=validation_img_generator
)

model.summary()





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
