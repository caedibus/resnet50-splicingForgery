import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
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
ap.add_argument("-v","--validation", required=True, help="Path to validation directory")
args = vars(ap.parse_args())

#Defined path for training new weights for the top layer
resnet50_weights_path = 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\master_test\\resnet50_weights_tf_kernels_notop.h5'
training_path= 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\master_test\\image_path\\train'
validation_path= 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\master_test\\image_path\\validation'

EPOCHS = 20
BATCH_SIZE = 5
TESTING_SIZE = 0.9
VALIDATION_SIZE = 0.1
IMG_SIZE = 224


#Decalre dataset for training split
# img_generator = keras.preprocessing.image.ImageDataGenerator(
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#Don't know if I need further preprocessing here:
# train_img_generator = img_generator.flow_from_directory(
train_img_generator = train_datagen.flow_from_directory(
    directory = training_path,
    target_size=(224,224),
    batch_size = BATCH_SIZE,
    # class_mode = 'categorical',
    class_mode = 'binary',
    subset = 'training'
    # shuffle = True,
    )
# print(train_img_generator[0])

# img_validation_generator = keras.preprocessing.image.ImageDataGenerator(
img_validation_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = validation_path,
    target_size = (224,224),
    batch_size = BATCH_SIZE,
    # class_mode = 'categorical',
    class_mode = 'binary',
    subset = 'validation'
    # shuffle = True,
    )

#Future TODO when tweaking model:
callback_args = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3),
    # tf.keras.callbacks.ModelCheckpoint(filepath=)
]

#---------------------------------- MODEL --------------------------------


# model.add(tf.keras.layers.ZeroPadding2D((0,0), input_shape=(IMG_SIZE, IMG_SIZE, 3)))

# model.add(ResNet50(
# model = ResNet50(
#     # include_top=True,
#     include_top=False,  #removes the last dense layer in the pretrained model
#     weights="imagenet",
#     # input_tensor=inputTensor,
#     input_tensor=None,
#     input_shape=(224, 224, 3),
#     pooling='avg',
#     # classes=1,
# )
# ))
# output = model.layers[0].output
# output = keras.layers.Flatten()(output)
# model = Model(model.input, output = output)
#
# for layer in model.layers:
#     layer.trainable = False

# model.summary()

#Add resnet50 model to the sequential model
# inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
#Adding a Dense layer with the two classes forged and valid images
model.add(Dense(2, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable=False
#The first layer is not trained with new weights from this project
# model.layers[0].trainable = False
model.summary()

#Define optimizer function
adam = tf.keras.optimizers.Adam(lr = 0.01, decay = 1e-6)
model.compile(
    optimizer = adam,
    # loss="binary_crossentropy",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

model_history = model.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    validation_data=validation_img_generator
    #callbacks?
    # loss
)

model.summary()
