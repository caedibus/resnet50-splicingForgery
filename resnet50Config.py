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
ap.add_argument("-fn", "--csvName", default='saved-', help ="Filename of csv output")
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

#TODO when tweaking model:
callback_args = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3),
    # tf.keras.callbacks.ModelCheckpoint(filepath=)
]

#----------------------------------Base model --------------------------------
inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# pretrained_resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input(shape=(IMG_SIZE,IMG_SIZE,3)))
pretrained_resnet_model = ResNet101(weights='imagenet', include_top=False, input_tensor=keras.Input(shape=(IMG_SIZE,IMG_SIZE,3)))
pretrained_resnet_model.trainable = False
for layer in pretrained_resnet_model.layers:  #[:-1]:
    layer.trainable = False
# for layer in pretrained_resnet_model.layers[0:143]:  #[:-1]:
output = pretrained_resnet_model.layers[-1].output
output = keras.layers.Flatten()(output)








out_model = Model(inputs = pretrained_resnet_model.input, outputs = output)

#Train the last stage of ResNet50

# for layer in pretrained_resnet_model.layers[143:]:
#   layer.trainable = True

out_model.summary()
print(type(out_model))

# out_model = pretrained_resnet_model.output
out_model = layers.GlobalAveragePooling2D()(out_model)
out_model = layers.Flatten()(out_model)
out_model = layers.Dense(2048, activation='relu')(out_model)
out_model = layers.Dropout(.4)(out_model)
out_model = layers.Dense(1024, activation='relu')(out_model)

out_model.summary()
#Averagepooling layer
# out_model = AveragePooling2D(pool_size=(7,7))(out_model)
# out_model = Flatten(name="flatten")(out_model)
# out_model = Flatten()(out_model)

#Fully connected layer with 256 neurons
out_model = layers.Dense(256, activation='relu')(out_model)
#Fully connected layer with 1 neuron to predict forged or valid image
prediction = layers.Dense(1, activation='sigmoid')(out_model)
# prediction = layers.Dense(1, activation='softmax')(out_model)

# TODO: adding early stoping condition
model = Model(inputs=pretrained_resnet_model.input, outputs=prediction)
model.summary()


#---------------------------------- Sequential MODEL --------------------------------


#Add resnet50 model to the sequential model
# inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))#dECLARE RESIZING OF IMAGES
#
# model = Sequential()
# model.add(ResNet101(
#     include_top=False,
#     pooling='avg',
#     # pooling='max',
#     weights='imagenet',
#     input_tensor = inputTensor,     #TODO figure out if i need this line
#     # weights = resnet50_pretrained_weights_notop
#     )
# )
#
# for layer in model.layers:
#     layer.trainable = False
#
# #Adding a Dense layer with the two classes forged and valid images
# # model.add(Dense(2, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))   #Sigmoid is used for binary classification
# model.layers[0].trainable=False     ## TODO: Probably remove thsi line
# #The first layer is not trained with new weights from this project
# model.summary()

#Define optimizer function
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = 1e-6)
sgd = tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum=0.9)
model.compile(
    optimizer = adam,
    # loss="binary_crossentropy",
    loss="categorical_crossentropy",
    # loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

#Save predictions to csv file
# tf.keras.callbacks.CSVLogger('output history.csv', separator=",", append=False)
csv_logger = CSVLogger(args["csvName"])



#Inspired by: https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
#fit_generator is used for big datasets that does not fit in to memory
history = model.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    callbacks = [csv_logger],
    validation_data=(validation_img_generator),
    # validation_steps=2000
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
