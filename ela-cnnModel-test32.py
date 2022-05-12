import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.utils import compute_class_weight

import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer, Conv2D, Activation, Flatten, Dropout, Dense, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.metrics import Precision, Recall


ap = argparse.ArgumentParser()
ap.add_argument("-t","--train", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-ELA-quality90\train', help="Path to training directory")
ap.add_argument("-v","--validation", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-ELA-quality90\validation', help="Path to validation directory")
ap.add_argument("-e", "--epochs", type =int, default = 100, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =16, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='elaCNN-saved-output.csv', help ="Filename of csv output")
ap.add_argument("-sm", "--saveModel", default='res101-save_model2', help ="saved model output")
args = vars(ap.parse_args())


EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
IMG_SIZE = 256
SEED_VALUE = 30

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    args["train"],
    batch_size = BATCH_SIZE,
    image_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'rgb',
    label_mode = 'binary',
    seed = 1,
    subset="training",
    validation_split = 0.2,
    labels="inferred"
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # args["validation"],
    args["train"],
    subset="validation",
    validation_split = 0.2,
    batch_size = BATCH_SIZE,
    image_size = (IMG_SIZE,IMG_SIZE),
    color_mode = 'rgb',
    label_mode = 'binary',
    seed = 1,
    labels="inferred"
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32 , kernel_size=(3,3), activation ='relu')) #input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(Dropout(0.25))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(32 , kernel_size=(3,3), activation ='relu')) #input_shape=(IMG_SIZE,IMG_SIZE,3)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

#SGD from the 2016-paper
# sgd = tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum=0.99, decay= 0.0001)
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.0001,
#     decay_steps=10000,
#     decay_rate=0.9)

sgd = tf.keras.optimizers.SGD(learning_rate = 0.00001)
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)


# sgd = tf.keras.optimizers.SGD(learning_rate = 0.000001) #, momentum=0.99, decay= 0.0003)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=0.001)


model.compile(
    optimizer=adam,
    loss = 'binary_crossentropy',
    metrics=['accuracy', Precision(), Recall(), tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)]
    # metrics = 'accuracy'
)


# class_label = ["Au", "Tp"]

# class_label = np.concatenate([y for x, y in train_dataset], axis=-1)
# print("\nclass label: ",class_label)

csv_logger = CSVLogger(args["csvName"])
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=30)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=0.001)

# class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(class_label), y = class_label)
# class_weight = dict(zip(np.unique(class_label), class_weight))

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    # validation_split=0.2,
    # class_weight=class_weight,
    callbacks = [csv_logger, early_stopping],
    validation_data=validation_dataset,
)

# outFile = open("historyOutTest.txt", "a")
# for line in history:
#     strOut = str(line)
#     outFile.write(strOut)
# outFile.close()

model.save(args["saveModel"])

#Plot model accuracy in graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']
#
# epochs = range(len(acc))
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy')
# plt.legend(['Train_acc','Val_acc'])
# # plt.savefig("acc.pdf")
# plt.figure() #Train accuracy
#
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss')
# # plt.legend(['Train','Val']) #Train Loss
# plt.legend(['loss','Val_loss']) #Train Loss
#
# # plt.show()
# # plt.savefig("loss.pdf")
#
# plt.plot(epochs, precision)
# plt.plot(epochs, val_precision)
# plt.title('Training and validation precision')
# plt.legend(['Precision','Val_precision']) #Train Loss
# plt.show()
# plt.savefig("precision.pdf")

# plt.plot(EPOCHS, recall)
# # plt.plot(EPOCHS, val_recall)
# plt.title('Training and validation recall')
# plt.legend(['Train_acc','Val_acc', 'loss','Val_loss', 'Precision','Val_precision', 'Recall','Val_recall']) #Train Loss
# plt.show()
# plt.savefig("recall.pdf")
