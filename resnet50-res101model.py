import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from imutils import paths
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications.resnet  import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputLayer

from sklearn.utils import compute_class_weight

ap = argparse.ArgumentParser()
ap.add_argument("-t","--training", default=r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\train', help="Path to training directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='res50asRes101-saved-output.csv', help ="Filename of csv output")
ap.add_argument("-sm", "--saveModel", default='save_model-res50asRes101', help ="saved model output")
ap.add_argument("-v","--validation", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\validation', help="Path to validation directory")
ap.add_argument("-test","--testDirectory", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\test', help="Path to testing directory")

args = vars(ap.parse_args())

EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]
TESTING_SIZE = 0.75
VALIDATION_SIZE = 0.1
IMG_SIZE = 224
SEED_VALUE = 30
lenValidation = len(list(paths.list_images(args["validation"])))
print("lenValidation ", lenValidation)

#Decalre dataset for training split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # validation_split=VALIDATION_SIZE,
    # rotation_range=30,
    height_shift_range=0.2,
    width_shift_range=0.2,
    # vertical_flip = True,
    # horizontal_flip=True
)

#Don't know if I need further preprocessing here:
train_img_generator = train_datagen.flow_from_directory(
    directory = args["training"],
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    # color_mode = 'rgb',
    # subset = 'training'
)

img_validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # validation_split=VALIDATION_SIZE,
    rotation_range=30,
    height_shift_range=0.2,
    vertical_flip = True,
    horizontal_flip=True
)

#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = args["validation"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = 10,
    # batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    # subset = 'validation',
)

testing_generator = img_validation_generator.flow_from_directory(
    directory = args["testDirectory"],
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
)

inputTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

pretrained_resnet50 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputTensor)

for layer in pretrained_resnet50.layers:
    layer.trainable = False

output = pretrained_resnet50.output
output = keras.layers.GlobalAveragePooling2D()(output)
output = keras.layers.Flatten()(output)
output = keras.layers.Dense(1024, activation='relu',  kernel_regularizer=regularizers.l2(0.003))(output)
output = keras.layers.Dropout(0.15)(output)
output = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.003))(output)
output = keras.layers.Dropout(0.25)(output)
output = keras.layers.Dense(1, activation='sigmoid')(output)  #Only use softmax for categorical class_mode
pretrained_resnet50 = Model(inputs=pretrained_resnet50.input, outputs = output)


#see: https://pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
epochNumb = args["epochs"]
# adam = tf.keras.optimizers.Adam(learning_rate = 0.001, decay=0.001/epochNum)
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
sgd = tf.keras.optimizers.SGD(learning_rate = 0.001)#0, decay = 0.0001)


# pretrained_resnet50.summary()

print("Compile model:")
pretrained_resnet50.compile(
    optimizer = sgd,
    loss="binary_crossentropy",
    metrics=['accuracy', Precision(), Recall(), tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)])

#Save predictions to csv file
csv_logger = CSVLogger(args["csvName"])

#Inspired by: https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/

#Define early stopping condition
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=50)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=0.001)

#https://stackoverflow.com/questions/49031309/valueerror-class-weight-must-contain-all-classes-in-the-data-the-classes-1
# class_weight = {0:3, 1:2}
# class_weight = {0:1, 1:0.6}
class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(train_img_generator.classes), y = train_img_generator.classes)
class_weight = dict(zip(np.unique(train_img_generator.classes), class_weight))

history = pretrained_resnet50.fit(
    train_img_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose = 1,
    class_weight=class_weight,
    callbacks = [csv_logger, reduce_lr, early_stopping],  #early_stopping
    validation_data=validation_img_generator,
    # validation_steps = lenValidation,
    # validation_steps=len(list(paths.list_images(args["validation"])))
)

# pretrained_resnet50.summary()



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


pretrained_resnet50.save(args["saveModel"])
