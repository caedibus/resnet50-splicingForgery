import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import tensorflow as tf
import tensorflow_addons as tfa
import copy

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


ap = argparse.ArgumentParser()
ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
args = vars(ap.parse_args())

# VALIDATION_SIZE = 0.2
LOADED_MODEL = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\save_model'
IMG_SIZE = 224
VALIDATION_SIZE = 0.99
SEED_VALUE = 1
EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]


## FUNCTIONS
#Function returns labels of the predicted images
def predicted_label(pred, threshold):
    label = np.arange(len(pred))
    label[:] = pred[:,0]<threshold
    label[:] = pred[:,0]>=threshold
    return label

# Function for displaying wrongly classified images
def compare_labels(img, true_label, predicted_label):
    for i in range(len(true_label)):
        if true_label[i] == predicted_label[i]:
            print("Labels are equal")
        else:
            print("Labels are different")
            tmp_img = copy.copy(img[i])
            tmp_img2 = undo_preprocessing(tmp_img)
            plt.imshow(tmp_img2.astype('uint8'))
            plt.show()

def undo_preprocessing(x):
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    x = x[..., ::-1]
    # x = np.clip(x, 0, 255).astype('int')
    return x
# def restore_original_image_from_array(x, data_format='channels_first'):
    # mean = [103.939, 116.779, 123.68]
    #
    # # Zero-center by mean pixel
    # if data_format == 'channels_first':
    #     if x.ndim == 3:
    #         x[0, :, :] += mean[0]
    #         x[1, :, :] += mean[1]
    #         x[2, :, :] += mean[2]
    #     else:
    #         x[:, 0, :, :] += mean[0]
    #         x[:, 1, :, :] += mean[1]
    #         x[:, 2, :, :] += mean[2]
    # else:
    #     x[..., 0] += mean[0]
    #     x[..., 1] += mean[1]
    #     x[..., 2] += mean[2]
    #
    # if data_format == 'channels_first':
    #     # 'BGR'->'RGB'
    #     if x.ndim == 3:
    #         x = x[::-1, ...]
    #     else:
    #         x = x[:, ::-1, ...]
    # else:
    #     # 'BGR'->'RGB'
    #     x = x[..., ::-1]
    #
    # return x

img_validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SIZE,
    # rotation_range=30,
    # height_shift_range=0.2,
    # vertical_flip = True,
    # horizontal_flip=True
)

#Declare dataset for validation split
validation_img_generator = img_validation_generator.flow_from_directory(
    directory = args["training"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = 3,
    # batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    # color_mode = 'rgb',
    subset = 'validation'
    )

model = keras.models.load_model(LOADED_MODEL)    #Loading entire pretrained model
# model = tf.keras.Model.load_weights('save_model2\\resnet101_weights1.h5')  #Loading weights from pretrained model

# model.summary()

csv_logger = CSVLogger(args["csvName"])

# model.fit(
#     validation_img_generator,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose = 1,
#     callbacks = [csv_logger],
# )





pred_dataset = validation_img_generator
wrong_classified_img = dict(authentic=[], tampered=[])
print("Number of batches: ", len(pred_dataset), "\n")
for batchNum in range(len(pred_dataset)):
    print("Batch#:", batchNum)
    batch = pred_dataset[batchNum]

    # Gather images and true labels.
    batch_img = batch[0]
    batch_label = batch[1]
    print("Batch_True_Label:", batch[1])
    # print(type(batch[1][0]))

    # Perform prediction and calculate label.
    batch_prediction = model.predict(batch_img, use_multiprocessing=True)
    print("batc prediction", batch_prediction)
    pred_label = predicted_label(batch_prediction, 0.50)
    print("pred_label", pred_label)

    # Compare true and predicted labels.
    compare_labels(batch_img, batch_label, pred_label)
    print("")
