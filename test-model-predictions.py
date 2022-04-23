import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models
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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



ap = argparse.ArgumentParser()
# ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-test","--testDirectory", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest2\test', help="Path to testing directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
ap.add_argument("-l", "--loadModel", default='res101-test94', help ="loaded model")

args = vars(ap.parse_args())

LOADED_MODEL = args["loadModel"]
IMG_SIZE = 224
SEED_VALUE = 1
# EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]

f1_score = tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)

# def predicted_label(pred, threshold):
#     label = np.arange(len(pred))
#     label[:] = pred[:,0]<threshold
#     label[:] = pred[:,0]>=threshold
#     return label


img_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

#Declare dataset for validation split
testing_generator = img_generator.flow_from_directory(
    directory = args["testDirectory"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    seed = SEED_VALUE,
    shuffle=False
)

print("Loadedmodel:", LOADED_MODEL)
model = keras.models.load_model(LOADED_MODEL, custom_objects={'f1_score': f1_score})    #Loading entire pretrained model
# model = keras.models.load_model(LOADED_MODEL)
# model = tf.keras.Model.load_weights('save_model2\\resnet101_weights1.h5')  #Loading weights from pretrained model

# model.summary()

csv_logger = CSVLogger(args["csvName"])

history = model.predict_generator(
    testing_generator,
    verbose = 2,
    callbacks = [csv_logger]
)
print("Prediction completed")
#DeepLizard tutorial
# test_img, test_label = next(testing_generator)
# plots(test_img, titles=test_label)

print("\nHistory: ", history)

# Get predicte dclasses from model.fit()
# https://stackoverflow.com/questions/64622210/how-to-extract-classes-from-prefetched-dataset-in-tensorflow-for-confusion-matri
# predicted_class = np.argmax(history, axis = 1)
predicted_class = (history > 0.5).astype('int32')
print("\nPredicted class: ", predicted_class)
#extract correctly predicted classes
true_classes = testing_generator.classes
print("")
print("true_classes ", true_classes)

# print("Sets:",set(true_classes) - set(predicted_class))

# true_classes = pd.concat([y for x, y in testing_generator], axis = 0)
#Extract true labels
class_label = list(testing_generator.class_indices.keys())
print("")
print("Class_label ", class_label)


# f1_score = sklearn.metrics.f1_score(history, true_classes, zero_division=1)
#Define test report
test_report = classification_report(true_classes, predicted_class, target_names=class_label)
conf_matrix = confusion_matrix(true_classes, predicted_class)
print("\ntest report: ", test_report)
print("")
print(conf_matrix)

# conf_mat = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=testing_generator.classes)
conf_mat = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_label)
# conf_mat.plot()
plt.imshow(conf_matrix)
conf_mat.plot()
plt.show()
# history = np.argmax(history, axis = 1)
# print(classification_report(img_generator.labels, history,
#                             target_names=["class 1", "class 2"]))
# print(history)

# model.fit(
#     validation_img_generator,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose = 1,
#     callbacks = [csv_logger],
# )