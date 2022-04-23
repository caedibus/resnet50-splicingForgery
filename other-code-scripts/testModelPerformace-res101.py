import numpy as np
import matplotlib.pyplot as plt
import argparse
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
# ap.add_argument("-t","--training", required=True, help="Path to training directory")
ap.add_argument("-test","--testDirectory", default = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\test', help="Path to testing directory")
ap.add_argument("-e", "--epochs", type =int, default = 20, help ="Number of epochs for training")
ap.add_argument("-b", "--batchsize", type=int, default =32, help = "Number of batch size")
ap.add_argument("-fn", "--csvName", default='saved-output.csv', help ="Filename of csv output")
args = vars(ap.parse_args())

LOADED_MODEL = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\sm-test86'
IMG_SIZE = 224
SEED_VALUE = 1
EPOCHS = args["epochs"]
BATCH_SIZE = args["batchsize"]


img_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

#Declare dataset for validation split
testing_generator = img_generator.flow_from_directory(
    directory = args["testDirectory"],
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = SEED_VALUE,
    shuffle=False,
    )

model = keras.models.load_model(LOADED_MODEL)    #Loading entire pretrained model
# model = tf.keras.Model.load_weights('save_model2\\resnet101_weights1.h5')  #Loading weights from pretrained model

# model.summary()

csv_logger = CSVLogger(args["csvName"])

history = model.predict(
    testing_generator,
)


# model.fit(
#     validation_img_generator,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose = 1,
#     callbacks = [csv_logger],
# )
