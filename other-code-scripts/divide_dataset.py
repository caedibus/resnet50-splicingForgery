import os
from imutils import paths
import random
import shutil

#Path to original dataset
original_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2.0-CM-splicing-jpg'
#path to where edivided dataset is located
BASE_PATH = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest-jpg'

#define the derived folders for train, validation and testing training_directory

TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

train_split = 0.8
val_split = 0.2

imagePath = list(paths.list_images(original_path))
random.seed(42)
random.shuffle(imagePath)

#Define training split
i = int(len(imagePath) * train_split)
train_path = imagePath[:i]
testPath = imagePath[i:]


#Define validation split
i = int(len(train_path) * val_split)
val_path = train_path[:i]
train_path = train_path[i:]


dataset = [
    ('train', train_path, TRAIN_PATH), ('test', testPath, TEST_PATH), ('validation', val_path, VAL_PATH)
]


for (datatype, imagePaths, baseOutput) in dataset:
    for inputPath in imagePaths:
        filename = inputPath.split(os.path.sep)[-1]
        # print("Filename[-1]: ", filename)
        label = inputPath.split(os.path.sep)[-2]
        # print("Label: ", label)
        labelPath = os.path.sep.join([baseOutput, label])
        # print("label path: ", labelPath)
        dest = os.path.sep.join([labelPath, filename])
        shutil.copy(inputPath, dest)
