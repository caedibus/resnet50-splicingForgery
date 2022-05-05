import fnmatch
import os

path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\Old_datasets\CASIA2.0_revised\Tp'

for file in os.listdir(path):
    if fnmatch.fnmatch(path + '\\' +file, '*.tif'):
        print("tif image is file name:". file)
