import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-80-20-jpg\test\Au'
destination = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\TEST-dataset\CASIA2-cmsf-brighteningTEST\Brightened-jpg\Au'

for object in os.listdir(path):
    print(path +'\\'+object)
    img = cv2.imread(path+'\\'+object)
    tmp_img = np.zeros(img.shape, img.dtype)
    alpha = 1.3
    beta = 30
    tmp_img = cv2.convertScaleAbs(img, alpha=alpha, beta= beta)
    cv2.imwrite(destination +'\\'+ object, tmp_img)
