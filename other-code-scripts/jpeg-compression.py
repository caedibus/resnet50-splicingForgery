import cv2
import os
import sys
import threading
import numpy as np
import argparse
from PIL import Image, ImageChops, ImageEnhance

# ap = argparse.ArgumentParser()
# ap.add_argument('--q', '--quality', type = int, default=90, help = "Decide quality of ELA")
# args = vars(ap.parse_args())

# img_path = 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\CASIA2.0-splicing-jpg\\Tp'
img_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-80-20-jpg\test\Tp'
dst_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\TEST-dataset\CASIA2-cmsf-jpegComp-quality80-TEST\compressed80\Tp'
# tmp_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\tmp-folder'
# ela_path = 'C:\\Users\\Malene\\OneDrive - NTNU\\Documents\\NTNU\\MasterThesis-2022\\Code-testing\\CASIA2.0-splicing-ELA\\Tp'

for filename in os.listdir(img_path):
    print(filename)
    # strip_name, ext = os.path.splitext(filename)
    # print("Strip_name ", strip_name)

    original_img = Image.open(os.path.join(img_path, filename))
    print("original_img")

    tmp_name = os.path.join(dst_path, filename)
    print("tmp_name ", tmp_name)
    # name, ext = os.path.splitext(filename)
    original_img.save(tmp_name, 'JPEG', quality = 80)
    tmp_img = Image.open(tmp_name)
