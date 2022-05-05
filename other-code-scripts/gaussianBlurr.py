import cv2
import os
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# image path
path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-jpg\test\Tp'
# path =
dest_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-cmsf-gaussianBlurrTEST\Tp'

for file in os.listdir(path):
    #check if file is png image
    # if file.endswith(".jpg"):
        #Read png image
    img = cv2.imread(path + '/' +str(file))
    # print(file)
    # cv2.imshow('Original image before blurring',img)
    # cv2.waitKey(0)
    # #Blur image
    blurring = cv2.GaussianBlur(img,( 5 , 5 ), 0 )
    #
    # cv2.imshow('Gaussian Blurring' , blurring)
    # cv2.waitKey(0);
    cv2.imwrite(dest_path + '\\' + str(file), blurring)
    cv2.destroyAllWindows();
    # cv2.waitKey(0)
    # break
    #
















# using imread()
# for images in os.listdir(path):
# input = os.path.join(path,images)

# for directories, images in os.walk(path):
#     for image in images:
#         img = cv2.imread(path)
#         dst = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
#         cv2.imshow('image', numpy.hstack((img, dst)))
#         cv2.waitKey(0);
#         cv2.destroyAllWindows();
#         cv2.waitKey(1)
#
# def gaussianBlur(path):
#         img = cv2.imread(path)
#         dst = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
#         cv2.imshow('image', numpy.hstack((img, dst)))
#         cv2.waitKey(0);
#         cv2.destroyAllWindows();
#         cv2.waitKey(1)
#
# if __name__ == "__main__":
#     image_paths = get_image_paths(".");
#     # print(json.dumps(image_paths, indent=4))
#
#     # Display all images inside image_paths
#     for image_path in image_paths:
#         img = show_images(image_path);
#         # gaussianBlur(img)
#         print('\n')



    # new_path = os.path.join(dest_path, 'blurred'+images)
