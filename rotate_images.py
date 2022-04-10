import cv2
import os
import imutils


image = cv2.imread(r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\test\Tp\Tp_D_CNN_S_B_ind00018_ind00089_00007.tifela.jpg)

rotate_img = imutils.rotate(image, angle=20)

cv2.imshow(rotate_img)
cv2.waitkey(0)
