import cv2
import os
import imutils


# image = cv2.imread(r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\test\Tp\Tp_D_CNN_S_B_ind00018_ind00089_00007.tifela.jpg)
path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\image_path\train\authentic'
dest = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\resnet50-splicingForgery\image_path\fliped_images\flipped-1'

for file in os.listdir(path):
    original_img = cv2.imread(path+ '\\' + file)

    flipped_img = cv2.flip(original_img, -1)
    # cv2.imshow("flipped image",flipped_img)
    cv2.imwrite(dest + "\\" + file, flipped_img)
    cv2.waitKey(0)
    # print(file)



# rotate_img = imutils.rotate(image, angle=20)

# cv2.imshow(rotate_img)
# cv2.waitkey(0)
