import cv2
import os
import imutils


# image = cv2.imread(r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-trainValTest\test\Tp\Tp_D_CNN_S_B_ind00018_ind00089_00007.tifela.jpg)
path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-80-20-jpg\test\Au'
dest = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\TEST-dataset\CASIA2-cmsf-rotateVerticallyANDupsideDownTEST\rotateVertically-0\Au'

for file in os.listdir(path):
    original_img = cv2.imread(path+ '\\' + file)
    #Flip image upside down
    # flipped_img = cv2.flip(original_img, -1)

    #Flip image 33 degrees
    flipped_img = imutils.rotate_bound(original_img, 45)
    # cv2.imshow("flipped image",flipped_img)
    cv2.imwrite(dest + "\\" + file, flipped_img)
    # cv2.waitKey(0)



# rotate_img = imutils.rotate(image, angle=20)

# cv2.imshow(rotate_img)
# cv2.waitkey(0)
