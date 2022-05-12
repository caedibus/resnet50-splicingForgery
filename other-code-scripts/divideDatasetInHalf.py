import os
from PIL import Image
import cv2

path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-jpg\test\Tp'
destination1 = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-80-20-jpg\test\Tp'
destination2 = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-80-20-jpg\validation\Tp'

# n = 0
for n, image in enumerate(os.listdir(path)):
    print(n, image,type(image))
    img = cv2.imread(path +'\\'+ image)
    if n % 2 == 0:
        cv2.imwrite(destination1 + '\\' + image, img)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # print("Partall")
    else:
        cv2.imwrite(destination2 + '\\' + image, img)
        # print("oddetall")
    # n += 1
