import os

# base_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2-NEW-trainValTest-jpg\test\Au'
base_path = r'C:\Users\Malene\OneDrive - NTNU\Documents\NTNU\MasterThesis-2022\Code-testing\CASIA2.0-CM-splicing-jpg\Au'

i = 0
string = 'tmp'
for file in os.listdir(base_path):
    # print("Im n a loop")
    # i +=1
    # print(i)
    if string in file:
        print("Found tmp ")
        print(file)
        os.remove(os.path.join(base_path, file))
