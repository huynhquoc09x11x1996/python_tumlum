import os
import cv2
import numpy as np

static_path ='./images'

def getDanhSachFolderName():
    return os.listdir(static_path)



def getDanhSachFiles(f_n):
    return os.listdir(static_path+"/"+f_n)


def getListAbsolutePath_ofFolder(folder_name):
    list_path=[]
    files=getDanhSachFiles(folder_name)
    for x in range(len(files)):
        list_path.append(static_path+"/"+folder_name+"/"+files[x])
    return list_path

# try:
#     print(getListAbsolutePath_ofFolder('_rooster'))
# except:
#     print("ten folder bi sai!")
